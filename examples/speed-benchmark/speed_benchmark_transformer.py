# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
import importlib.util


def is_module_installed(module_name):
    module_spec = importlib.util.find_spec(module_name)
    return module_spec is not None


class SpeedBenchmarkTransformer:

    SEED = 1024
    BATCH_SIZE = 1
    USE_FLASH_ATTN = True
    COMMENT = 'default'
    DEVICE_MAP = 'auto'
    TORCH_DTYPE = 'auto'
    GENERATE_LENGTH_PER_EXPERIMENT = 2048
    OVERWRITE_RESULT = False
    DUMMY_INPUT = '我'

    def __init__(self, model_id, use_modelscope: bool = True, outputs_dir: str = 'outputs/transformer'):
        """
        Speed benchmark for transformer(pt) inference.

        Args:
            model_id: The model id on ModelScope or HuggingFace hub.
            use_modelscope: Use ModelScope, otherwise HuggingFace.
            outputs_dir: The output directory. Default is 'outputs/transformer'.
        """

        set_seed(self.SEED)
        self.model_id = model_id
        self.outputs_dir = outputs_dir

        if use_modelscope:
            if not is_module_installed('modelscope'):
                raise ImportError("Please install modelscope: pip install modelscope[framework]")
            from modelscope import snapshot_download
        else:
            if not is_module_installed('huggingface_hub'):
                raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
            from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        attn_impl = 'flash_attention_2' if self.USE_FLASH_ATTN else 'eager'
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          torch_dtype=self.TORCH_DTYPE,
                                                          device_map=self.DEVICE_MAP,
                                                          attn_implementation=attn_impl
                                                          ).eval()

        self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    def run(self, context_length: int) -> str:

        # Specify hyperparameters for generation
        self.generation_config.min_length = self.GENERATE_LENGTH_PER_EXPERIMENT + context_length
        self.generation_config.max_new_tokens = self.GENERATE_LENGTH_PER_EXPERIMENT

        # Prepare inputs
        batch_size = self.BATCH_SIZE
        context_str = self.DUMMY_INPUT * context_length
        inputs = self.tokenizer([context_str for _ in range(batch_size)], return_tensors='pt')
        assert inputs['input_ids'].shape[1] == context_length
        assert inputs['input_ids'].shape[0] == batch_size
        inputs = inputs.to(self.model.device)

        # Run inference
        start_time = time.time()
        torch.cuda.synchronize()
        pred = self.model.generate(**inputs, generation_config=self.generation_config)
        torch.cuda.synchronize()
        time_cost = time.time() - start_time
        assert pred.shape[1] == self.generation_config.min_length
        m = 0
        max_gpu_memory_cost = 0
        for i in range(torch.cuda.device_count()):
            m += torch.cuda.max_memory_allocated(i)
        max_gpu_memory_cost = max(max_gpu_memory_cost, m)
        torch.cuda.empty_cache()

        # Prepare results
        tokens_per_second: float = self.GENERATE_LENGTH_PER_EXPERIMENT / time_cost
        # Compute the maximum GPU memory cost (in GB)
        max_gpu_memory_cost_gb = max_gpu_memory_cost / 1024 / 1024 / 1024

        data = {
            "model_id": self.model_id,
            "batch_size": batch_size,
            "context_length_per_experiment": context_length,
            "generate_length_per_experiment": self.GENERATE_LENGTH_PER_EXPERIMENT,
            "use_flash_attn": self.USE_FLASH_ATTN,
            "comment": self.COMMENT,
            "tokens_per_second": tokens_per_second,
            "max_gpu_memory_cost_gb": max_gpu_memory_cost_gb
        }

        # Dump results to CSV file
        out_file: str = os.path.join(self.outputs_dir,
                                     f"{self.model_id.replace('/', '__')}-context_length-{context_length}.csv")
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        self.save_result(data, out_file)

        return out_file

    @staticmethod
    def save_result(data: dict, out_file: str) -> None:
        df = pd.DataFrame([data])
        df.to_csv(out_file, index=False)
        print(f"Results saved to {out_file}")


def main():

    import argparse

    # Parse args
    parser = argparse.ArgumentParser(description='Speed benchmark for transformer(pt) inference')
    parser.add_argument('--model_id', type=str, help='The model id on ModelScope or HuggingFace hub')
    parser.add_argument('--input_len', type=int, help='The input length for each experiment')
    parser.add_argument('--gpus', type=str, help='gpus, e.g. 0,1,2,3, or 4,5')
    parser.add_argument('--use_modelscope', type=bool, default=True, help='Use ModelScope, otherwise HuggingFace')
    parser.add_argument('--outputs_dir', type=str, default='outputs/transformer', help='The output directory')

    args = parser.parse_args()

    model_id: str = args.model_id
    envs: str = args.gpus
    in_length: int = args.input_len
    use_modelscope: bool = args.use_modelscope
    outputs_dir: str = args.outputs_dir

    print(f"Set CUDA_VISIBLE_DEVICES={envs} for model {model_id} with input_length {in_length}")
    os.environ["CUDA_VISIBLE_DEVICES"] = envs

    speed_benchmark = SpeedBenchmarkTransformer(model_id=model_id,
                                                use_modelscope=use_modelscope,
                                                outputs_dir=outputs_dir)
    speed_benchmark.run(context_length=in_length)


if __name__ == '__main__':
    main()
