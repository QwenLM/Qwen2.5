# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen2.5 Speed Benchmark for transformer(pt) inference.
"""

import os
import time
import json
import csv

import torch
from transformers.trainer_utils import set_seed


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

    def __init__(self, model_id_or_path, use_modelscope: bool = True, outputs_dir: str = 'outputs/transformer'):
        """
        Speed benchmark for transformer(pt) inference.

        Args:
            model_id_or_path: The model id on ModelScope or HuggingFace hub, or local model path.
            use_modelscope: Use ModelScope, otherwise HuggingFace.
            outputs_dir: The output directory. Default is 'outputs/transformer'.
        """

        set_seed(self.SEED)
        self.model_id_or_path = model_id_or_path
        self.outputs_dir = outputs_dir

        if use_modelscope:
            from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        attn_impl = 'flash_attention_2' if self.USE_FLASH_ATTN else 'eager'
        self.model = AutoModelForCausalLM.from_pretrained(model_id_or_path,
                                                          torch_dtype=self.TORCH_DTYPE,
                                                          device_map=self.DEVICE_MAP,
                                                          attn_implementation=attn_impl
                                                          ).eval()

        self.generation_config = GenerationConfig.from_pretrained(model_id_or_path, trust_remote_code=True)

    def run(self, context_length: int) -> str:

        # Specify hyperparameters for generation
        self.generation_config.min_length = self.GENERATE_LENGTH_PER_EXPERIMENT + context_length
        self.generation_config.max_new_tokens = self.GENERATE_LENGTH_PER_EXPERIMENT
        print(f'Generation config: {self.generation_config}')

        # Prepare inputs
        batch_size = self.BATCH_SIZE
        context_str = self.DUMMY_INPUT * context_length
        inputs = self.tokenizer([context_str for _ in range(batch_size)], return_tensors='pt')
        assert inputs['input_ids'].shape[1] == context_length
        assert inputs['input_ids'].shape[0] == batch_size
        inputs = inputs.to(self.model.device)

        # Run inference
        print(f'Start running inference for model {self.model_id_or_path} with input length {context_length} ...')
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
            "model_id_or_path": self.model_id_or_path,
            "batch_size": batch_size,
            "context_length_per_experiment": context_length,
            "generate_length_per_experiment": self.GENERATE_LENGTH_PER_EXPERIMENT,
            "use_flash_attn": self.USE_FLASH_ATTN,
            "comment": self.COMMENT,
            "tokens_per_second": round(tokens_per_second, 4),
            "max_gpu_memory_cost_gb": round(max_gpu_memory_cost_gb, 4),
        }
        data_json = json.dumps(data, indent=4, ensure_ascii=False)
        print(f'**Final result **\n{data_json}\n')

        # Dump results to CSV file
        from datetime import datetime
        now = datetime.now()
        timestamp: str = now.strftime("%m%d%H%M%S")

        model_id_or_path_str = self.model_id_or_path.split(os.sep)[-1] \
            if os.path.isdir(self.model_id_or_path) else self.model_id_or_path.replace('/', '__')

        out_file: str = os.path.join(self.outputs_dir,
                                     f"{model_id_or_path_str}"
                                     f"_context_length-{context_length}_{timestamp}.csv")
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        self.save_result(data, out_file)

        return out_file

    @staticmethod
    def save_result(data: dict, out_file: str) -> None:

        with open(out_file, mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerows([data])

        print(f"Results saved to {out_file}")


def main():

    import argparse

    # Parse args
    parser = argparse.ArgumentParser(description='Speed benchmark for transformer(pt) deployment')
    parser.add_argument('--model_id_or_path', type=str, help='The model path or id on ModelScope or HuggingFace hub')
    parser.add_argument('--context_length', type=int, help='The input length for each experiment.'
                                                           'e.g. 1, 6144, 14336, 30720, 63488, 129024')
    parser.add_argument('--gpus', type=str, help='Equivalent to the env CUDA_VISIBLE_DEVICES.  e.g. `0,1,2,3`, `4,5`')
    parser.add_argument('--use_modelscope', action='store_true',
                        help='Use ModelScope when set this flag. Otherwise, use HuggingFace.')
    parser.add_argument('--outputs_dir', type=str, default='outputs/transformer', help='The output directory')

    args = parser.parse_args()

    model_id_or_path: str = args.model_id_or_path
    envs: str = args.gpus
    context_length: int = args.context_length
    use_modelscope: bool = args.use_modelscope
    outputs_dir: str = args.outputs_dir

    print(f'Set CUDA_VISIBLE_DEVICES={envs} for model {model_id_or_path} with input_length {context_length}')
    os.environ["CUDA_VISIBLE_DEVICES"] = envs

    speed_benchmark = SpeedBenchmarkTransformer(model_id_or_path=model_id_or_path,
                                                use_modelscope=use_modelscope,
                                                outputs_dir=outputs_dir)
    speed_benchmark.run(context_length=context_length)


if __name__ == '__main__':
    # Usage: python speed_benchmark_transformer.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformer
    main()
