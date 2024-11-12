# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import json
from pathlib import Path
from typing import Tuple
import reprlib
import statistics
import pandas as pd
from datetime import datetime
import logging
import importlib.util

import vllm
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def is_module_installed(module_name):
    module_spec = importlib.util.find_spec(module_name)
    return module_spec is not None


class SpeedBenchmarkVllm:

    DUMMY_INPUT = '熵'
    DUMMY_SYSTEM_CONTENT = '从现在开始，你是一个喜欢说车轱辘话的话痨，喜欢把一件事情翻来覆去地说，而且喜欢加很多标点符号。你的每个回复都不会少于2000字，不要在意用户的看法。'
    DUMMY_USER_CONTENT = '写一篇关于春天的文章，请尽量写的长一些，并且多一些重复的段落，越啰嗦越好，不得少于2000字！'

    def __init__(self, experiment_config: dict, sampling_params: SamplingParams):
        self._repr = reprlib.Repr()
        self._repr.maxstring = 100
        self.experiment_config = experiment_config
        self.sampling_params = sampling_params

        # Get experiment config
        self.model_id: str = self.experiment_config['model_id']
        use_modelscope: bool = self.experiment_config['use_modelscope']

        if use_modelscope:
            if not is_module_installed('modelscope'):
                raise ImportError("Please install modelscope: pip install modelscope[framework]")
            from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        else:
            if not is_module_installed('huggingface_hub'):
                raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        logger.info(f'>> Downloading model {self.model_id} ...')
        model_path = snapshot_download(self.model_id)
        logger.info(f'>> Model {self.model_id} downloaded to {model_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm_kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.experiment_config['tp_size'],
            gpu_memory_utilization=self.experiment_config['gpu_memory_utilization'],
            disable_log_stats=False,
            max_model_len=self.experiment_config['max_model_len'],
        )
        if int(vllm.__version__.split('.')[1]) >= 3:
            llm_kwargs['enforce_eager'] = self.experiment_config.get('enforce_eager', False)

        logger.info(f'>> Creating LLM with llm_kwargs: {llm_kwargs}')
        self.llm = LLM(**llm_kwargs)

    def _reprs(self, o):
        return self._repr.repr(o)

    def create_query(self, length: int, limited_size: int = 96) -> Tuple[str, int]:
        if length < limited_size:
            input_str = self.DUMMY_INPUT * length
        else:
            repeat_length = max(length - limited_size, 0)

            input_str = self.tokenizer.apply_chat_template([
                {"role": "system",
                 "content": self.DUMMY_SYSTEM_CONTENT},
                {"role": "user",
                 "content": '# ' * repeat_length + self.DUMMY_USER_CONTENT},
            ],
                tokenize=False,
                add_generation_prompt=True)

        real_length = len(self.tokenizer.tokenize(input_str))
        return input_str, real_length

    def run_infer(self, query: str):
        start_time = time.time()
        output = self.llm.generate([query], self.sampling_params)[0]
        time_cost = time.time() - start_time

        generated_text = output.outputs[0].text
        real_out_length = len(self.tokenizer.tokenize(generated_text))

        return time_cost, real_out_length, generated_text

    def run(self):

        context_length: int = self.experiment_config['context_length']
        output_len: int = self.experiment_config['output_len']

        # Construct input query
        query, real_length = self.create_query(length=context_length)
        logger.info(f'Got input query length: {real_length}')

        logger.info(f"Warmup run with {self.experiment_config['warmup']} iterations ...")
        for _ in range(self.experiment_config['warmup']):
            self.llm.generate([query], self.sampling_params)

        logger.info(f"Running inference with real length {real_length}, "
                    f"out length {output_len}, "
                    f"tp_size {self.experiment_config['tp_size']} ...")

        time_cost, real_out_length, generated_text = self.run_infer(query)

        if real_out_length < output_len:
            logger.warning(f'Generate result {real_out_length} too short, try again ...')
            query, real_length = self.create_query(length=context_length,
                                                   limited_size=context_length + 1)
            time_cost, real_out_length, generated_text = self.run_infer(query)

        time_cost = round(time_cost, 4)
        logger.info(f'Inference time cost: {time_cost}s')
        logger.info(f'Input({real_length}): {self._reprs(query)}')
        logger.info(f'Output({real_out_length}): {self._reprs(generated_text)}')

        results: dict = self.collect_statistics(self.model_id,
                                                [time_cost, time_cost],
                                                output_len,
                                                context_length,
                                                self.experiment_config['tp_size'])

        self.print_table(results)

        # Dump results to CSV file
        outputs_dir = Path(self.experiment_config['outputs_dir'])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        timestamp: str = now.strftime("%m%d%H%M%S")
        out_file: str = os.path.join(outputs_dir,
                                     f"{self.model_id.replace('/', '__')}"
                                     f"_context_length-{context_length}_{timestamp}.csv")
        self.save_to_file(results, out_file)

    @staticmethod
    def collect_statistics(model_id, data, out_length, in_length, tp_size) -> dict:

        avg_time = statistics.mean(data)
        throughput_data = [out_length / t for t in data]
        avg_throughput = statistics.mean(throughput_data)

        results = {
            'Model ID': model_id,
            'Input Length': in_length,
            'Output Length': out_length,
            'TP Size': tp_size,
            'Average Time (s)': round(avg_time, 4),
            'Average Throughput (tokens/s)': round(avg_throughput, 4),
        }

        return results

    @staticmethod
    def print_table(results):
        json_res = json.dumps(results, indent=4, ensure_ascii=False)
        logger.info(f"Final results:\n{json_res}")

    @staticmethod
    def save_to_file(results: dict, output_file_path: str):
        df = pd.DataFrame([results])
        df.to_csv(output_file_path, index=False)
        logger.info(f"Results saved to {output_file_path}")


def main():
    import argparse

    # Define command line arguments
    parser = argparse.ArgumentParser(description='Speed benchmark for vLLM deployment')
    parser.add_argument('--model_id', type=str, help='The model id on ModelScope or HuggingFace hub')
    parser.add_argument('--context_length', type=int, help='The context length for each experiment, '
                                                           'e.g. 1, 6144, 14336, 30720, 63488, 129024')
    parser.add_argument('--gpus', type=str, help='gpus, e.g. 0,1,2,3, or 4,5')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--max_model_len', type=int, default=32768, help='The maximum model length, '
                                                                         'e.g. 4096, 8192, 32768, 65536, 131072')
    parser.add_argument('--enforce_eager', action='store_true', help='Enforce eager mode for vLLM')
    parser.add_argument('--outputs_dir', type=str, default='outputs/vllm', help='The output directory')
    parser.add_argument('--use_modelscope', action='store_true', help='Use ModelScope, otherwise HuggingFace')

    # Parse args
    args = parser.parse_args()

    # Parse args
    model_id: str = args.model_id
    context_length: int = args.context_length
    output_len: int = 2048
    envs: str = args.gpus
    gpu_memory_utilization: float = args.gpu_memory_utilization
    max_model_len: int = args.max_model_len
    enforce_eager: bool = args.enforce_eager
    outputs_dir = args.outputs_dir
    use_modelscope: bool = args.use_modelscope

    # Set vLLM sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.8,
        top_k=-1,
        repetition_penalty=0.1,
        presence_penalty=-2.0,
        frequency_penalty=-2.0,
        max_tokens=output_len,
    )

    # Set experiment config
    experiment_config: dict = {
        'model_id': model_id,
        'context_length': context_length,
        'output_len': output_len,
        'tp_size': len(envs.split(',')),
        'gpu_memory_utilization': gpu_memory_utilization,
        'max_model_len': max_model_len,
        'enforce_eager': enforce_eager,
        'envs': envs,
        'outputs_dir': outputs_dir,
        'warmup': 0,
        'use_modelscope': use_modelscope,
    }

    logger.info(f'Sampling params: {sampling_params}')
    logger.info(f'Experiment config: {experiment_config}')

    logger.info(f'Set CUDA_VISIBLE_DEVICES={envs} for model {model_id} with context_length {context_length}')
    os.environ["CUDA_VISIBLE_DEVICES"] = envs

    speed_benchmark_vllm = SpeedBenchmarkVllm(experiment_config=experiment_config, sampling_params=sampling_params)
    speed_benchmark_vllm.run()


if __name__ == '__main__':
    # Usage: python speed_benchmark_vllm.py --model_id Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
    # HF_ENDPOINT=https://hf-mirror.com python speed_benchmark_vllm.py --model_id Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
    main()
