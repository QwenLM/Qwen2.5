## Speed Benchmark

This document introduces the speed benchmark testing process for the Qwen2.5 series models (original and quantized models). For detailed reports, please refer to the [Qwen2.5 SpeedBenchmark](../../docs/source/benchmark/speed_benchmark.rst)

### 1. Model Collections

For models hosted on HuggingFace, please refer to [Qwen2.5 Collections-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)

For models hosted on ModelScope, please refer to [Qwen2.5 Collections-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768)

### 2. Environment Installation

For inference using HuggingFace transformers:

```shell
pip install -r requirements/perf_transformer.txt

# Note: For auto_gptq, you may need to install from the source code.
```

For inference using vLLM:

```shell
pip install -r requirements/perf_vllm.txt

```


### 3. Run experiments

#### 3.1 Inference using HuggingFace Transformers

```shell
python speed_benchmark_transformer.py --model_id Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformer

```

Parameters:

    `--model_id`: Model ID, optional values refer to the Model Resources section.  
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; for specifics, refer to the `Qwen2.5 SpeedBenchmark`.  
    `--gpus`: Number of GPUs to use, e.g., 0,1.  
    `--use_modelscope`: Whether to use ModelScope; if False, HuggingFace is used; default is True.  
    `--outputs_dir`: Output directory; default is outputs/transformer.  


#### 3.2 Inference using vLLM

```shell
python speed_benchmark_vllm.py --model_id Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm

```

Parameters:

    `--model_id`: Model ID, optional values refer to the Model Resources section.  
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; for specifics, refer to the `Qwen2.5 SpeedBenchmark`.  
    `--max_model_len`: Maximum model length in tokens; default is 32768.  
    `--gpus`: Number of GPUs to use, e.g., 0,1.  
    `--use_modelscope`: Whether to use ModelScope; if False, HuggingFace is used; default is True.  
    `--gpu_memory_utilization`: GPU memory utilization; range is (0, 1]; default is 0.9.  
    `--outputs_dir`: Output directory; default is outputs/vllm.  
    `--enforce_eager`: Whether to enforce eager mode; default is False.  


#### 3.3 Tips

- Run multiple experiments and compute the average result; a typical number is 3 times.
- Make sure the GPU is idle before running experiments.


### 4. Results

Please check the `outputs` directory, which includes two directories by default: `transformer` and `vllm`, containing the experiments results for HuggingFace transformers and vLLM, respectively.

