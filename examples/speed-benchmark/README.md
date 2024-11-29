# Speed Benchmark

This document introduces the speed benchmark testing process for the Qwen2.5 series models (original and quantized models). For detailed reports, please refer to the [Qwen2.5 Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## 1. Model Collections

For models hosted on HuggingFace, refer to [Qwen2.5 Collections-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e).

For models hosted on ModelScope, refer to [Qwen2.5 Collections-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768).

## 2. Environment Setup


For inference using HuggingFace transformers:

```shell
conda create -n qwen_perf_transformers python=3.10
conda activate qwen_perf_transformers

pip install torch==2.3.1
pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.8
pip install -r requirements-perf-transformers.txt
```

> [!Important]
> - For `flash-attention`, you can use the prebulit wheels from [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.8) or installing from source, which requires a compatible CUDA compiler.
>   - You don't actually need to install `flash-attention`. It has been intergrated into `torch` as a backend of `sdpa`.
> - For `auto_gptq` to use efficent kernels, you need to install from source, because the prebuilt wheels require incompatible `torch` versions. Installing from source also requires a compatible CUDA compiler.
> - For `autoawq` to use efficent kenerls, you need `autoawq-kernels`, which should be automatically installed. If not, run `pip install autoawq-kernels`.

For inference using vLLM:

```shell
conda create -n qwen_perf_vllm python=3.10
conda activate qwen_perf_vllm

pip install -r requirements-perf-vllm.txt
```

## 3. Execute Tests

Below are two methods for executing tests: using a script or the Speed Benchmark tool.

### Method 1: Testing with Speed Benchmark Tool

Use the Speed Benchmark tool developed by [EvalScope](https://github.com/modelscope/evalscope), which supports automatic model downloads from ModelScope and outputs test results. It also allows testing by specifying the model service URL. For details, please refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/speed_benchmark.html).

**Install Dependencies**
```shell
pip install 'evalscope[perf]' -U
```

#### HuggingFace Transformers Inference

Execute the command as follows:
```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark 
```

#### vLLM Inference

```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark
```

#### Parameter Explanation
- `--parallel` sets the number of worker threads for concurrent requests, should be fixed at 1.
- `--model` specifies the model file path or model ID, supporting automatic downloads from ModelScope, e.g., Qwen/Qwen2.5-0.5B-Instruct.
- `--attn-implementation` sets the attention implementation method, with optional values: flash_attention_2|eager|sdpa.
- `--log-every-n-query`: sets how often to log every n requests.
- `--connect-timeout`: sets the connection timeout in seconds.
- `--read-timeout`: sets the read timeout in seconds.
- `--max-tokens`: sets the maximum output length in tokens.
- `--min-tokens`: sets the minimum output length in tokens; both parameters set to 2048 means the model will output a fixed length of 2048.
- `--api`: sets the inference interface; local inference options are local|local_vllm.
- `--dataset`: sets the test dataset; options are speed_benchmark|speed_benchmark_long.

#### Test Results

Test results can be found in the `outputs/{model_name}/{timestamp}/speed_benchmark.json` file, which contains all request results and test parameters.

### Method 2: Testing with Scripts

#### HuggingFace Transformers Inference

- Using HuggingFace Hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers
```

- Using ModelScope Hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformers
```

Parameter Explanation:

    `--model_id_or_path`: Model ID or local path, optional values refer to the `Model Resources` section  
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; refer to the `Qwen2.5 Model Efficiency Evaluation Report` for specifics  
    `--generate_length`: Number of tokens to generate; default is 2048
    `--gpus`: Equivalent to the environment variable CUDA_VISIBLE_DEVICES, e.g., `0,1,2,3`, `4,5`  
    `--use_modelscope`: If set, uses ModelScope to load the model; otherwise, uses HuggingFace  
    `--outputs_dir`: Output directory, default is `outputs/transformers`  

#### vLLM Inference

- Using HuggingFace Hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

- Using ModelScope Hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

Parameter Explanation:

    `--model_id_or_path`: Model ID or local path, optional values refer to the `Model Resources` section  
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; refer to the `Qwen2.5 Model Efficiency Evaluation Report` for specifics  
    `--generate_length`: Number of tokens to generate; default is 2048
    `--max_model_len`: Maximum model length in tokens; default is 32768  
    `--gpus`: Equivalent to the environment variable CUDA_VISIBLE_DEVICES, e.g., `0,1,2,3`, `4,5`   
    `--use_modelscope`: If set, uses ModelScope to load the model; otherwise, uses HuggingFace  
    `--gpu_memory_utilization`: GPU memory utilization, range (0, 1]; default is 0.9  
    `--outputs_dir`: Output directory, default is `outputs/vllm`  
    `--enforce_eager`: Whether to enforce eager mode; default is False  

#### Test Results

Test results can be found in the `outputs` directory, which by default includes two folders for `transformers` and `vllm`, storing test results for HuggingFace transformers and vLLM respectively.

## Notes

1. Conduct multiple tests and take the average, with a typical value of 3 tests.
2. Ensure the GPU is idle before testing to avoid interference from other tasks.