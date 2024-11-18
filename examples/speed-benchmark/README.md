## Speed Benchmark

This document introduces the speed benchmark testing process for the Qwen2.5 series models (original and quantized models). For detailed reports, please refer to the [Qwen2.5 Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

### 1. Model Collections

For models hosted on HuggingFace, please refer to [Qwen2.5 Collections-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e).

For models hosted on ModelScope, please refer to [Qwen2.5 Collections-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768).

### 2. Environment Installation


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


### 3. Run Experiments

#### 3.1 Inference using HuggingFace Transformers

- Use HuggingFace hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers
```

- Use ModelScope hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformers
```

Parameters:

    `--model_id_or_path`: The model path or id on ModelScope or HuggingFace hub
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; Refer to the `Qwen2.5 SpeedBenchmark`.  
    `--generate_length`: Output length in tokens; default is 2048.
    `--gpus`: Equivalent to the environment variable CUDA_VISIBLE_DEVICES.  e.g. `0,1,2,3`, `4,5`  
    `--use_modelscope`: Use ModelScope when set this flag. Otherwise, use HuggingFace.  
    `--outputs_dir`: Output directory; default is outputs/transformers.  


#### 3.2 Inference using vLLM

- Use HuggingFace hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```


- Use ModelScope hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```


Parameters:

    `--model_id_or_path`: The model id on ModelScope or HuggingFace hub.
    `--context_length`: Input length in tokens; optional values are 1, 6144, 14336, 30720, 63488, 129024; Refer to the `Qwen2.5 SpeedBenchmark`.  
    `--generate_length`: Output length in tokens; default is 2048.
    `--max_model_len`: Maximum model length in tokens; default is 32768. Optional values are 4096, 8192, 32768, 65536, 131072.
    `--gpus`: Equivalent to the environment variable CUDA_VISIBLE_DEVICES.  e.g. `0,1,2,3`, `4,5`  
    `--use_modelscope`: Use ModelScope when set this flag. Otherwise, use HuggingFace.  
    `--gpu_memory_utilization`: GPU memory utilization; range is (0, 1]; default is 0.9.  
    `--outputs_dir`: Output directory; default is outputs/vllm.  
    `--enforce_eager`: Whether to enforce eager mode; default is False.  



#### 3.3 Tips

- Run multiple experiments and compute the average result; a typical number is 3 times.
- Make sure the GPU is idle before running experiments.


### 4. Results

Please check the `outputs` directory, which includes two directories by default: `transformers` and `vllm`, containing the experiments results for HuggingFace transformers and vLLM, respectively.
