## 效率评估

本文介绍Qwen2.5系列模型（原始模型和量化模型）的效率测试流程，详细报告可参考 [Qwen2.5模型效率评估报告](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)。

### 1. 模型资源

对于托管在HuggingFace上的模型，可参考 [Qwen2.5模型-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)。

对于托管在ModelScope上的模型，可参考 [Qwen2.5模型-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768)。


### 2. 环境安装


使用HuggingFace transformers推理，安装环境如下：

```shell
conda create -n qwen_perf_transformers python=3.10
conda activate qwen_perf_transformers

pip install torch==2.3.1
pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.8
pip install -r requirements-perf-transformers.txt
```

> [!Important]
> - 对于 `flash-attention`，您可以从 [GitHub 发布页面](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.8) 使用预编译的 wheel 包进行安装，或者从源代码安装，后者需要一个兼容的 CUDA 编译器。
>   - 实际上，您并不需要单独安装 `flash-attention`。它已经被集成到了 `torch` 中作为 `sdpa` 的后端实现。
> - 若要使 `auto_gptq` 使用高效的内核，您需要从源代码安装，因为预编译的 wheel 包依赖于与之不兼容的 `torch` 版本。从源代码安装同样需要一个兼容的 CUDA 编译器。
> - 若要使 `autoawq` 使用高效的内核，您需要安装 `autoawq-kernels`，该组件应当会自动安装。如果未自动安装，请运行 `pip install autoawq-kernels` 进行手动安装。


使用vLLM推理，安装环境如下：

```shell
conda create -n qwen_perf_vllm python=3.10
conda activate qwen_perf_vllm

pip install -r requirements-perf-vllm.txt
```


### 3. 执行测试

#### 3.1 使用HuggingFace transformers推理

- 使用HuggingFace hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers

# 指定HF_ENDPOINT
HF_ENDPOINT=https://hf-mirror.com python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --outputs_dir outputs/transformers
```

- 使用ModelScope hub

```shell
python speed_benchmark_transformers.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformers
```

参数说明：

    `--model_id_or_path`: 模型ID或本地路径， 可选值参考`模型资源`章节  
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`  
    `--generate_length`: 生成token数量；默认为2048
    `--gpus`: 等价于环境变量CUDA_VISIBLE_DEVICES，例如`0,1,2,3`，`4,5`  
    `--use_modelscope`: 如果设置该值，则使用ModelScope加载模型，否则使用HuggingFace  
    `--outputs_dir`: 输出目录， 默认为`outputs/transformers`  


#### 3.2 使用vLLM推理

- 使用HuggingFace hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm

# 指定HF_ENDPOINT
HF_ENDPOINT=https://hf-mirror.com python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

- 使用ModelScope hub

```shell
python speed_benchmark_vllm.py --model_id_or_path Qwen/Qwen2.5-0.5B-Instruct --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm
```

参数说明：

    `--model_id_or_path`: 模型ID或本地路径， 可选值参考`模型资源`章节  
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`  
    `--generate_length`: 生成token数量；默认为2048
    `--max_model_len`: 模型最大长度，单位为token数；默认为32768  
    `--gpus`: 等价于环境变量CUDA_VISIBLE_DEVICES，例如`0,1,2,3`，`4,5`   
    `--use_modelscope`: 如果设置该值，则使用ModelScope加载模型，否则使用HuggingFace  
    `--gpu_memory_utilization`: GPU内存利用率，取值范围为(0, 1]；默认为0.9  
    `--outputs_dir`: 输出目录， 默认为`outputs/vllm`  
    `--enforce_eager`: 是否强制使用eager模式；默认为False  


#### 3.3 注意事项

1. 多次测试，取平均值，典型值为3次
2. 测试前请确保GPU处于空闲状态，避免其他任务影响测试结果

### 4. 测试结果

测试结果详见`outputs`目录下的文件，默认包括`transformers`和`vllm`两个目录，分别存放HuggingFace transformers和vLLM的测试结果。
