## 效率评估

本文介绍Qwen2.5系列模型（原始模型和量化模型）的效率测试流程，详细报告可参考 [Qwen2.5模型效率评估报告](../../docs/source/benchmark/speed_benchmark.rst)

### 1. 模型资源

对于托管在HuggingFace上的模型，可参考 [Qwen2.5模型-HuggingFace](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)。

对于托管在ModelScope上的模型，可参考 [Qwen2.5模型-ModelScope](https://modelscope.cn/collections/Qwen25-dbc4d30adb768)


### 2. 环境安装

使用HuggingFace transformers推理，安装环境如下：

```shell
pip install -r requirements/perf_transformer.txt

# 注意：对于auto_gptq，可能需要从源码安装。
```


使用vLLM推理，安装环境如下：

```shell
pip install -r requirements/perf_vllm.txt

```


### 3. 执行测试

#### 3.1 使用HuggingFace transformers推理

```shell
python speed_benchmark_transformer.py --model_id Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 --context_length 1 --gpus 0 --use_modelscope --outputs_dir outputs/transformer
```

- 参数说明：
    `--model_id`: 模型ID， 可选值参考`模型资源`章节
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`
    `--gpus`: 使用的GPU数量，例如`0,1`
    `--use_modelscope`: 是否使用ModelScope，如果为False，则使用HuggingFace；默认为True
    `--outputs_dir`: 输出目录， 默认为`outputs/transformer`

#### 3.2 使用vLLM推理

```shell
python speed_benchmark_vllm.py --model_id Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 --context_length 1 --max_model_len 32768 --gpus 0 --use_modelscope --gpu_memory_utilization 0.9 --outputs_dir outputs/vllm

```

- 参数说明：
    `--model_id`: 模型ID， 可选值参考`模型资源`章节
    `--context_length`: 输入长度，单位为token数；可选值为1, 6144, 14336, 30720, 63488, 129024；具体可参考`Qwen2.5模型效率评估报告`
    `--max_model_len`: 模型最大长度，单位为token数；默认为32768
    `--gpus`: 使用的GPU数量，例如`0,1`
    `--use_modelscope`: 是否使用ModelScope，如果为False，则使用HuggingFace；默认为True
    `--gpu_memory_utilization`: GPU内存利用率，取值范围为(0, 1]；默认为0.9
    `--outputs_dir`: 输出目录， 默认为`outputs/vllm`
    `--enforce_eager`: 是否强制使用eager模式；默认为False

#### 3.3 注意事项

1. 多次测试，取平均值，典型值为3次
2. 测试前请确保GPU处于空闲状态，避免其他任务影响测试结果

### 4. 测试结果

测试结果详见`outputs`目录下的文件，默认包括`transformer`和`vllm`两个目录，分别存放HuggingFace transformers和vLLM的测试结果。
