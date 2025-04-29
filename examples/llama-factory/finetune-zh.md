# 使用LLaMA-Factory微调Qwen模型

## LLAMA-Factory简介
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)是一个简单易用且高效的大模型训练框架，支持上百种大模型的训练，框架特性主要包括：
- 模型种类：LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi 等等。
- 训练算法：（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO 训练、DPO 训练、KTO 训练、ORPO 训练等等。
- 运算精度：16比特全参数微调、冻结微调、LoRA微调和基于AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ的2/3/4/5/6/8比特QLoRA 微调。
- 优化算法：GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ和PiSSA。
- 加速算子：FlashAttention-2和Unsloth。
- 推理引擎：Transformers和vLLM。
- 实验面板：LlamaBoard、TensorBoard、Wandb、MLflow等等。

本文将介绍如何使用LLAMA-Factory对Qwen2系列大模型进行微调（Qwen1.5系列模型也适用），更多特性请参考[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

## 安装LLaMA-Factory
下载并安装LLaMA-Factory：
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

安装完成后，执行`llamafactory-cli version`，若出现以下提示，则表明安装成功：
```
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.8.4.dev0           |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

## 准备训练数据
自定义的训练数据应保存为jsonl文件，每一行的格式如下：
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me something about large language models."
        },
        {
            "role": "assistant",
            "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
        },
        {
            "role": "user",
            "content": "How about Qwen2?"
        },
        {
            "role": "assistant",
            "content": "Qwen2 is a large language model developed by Alibaba Cloud..."
        }
      
    ]
}
```

在LLaMA-Factory文件夹下的`data/dataset_info.json`文件中注册自定义的训练数据，在文件尾部添加如下配置信息：
```
"qwen_train_data": {
    "file_name": "PATH-TO-YOUR-TRAIN-DATA",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
```

## 配置训练参数
设置训练参数的配置文件，我们提供了全量参数、LoRA、QLoRA训练所对应的示例文件，你可以根据自身需求自行修改，配置详情见本目录下对应的文件:
- `qwen2-7b-full-sft.yaml`: 全量参数训练
- `qwen2-7b-lora-sft.yaml`: LoRA训练
- `qwen2-7b-qlora-sft.yaml`: QLoRA训练

全量参数训练时的deepspeed配置文件可参考[文件](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/deepspeed)

部分训练参数说明：

| 参数                          | 说明                                                                                           |
|-----------------------------|----------------------------------------------------------------------------------------------|
| model_name_or_path          | 模型名称或路径                                                                                      |
| stage                       | 训练阶段，可选: rm(reward modeling), pt(pretrain), sft(Supervised Fine-Tuning), PPO, DPO, KTO, ORPO |
| do_train                    | true用于训练, false用于评估                                                                          |
| finetuning_type             | 微调方式。可选: freeze, LoRA, full                                                                  |
| lora_target                 | 采取LoRA方法的目标模块，默认值为all。                                                                       |
| dataset                     | 使用的数据集，使用”,”分隔多个数据集                                                                          |
| template                    | 数据集模板，请保证数据集模板与模型相对应。                                                                        |
| output_dir                  | 输出路径                                                                                         |
| logging_steps               | 日志输出步数间隔                                                                                     |
| save_steps                  | 模型断点保存间隔                                                                                     |
| overwrite_output_dir        | 是否允许覆盖输出目录                                                                                   |
| per_device_train_batch_size | 每个设备上训练的批次大小                                                                                 |
| gradient_accumulation_steps | 梯度积累步数                                                                                       |
| learning_rate               | 学习率                                                                                          |
| lr_scheduler_type           | 学习率曲线，可选 linear, cosine, polynomial, constant 等。                                             |
| num_train_epochs            | 训练周期数                                                                                        |
| bf16                        | 是否使用 bf16 格式                                                                                 |

## 开始训练

全量参数训练：
```bash
FORCE_TORCHRUN=1 llamafactory-cli train qwen2-7b-full-sft.yaml 
```

LoRA训练：
```bash
llamafactory-cli train qwen2-7b-lora-sft.yaml 
```

QLoRA训练：
```bash
llamafactory-cli train qwen2-7b-qlora-sft.yaml 
```

使用上述训练配置，各个方法实测的显存占用如下。训练中的显存占用与训练参数配置息息相关，可根据自身实际需求进行设置。
- 全量参数训练：42.18GB
- LoRA训练：20.17GB
- QLoRA训练: 10.97GB

## 合并模型权重
如果采用LoRA或者QLoRA进行训练，脚本只保存对应的LoRA权重，需要合并权重才能进行推理。**全量参数训练无需执行此步骤**


```bash
llamafactory-cli export qwen2-7b-merge-lora.yaml
```

权重合并的部分参数说明：

| 参数                   | 说明          |
|----------------------|-------------|
| model_name_or_path   | 预训练模型的名称或路径 |
| template             | 模型模板        |
| export_dir           | 导出路径        |
| export_size          | 最大导出模型文件大小  |
| export_device        | 导出设备        |
| export_legacy_format | 是否使用旧格式导出   |

注意：
- 合并Qwen2模型权重，务必将template设为`qwen`；无论LoRA还是QLoRA训练，合并权重时，`finetuning_type`均为`lora`。
- adapter_name_or_path需要与微调中的适配器输出路径output_dir相对应。

## 模型推理
训练完成，合并模型权重之后，即可加载完整的模型权重进行推理， 推理的示例脚本如下：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
model_name_or_path = YOUR-MODEL-PATH

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
