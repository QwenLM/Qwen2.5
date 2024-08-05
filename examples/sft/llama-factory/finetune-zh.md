# 使用LLaMA-Factory微调Qwen模型

## 安装LLaMA-Factory
下载并安装LLaMA-Factory，更多依赖包的版本信息详见[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
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
        }
    ]
}
```

在LLaMA-Factory的`data/dataset_info.json`文件中注册自定义的训练数据，在文件尾部添加如下配置信息：
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

## 设置训练参数配置
设置训练参数的配置文件，我们提供了全量参数、LoRA、QLoRA训练所对应的示例文件，详情见本目录下对应的文件:
- `qwen2-7b-full-sft.yaml`: 全量参数训练
- `qwen2-7b-lora-sft.yaml`: LoRA训练
- `qwen2-7b-qlora-sft.yaml`: QLoRA训练

全量参数训练时的deepspeed配置文件可参考[文件](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/deepspeed)

## 开始训练

全量参数训练：
```bash
FORCE_TORCHRUN=1 llamafactory-cli train qwen2-7b-full-sft.json 
```

LoRA训练：
```bash
llamafactory-cli train qwen2-7b-lora-sft.json 
```

QLoRA训练：
```bash
llamafactory-cli train qwen2-7b-qlora-sft.json 
```

## 合并模型权重
如果采用LoRA或者QLoRA进行训练，脚本只保存对应的LoRA权重，需要合并权重才能进行推理。

```bash
llamafactory-cli export qwen2-7b-merge-lora.yaml
```

## 模型推理
模型推理的示例脚本如下：
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

prompt = "Give me a short introduction to large language model."
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