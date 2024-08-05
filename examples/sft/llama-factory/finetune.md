# Fine-tune Qwen with LLaMA-Factory

## Install LLaMA-Factory
Download and install LLaMA-Factory, more dependency package versions can be found in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## Prepare Training Data
Your custom training data should be saved as a jsonl file, each line should be in the following format:
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

Register your custom training data in `data/dataset_info.json`, add the following configuration at the end of the file:
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

## Set Training Configuration
Set the training configuration file, we provide example files for full parameter, LoRA, and QLoRA training. You can see more details in the corresponding files in this directory:
- `qwen2-7b-full-sft.yaml`: Full parameter training.
- `qwen2-7b-lora-sft.yaml`: Fine-tune model with LoRA.
- `qwen2-7b-qlora-sft.yaml`: Fine-tune model with QLoRA.

The deepspeed setting of full parameter training can be found in [deepspeed](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/deepspeed)

## Start training

Full parameter training:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train qwen2-7b-full-sft.json 
```

Fine-tuning with LoRA:
```bash
llamafactory-cli train qwen2-7b-lora-sft.json 
```

Fine-tuning with QLoRA:
```bash
llamafactory-cli train qwen2-7b-qlora-sft.json 
```

## Merge LoRA weights
If you use LoRA or QLoRA to train your model, you need to merge the LoRA weights before inference.

```bash
llamafactory-cli export qwen2-7b-merge-lora.yaml
```

## Inference
The following is an example of inference script:
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