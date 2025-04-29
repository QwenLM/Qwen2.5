# AWQ

:::{attention}
To be updated for Qwen3.
:::

For quantized models, one of our recommendations is the usage of [AWQ](https://arxiv.org/abs/2306.00978) with [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). 

**AWQ** refers to Activation-aware Weight Quantization, a hardware-friendly approach for LLM low-bit weight-only quantization. 

**AutoAWQ** is an easy-to-use Python library for 4-bit quantized models. 
AutoAWQ speeds up models by 3x and reduces memory requirements by 3x compared to FP16. 
AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs. 

In this document, we show you how to use the quantized model with Hugging Face `transformers` and also how to quantize your own model.

## Usage of AWQ Models with Hugging Face transformers

Now, `transformers` has officially supported AutoAWQ, which means that you can directly use the quantized model with `transformers`. 
The following is a very simple code snippet showing how to run `Qwen2.5-7B-Instruct-AWQ` with the quantized model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Usage of AWQ  Models with vLLM

vLLM has supported AWQ, which means that you can directly use our provided AWQ models or those quantized with `AutoAWQ` with vLLM.
We recommend using the latest version of vLLM (`vllm>=0.6.1`) which brings performance improvements to AWQ models; otherwise, the performance might not be well-optimized.

Actually, the usage is the same with the basic usage of vLLM. 
We provide a simple example of how to launch OpenAI-API compatible API with vLLM and `Qwen2.5-7B-Instruct-AWQ`:

Run the following in a shell to start an OpenAI-compatible API service:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ
```

Then, you can call the API as 

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

or you can use the API client with the `openai` Python package as shown below:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```

## Quantize Your Own Model with AutoAWQ

If you want to quantize your own model to AWQ quantized models, we advise you to use AutoAWQ. 

```bash
pip install "autoawq<0.2.7"
```

Suppose you have finetuned a model based on `Qwen2.5-7B`, which is named `Qwen2.5-7B-finetuned`, with your own dataset, e.g., Alpaca. 
To build your own AWQ quantized model, you need to use the training data for calibration. 
Below, we provide a simple demonstration for you to run:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)
```

Then you need to prepare your data for calibration. 
What you need to do is just put samples into a list, each of which is a text. 
As we directly use our finetuning data for calibration, we first format it with ChatML template. 
For example,

```python
data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())
```

where each `msg` is a typical chat message as shown below:

```json
[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]
```

Then just run the calibration process by one line of code:

```python
model.quantize(tokenizer, quant_config=quant_config, calib_data=data)
```

Finally, save the quantized model:

```python
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
```

Then you can obtain your own AWQ quantized model for deployment. 
Enjoy!
