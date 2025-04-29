# GPTQ

:::{attention}
To be updated for Qwen3.
:::

[GPTQ](https://arxiv.org/abs/2210.17323) is a quantization method for GPT-like LLMs, which uses one-shot weight quantization based on approximate second-order information.
In this document, we show you how to use the quantized model with Hugging Face `transformers` and also how to quantize your own model with [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

## Usage of GPTQ Models with Hugging Face transformers

:::{note}

To use the official Qwen2.5 GPTQ models with `transformers`, please ensure that `optimum>=1.20.0` and compatible versions of `transformers` and `auto_gptq` are installed.

You can do that by 
```bash
pip install -U "optimum>=1.20.0"
```
:::

Now, `transformers` has officially supported AutoGPTQ, which means that you can directly use the quantized model with `transformers`. 
For each size of Qwen2.5, we provide both Int4 and Int8 GPTQ quantized models.
The following is a very simple code snippet showing how to run `Qwen2.5-7B-Instruct-GPTQ-Int4`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto"
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

## Usage of GPTQ Models with vLLM

vLLM has supported GPTQ, which means that you can directly use our provided GPTQ models or those trained with `AutoGPTQ` with vLLM.
If possible, it will automatically use the GPTQ Marlin kernel, which is more efficient.

Actually, the usage is the same with the basic usage of vLLM. 
We provide a simple example of how to launch OpenAI-API compatible API with vLLM and `Qwen2.5-7B-Instruct-GPTQ-Int4`:

Run the following in a shell to start an OpenAI-compatible API service:

```bash
vllm serve Qwen2.5-7B-Instruct-GPTQ-Int4
```

Then, you can call the API as 

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
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
    model="Qwen2.5-7B-Instruct-GPTQ-Int4",
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

## Quantize Your Own Model with AutoGPTQ

If you want to quantize your own model to GPTQ quantized models, we advise you to use AutoGPTQ. 
It is suggested installing the latest version of the package by installing from source code:

```bash
git clone https://github.com/AutoGPTQ/AutoGPTQ
cd AutoGPTQ
pip install -e .
```

Suppose you have finetuned a model based on `Qwen2.5-7B`, which is named `Qwen2.5-7B-finetuned`, with your own dataset, e.g., Alpaca. 
To build your own GPTQ quantized model, you need to use the training data for calibration. 
Below, we provide a simple demonstration for you to run:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quantize_config = BaseQuantizeConfig(
    bits=8, # 4 or 8
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192

# Load your tokenizer and model with AutoGPTQ
# To learn about loading model to multiple GPUs,
# visit https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/tutorial/02-Advanced-Model-Loading-and-Best-Practice.md
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
```

However, if you would like to load the model on multiple GPUs, you need to use `max_memory` instead of `device_map`.
Here is an example:

```python
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    max_memory={i: "20GB" for i in range(4)}
)
```

Then you need to prepare your data for calibration. 
What you need to do is just put samples into a list, each of which is a text. 
As we directly use our finetuning data for calibration, we first format it with ChatML template. 
For example,

```python
import torch

data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    model_inputs = tokenizer([text])
    input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
    data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))
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
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
model.quantize(data, cache_examples_on_gpu=False)
```

Finally, save the quantized model:

```python
model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
```

It is unfortunate that the `save_quantized` method does not support sharding. 
For sharding, you need to load the model and use `save_pretrained` from transformers to save and shard the model.
Except for this, everything is so simple. 
Enjoy!


## Known Issues

### Qwen2.5-72B-Instruct-GPTQ-Int4 cannot stop generation properly

:Model: Qwen2.5-72B-Instruct-GPTQ-Int4
:Framework: vLLM, AutoGPTQ (including Hugging Face transformers)
:Description: Generation cannot stop properly. Continual generation after where it should stop, then repeated texts, either single character, a phrase, or paragraphs, are generated.
:Workaround: The following workaround could be considered
    1. Using the original model in 16-bit floating point
    2. Using the AWQ variants or llama.cpp-based models for reduced chances of abnormal generation

### Qwen2.5-32B-Instruct-GPTQ-Int4 broken with vLLM on multiple GPUs

:Model: Qwen2.5-32B-Instruct-GPTQ-Int4
:Framework: vLLM
:Description: Deployment on multiple GPUs and only garbled text like `!!!!!!!!!!!!!!!!!!` could be generated.
:Workaround: Each of the following workaround could be considered
    1. Using the AWQ or GPTQ-Int8 variants
    2. Using a single GPU
    3. Using Hugging Face `transformers` if latency and throughput are not major concerns


## Troubleshooting

:::{dropdown} With `transformers` and `auto_gptq`, the logs suggest `CUDA extension not installed.` and the inference is slow.

`auto_gptq` fails to find a fused CUDA kernel compatible with your environment and falls back to a plain implementation.
Follow its [installation guide](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/INSTALLATION.md) to install a pre-built wheel or try installing `auto_gptq` from source.
:::


:::{dropdown} Self-quantized Qwen2.5-72B-Instruct-GPTQ with `vllm`, `ValueError: ... must be divisible by ...` is raised. The intermediate size of the self-quantized model is different from the official Qwen2.5-72B-Instruct-GPTQ models.

After quantization the size of the quantized weights are divided by the group size, which is typically 128.
The intermediate size for the FFN blocks in Qwen2.5-72B is 29568.
Unfortunately, {math}`29568 \div 128 = 231`.
Since the number of attention heads and the dimensions of the weights must be divisible by the tensor parallel size, it means you can only run the quantized model with `tensor_parallel_size=1`, i.e., one GPU card.

A workaround is to make the intermediate size divisible by {math}`128 \times 8 = 1024`.
To achieve that, the weights should be padded with zeros.
While it is mathematically equivalent before and after zero-padding the weights, the results may be slightly different in reality.

Try the following:

```python
import torch
from torch.nn import functional as F

from transformers import AutoModelForCausalLM

# must use AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct", torch_dtype="auto")

# this size is Qwen2.5-72B only
pad_size = 128

sd = model.state_dict()

for i, k in enumerate(sd):
    v = sd[k]
    print(k, i)
    # interleaving the padded zeros
    if ('mlp.up_proj.weight' in k) or ('mlp.gate_proj.weight' in k):
        prev_v = F.pad(v.unsqueeze(1), (0, 0, 0, 1, 0, 0)).reshape(29568*2, -1)[:pad_size*2]
        new_v = torch.cat([prev_v, v[pad_size:]], dim=0)
        sd[k] = new_v
    elif 'mlp.down_proj.weight' in k:
        prev_v= F.pad(v.unsqueeze(2), (0, 1)).reshape(8192, 29568*2)[:, :pad_size*2]
        new_v = torch.cat([prev_v, v[:, pad_size:]], dim=1)
        sd[k] = new_v

# this is a very large file; make sure your RAM is enough to load the model
torch.save(sd, '/path/to/padded_model/pytorch_model.bin')
```

This will save the padded checkpoint to the specified directory.
Then, copy other files from the original checkpoint to the new directory and modify the `intermediate_size` in `config.json` to `29696`.
Finally, you can quantize the saved model checkpoint.
:::