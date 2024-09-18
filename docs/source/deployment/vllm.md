# vLLM

We recommend you trying [vLLM](https://github.com/vllm-project/vllm) for your deployment of Qwen. 
It is simple to use, and it is fast with state-of-the-art serving throughput, efficient management of attention key value memory with PagedAttention, continuous batching of input requests, optimized CUDA kernels, etc. 
To learn more about vLLM, please refer to the [paper](https://arxiv.org/abs/2309.06180) and [documentation](https://vllm.readthedocs.io/).

## Installation

By default, you can install `vllm` by pip in a clean environment:

```bash
pip install vllm
```

Please note that the prebuilt `vllm` has strict dependencies on `torch` and its CUDA versions.
Check the note in the official document for installation ([link](https://docs.vllm.ai/en/latest/getting_started/installation.html)) for some help.
We also advise you to install ray by `pip install ray` for distributed serving.

## Offline Batched Inference

Models supported by Qwen2.5 codes are supported by vLLM.
The simplest usage of vLLM is offline batched inference as demonstrated below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## OpenAI-Compatible API Service

It is easy to build an OpenAI-compatible API service with vLLM, which can be deployed as a server that implements OpenAI API protocol.
By default, it starts the server at `http://localhost:8000`. 
You can specify the address with `--host` and `--port` arguments. 
Run the command as shown below:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

You don't need to worry about chat template as it by default uses the chat template provided by the tokenizer.

Then, you can use the [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) to communicate with Qwen:

::::{tab-set}

:::{tab-item} curl
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
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
:::

:::{tab-item} Python
You can use the API client with the `openai` Python package as shown below:

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
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
::::


:::{tip}
The OpenAI-compatible server in `vllm` comes with [a default set of sampling parameters](https://github.com/vllm-project/vllm/blob/v0.5.2/vllm/entrypoints/openai/protocol.py#L130),
which are not suitable for Qwen2.5 models and prone to repetition.
We advise you to always pass sampling parameters to the API.
:::


### Tool Use

The OpenAI-compatible API could be configured to support tool call of Qwen2.5.
For information, please refer to [our guide on Function Calling](../framework/function_call.md#vllm).

### Structured/JSON Output

Qwen 2.5, when used with vLLM, supports structured/JSON output. 
Please refer to [vllm's documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#extra-parameters-for-chat-api) for the `guided_json` parameters.
Besides, it is also recommended to instruct the model to generate the specific format in the system message or in your prompt.

## Multi-GPU Distributed Serving

To scale up your serving throughput, distributed serving helps you by leveraging more GPU devices. 
Besides, for large models like `Qwen2.5-72B-Instruct`, it is impossible to serve it on a single GPU.
Here, we demonstrate how to run `Qwen2.5-72B-Instruct` with tensor parallelism just by passing in the argument `tensor_parallel_size`:

::::{tab-set}
:::{tab-item} Offline
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=4)
```
:::

:::{tab-item} API

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4
```
:::
::::

## Extended Context Support

By default, the context length for Qwen2.5 models are set to 32,768 tokens.
To handle extensive inputs exceeding 32,768 tokens, we utilize [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

vLLM supports YARN and it can be enabled by add a `rope_scaling` field to the `config.json` file of the model.
For example,
```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

However, vLLM only supports _static_ YARN at present, which means the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts. 
We advise adding the `rope_scaling` configuration only when processing long contexts is required.


## Serving Quantized Models

vLLM supports different types of quantized models, including AWQ, GPTQ, SqueezeLLM, etc. 
Here we show how to deploy AWQ and GPTQ models. 
The usage is almost the same as above except for an additional argument for quantization. 
For example, to run an AWQ model. e.g., `Qwen2.5-7B-Instruct-AWQ`:

::::{tab-set}
:::{tab-item} Offline
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq")
```
:::

:::{tab-item} API

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq
```
:::
::::

or GPTQ models like `Qwen2.5-7B-Instruct-GPTQ-Int4`:

::::{tab-set}
:::{tab-item} Offline
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", quantization="gptq")
```
:::

:::{tab-item} API

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq
```
:::
::::


Additionally, vLLM supports the combination of AWQ or GPTQ models with KV cache quantization, namely FP8 E5M2 KV Cache. 
For example,

::::{tab-set}
:::{tab-item} Offline
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", quantization="gptq", kv_cache_dtype="fp8_e5m2")
```
:::

:::{tab-item} API

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq --kv-cache-dtype fp8_e5m2
```
:::
::::


## Troubleshooting

You may encounter OOM issues that are pretty annoying.
We recommend two arguments for you to make some fix.

- The first one is `--max-model-len`.
  Our provided default `max_position_embedding` is `32768` and thus the maximum length for the serving is also this value, leading to higher requirements of memory.
  Reducing it to a proper length for yourself often helps with the OOM issue.
- Another argument you can pay attention to is `--gpu-memory-utilization`.
  vLLM will pre-allocate this much GPU memory.
  By default, it is `0.9`.
  This is also why you find a vLLM service always takes so much memory.
  If you are in eager mode (by default it is not), you can level it up to tackle the OOM problem.
  Otherwise, CUDA Graphs are used, which will use GPU memory not controlled by vLLM, and you should try lowering it.
  If it doesn't work, you should try `--enforce-eager`, which may slow down infernece, or reduce the `--max-model-len`.
