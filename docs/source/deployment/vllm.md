# vLLM

We recommend you trying [vLLM](https://github.com/vllm-project/vllm) for your deployment of Qwen. 
It is simple to use, and it is fast with state-of-the-art serving throughput, efficient management of attention key value memory with PagedAttention, continuous batching of input requests, optimized CUDA kernels, etc. 
To learn more about vLLM, please refer to the [paper](https://arxiv.org/abs/2309.06180) and [documentation](https://docs.vllm.ai/).

## Environment Setup

By default, you can install `vllm` with pip in a clean environment:

```shell
pip install "vllm>=0.8.5"
```

Please note that the prebuilt `vllm` has strict dependencies on `torch` and its CUDA versions.
Check the note in the official document for installation ([link](https://docs.vllm.ai/en/latest/getting_started/installation.html)) for more help.

## API Service

It is easy to build an OpenAI-compatible API service with vLLM, which can be deployed as a server that implements OpenAI API protocol.
By default, it starts the server at `http://localhost:8000`. 
You can specify the address with `--host` and `--port` arguments. 
Run the command as shown below:
```shell
vllm serve Qwen/Qwen3-8B
```

By default, if the model does not point to a valid local directory, it will download the model files from the Hugging Face Hub.
To download model from ModelScope, set the following before running the above command:
```shell
export VLLM_USE_MODELSCOPE=true
```

For distributed inference with tensor parallelism, it is as simple as
```shell
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4
```
The above command will use tensor parallelism on 4 GPUs.
You should change the number of GPUs according to your demand.

### Basic Usage

Then, you can use the [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) to communicate with Qwen:

::::{tab-set}

:::{tab-item} curl
```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32768
}'
```
:::

:::{tab-item} Python
You can use the API client with the `openai` Python SDK as shown below:

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
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
```
::::

:::{tip}
`vllm` will use the sampling parameters from the `generation_config.json` in the model files.

While the default sampling parameters would work most of the time for thinking mode,
it is recommended to adjust the sampling parameters according to your application, 
and always pass the sampling parameters to the API.
:::


### Thinking & Non-Thinking Modes

Qwen3 models will think before respond.
This behavior could be controlled by either the hard switch, which could disable thinking completely, or the soft switch, where the model follows the instruction of the user on whether it should think.

The hard switch is available in vLLM through the following configuration to the API call.
To disable thinking, use

::::{tab-set}

:::{tab-item} curl
```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```
:::

:::{tab-item} Python
You can use the API client with the `openai` Python SDK as shown below:

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
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=8192,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20, 
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print("Chat response:", chat_response)
```
::::


:::{note}
Please note that passing `enable_thinking` is not OpenAI API compatible.
The exact method may differ among frameworks.
:::

:::{tip}
To completely disable thinking, you could use [a custom chat template](../../source/assets/qwen3_nonthinking.jinja) when starting the model:

```shell
vllm serve Qwen/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja
```

The chat template prevents the model from generating thinking content, even if the user instructs the model to do so with `/think`.
:::


:::{tip}
It is recommended to set sampling parameters differently for thinking and non-thinking modes.
:::


### Parsing Thinking Content

vLLM supports parsing the thinking content from the model generation into structured messages:
```shell
vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1
```

Since vLLM 0.9.0, one can also use
```shell
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

The response message will have a field named `reasoning_content` in addition to `content`, containing the thinking content generated by the model.

:::{note}
Please note that this feature is not OpenAI API compatible.
:::

:::{important}
As of vLLM 0.8.5, `enable_thinking=False` is not compatible with this feature.
If you need to pass `enable_thinking=False` to the API, you should disable parsing thinking content.
This is resolved in vLLM 0.9.0 with the `qwen3` reasoning parser. 
:::

### Parsing Tool Calls

vLLM supports parsing the tool calling content from the model generation into structured messages:
```shell
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

For more information, please refer to [our guide on Function Calling](../framework/function_call.md#vllm).

### Structured/JSON Output

vLLM supports structured/JSON output. 
Please refer to [vLLM's documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#extra-parameters-for-chat-api) for the `guided_json` parameters.
Besides, it is also recommended to instruct the model to generate the specific format in the system message or in your prompt.


### Serving Quantized models

Qwen3 comes with two types of pre-quantized models, FP8 and AWQ.

The command serving those models are the same as the original models except for the name change:
```shell
# For FP8 quantized model
vllm serve Qwen/Qwen3-8B-FP8

# For AWQ quantized model
vllm serve Qwen/Qwen3-8B-AWQ
```

:::{note}
The FP8 models of Qwen3 are block-wise quant, which is supported on NVIDIA GPUs with compute capability > 8.9, that is, Ada Lovelace, Hopper, and later GPUs and runs as w8a8.

Since vLLM v0.9.0, FP8 Marlin has supported block-wise quants (running as w8a16) and you can also run Qwen3 FP8 models on Ampere cards.
:::

:::{note}
If you encountered the following error when deploying the FP8 models, it indicates that the tensor parallel size does not agree with the model weights:
```
File ".../vllm/vllm/model_executor/layers/quantization/fp8.py", line 477, in create_weights
    raise ValueError(
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```

We recommend lowering the degree of tensor parallel, e.g., `--tensor-parallel-size 4` or enabling expert parallel, e.g., `--tensor-parallel-size 8 --enable-expert-parallel`.
:::

### Context Length

The context length for Qwen3 models in pretraining is up to 32,768 tokens.
To handle context length substantially exceeding 32,768 tokens, RoPE scaling techniques should be applied.
We have validated the performance of [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

vLLM supports YaRN, which can be configured as
```shell
vllm serve Qwen/Qwen3-8B --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072  
```

:::{note}
vLLM implements static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
We advise adding the `rope_scaling` configuration only when processing long contexts is required. 
It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 65,536 tokens, it would be better to set `factor` as 2.0. 
:::

:::{note}
The default `max_position_embeddings` in `config.json` is set to 40,960, which used by vLLM, if `--max-model-len` is not specified.
This allocation includes reserving 32,768 tokens for outputs and 8,192 tokens for typical prompts, which is sufficient for most scenarios involving short text processing and leave adequate room for model thinking.
If the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN in this scenario, as it may potentially degrade model performance.
:::

## Python Library

vLLM can also be directly used as a Python library, which is convenient for offline batch inference but lack some API-only features, such as parsing model generation to structure messages.

The following shows the basic usage of vLLM as a library:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="Qwen/Qwen3-8B")

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Set to False to strictly disable thinking
)

# Generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Since vLLM v0.9.0, you can also use the `LLM.chat` interface which includes support for `chat_template_kwargs`:

```python
from vllm import LLM, SamplingParams

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="Qwen/Qwen3-8B")

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]

# Generate outputs
outputs = llm.chat(
    [messages], 
    sampling_params,
    chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## FAQ

You may encounter OOM issues that are pretty annoying.
We recommend two arguments for you to make some fix.

- The first one is `--max-model-len`.
  Our provided default `max_position_embedding` is `40960` and thus the maximum length for the serving is also this value, leading to higher requirements of memory.
  Reducing it to a proper length for yourself often helps with the OOM issue.
- Another argument you can pay attention to is `--gpu-memory-utilization`.
  vLLM will pre-allocate this much GPU memory.
  By default, it is `0.9`.
  This is also why you find a vLLM service always takes so much memory.
  If you are in eager mode (by default it is not), you can level it up to tackle the OOM problem.
  Otherwise, CUDA Graphs are used, which will use GPU memory not controlled by vLLM, and you should try lowering it.
  If it doesn't work, you should try `--enforce-eager`, which may slow down inference, or reduce the `--max-model-len`.


For more usage guide with vLLM, please see vLLM's [Qwen3 Usage Guide](https://github.com/vllm-project/vllm/issues/17327).
