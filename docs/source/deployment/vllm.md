# vLLM

We recommend you trying [vLLM](https://github.com/vllm-project/vllm) for your deployment of Qwen. 
It is simple to use, and it is fast with state-of-the-art serving throughput, efficient management of attention key value memory with PagedAttention, continuous batching of input requests, optimized CUDA kernels, etc. 
To learn more about vLLM, please refer to the [paper](https://arxiv.org/abs/2309.06180) and [documentation](https://docs.vllm.ai/).

## Environment Setup

By default, you can install `vllm` with pip in a clean environment:

```shell
pip install "vllm>=0.8.4"
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

By default, if the model does not point to a valid local directory, it will download the model files from the HuggingFace Hub.
To download model from ModelScope, set the following before running the above command:
```shell
export VLLM_USE_MODELSCOPE=true
```

For distrbiuted inference with tensor parallelism, it is as simple as
```shell
vllm server Qwen/Qwen3-8B --tensor-parallel-size 4
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
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768,
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
This behaviour could be controled by either the hard switch, which could disable thinking completely, or the soft switch, where the model follows the instruction of the user on whether or not it should think.

The hard switch is availabe in vLLM through the following configuration to the API call.
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
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("Chat response:", chat_response)
```
::::

:::{tip}
It is recommended to set sampling parameters differently for thinking and non-thinking modes.
:::

### Parsing Thinking Content

vLLM supports parsing the thinking content from the model generation into structured messages:
```shell
vllm serve Qwen/Qwen3-8B --enable-reasoning-parser --reasoning-parser deepseek_r1
```

The response message will have a field named `reasoning_content` in addition to `content`, containing the thinking content generated by the model.

:::{note}
Please note that this feature is not OpenAI API compatible.
:::

### Parsing Tool Calls

vLLM supports parsing the tool calling content from the model generation into structured messages:
```shell
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

For more information, please refer to [our guide on Function Calling](../framework/function_call.md#vllm).

:::{note}
As of vLLM 0.5.4, it is not supported to parse the thinking content and the tool calling from the model generation at the same time.
:::

### Structured/JSON Output

vLLM supports structured/JSON output. 
Please refer to [vLLM's documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#extra-parameters-for-chat-api) for the `guided_json` parameters.
Besides, it is also recommended to instruct the model to generate the specific format in the system message or in your prompt.


### Serving Quantized models

Qwen3 comes with two types of pre-quantized models, FP8 and AWQ.

The command serving those models are the same as the original models except for the name change:
```shell
# For FP8 quantized model
vllm serve Qwen3/Qwen3-8B-FP8

# For AWQ quantized model
vllm serve Qwen3/Qwen3-8B-AWQ
```

:::{note}
FP8 computation is supported on NVIDIA GPUs with compute capability > 8.9, that is, Ada Lovelace, Hopper, and later GPUs.

FP8 models will run on compute capability > 8.0 (Ampere) as weight-only W8A16, utilizing FP8 Marlin.
:::

:::{important}
As of vLLM 0.5.4, there are currently compatibility issues with `vllm` with the Qwen3 FP8 checkpoints. 
For a quick fix, you should make the following changes to the file `vllm/vllm/model_executor/layers/linear.py`:
```python
# these changes are in QKVParallelLinear.weight_loader_v2() of vllm/vllm/model_executor/layers/linear.py
...
shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
shard_size = self._get_shard_size_mapping(loaded_shard_id)
# add the following code
if isinstance(param, BlockQuantScaleParameter):
    weight_block_size = self.quant_method.quant_config.weight_block_size
    block_n, _ = weight_block_size[0], weight_block_size[1]
    shard_offset = (shard_offset + block_n - 1) // block_n
    shard_size = (shard_size + block_n - 1) // block_n
# end of the modification
param.load_qkv_weight(loaded_weight=loaded_weight,
                        num_heads=self.num_kv_head_replicas,
                        shard_id=loaded_shard_id,
                        shard_offset=shard_offset,
                        shard_size=shard_size)
...
```
:::


### Context Length

The context length for Qwen3 models in pretraining is up to 32,768 tokenns.
To handle context length substantially exceeding 32,768 tokens, RoPE scaling techniques should be applied.
We have validated the performance of [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

vLLM supports YaRN, which can be configured as
```shell
vllm serve Qwen3/Qwen3-8B --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072  
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

vLLM can also be directly used as a Python library, which is convinient for offline batch inference but lack some API-only features, such as parsing model generation to structure messages.

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
)

# Generate outputs
outputs = llm.generate([text], sampling_params)

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
  If it doesn't work, you should try `--enforce-eager`, which may slow down infernece, or reduce the `--max-model-len`.
