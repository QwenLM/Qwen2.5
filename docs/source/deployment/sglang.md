# SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.

To learn more about SGLang, please refer to the [documentation](https://docs.sglang.ai/).

## Environment Setup

By default, you can install `sglang` with pip in a clean environment:

```shell
pip install "sglang[all]>=0.4.6.post1"
```

If you have encountered issues in installation, please feel free to check the official document for installation ([link](https://docs.sglang.ai/start/install.html)).

## API Service

It is easy to build an OpenAI-compatible API service with SGLang, which can be deployed as a server that implements OpenAI API protocol.
By default, it starts the server at `http://localhost:30000`. 
You can specify the address with `--host` and `--port` arguments. 
Run the command as shown below:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B
```

By default, if the `--model-path` does not point to a valid local directory, it will download the model files from the Hugging Face Hub.
To download model from ModelScope, set the following before running the above command:
```shell
export SGLANG_USE_MODELSCOPE=true
```

For distributed inference with tensor parallelism, it is as simple as
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tensor-parallel-size 4
```
The above command will use tensor parallelism on 4 GPUs.
You should change the number of GPUs according to your demand.

### Basic Usage

Then, you can use the [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) to communicate with Qwen:

::::{tab-set}

:::{tab-item} curl
```shell
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
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
# Set OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

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
While the default sampling parameters would work most of the time for thinking mode,
it is recommended to adjust the sampling parameters according to your application, 
and always pass the sampling parameters to the API.
:::


### Thinking & Non-Thinking Modes

Qwen3 models will think before respond.
This behavior could be controlled by either the hard switch, which could disable thinking completely, or the soft switch, where the model follows the instruction of the user on whether it should think.

The hard switch is available in SGLang through the following configuration to the API call.
To disable thinking, use

::::{tab-set}

:::{tab-item} curl
```shell
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
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
# Set OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

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
        "chat_template_kwargs": {"enable_thinking": True},
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
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja
```

The chat template prevents the model from generating thinking content, even if the user instructs the model to do so with `/think`.
:::


:::{tip}
It is recommended to set sampling parameters differently for thinking and non-thinking modes.
:::

### Parsing Thinking Content

SGLang supports parsing the thinking content from the model generation into structured messages:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --reasoning-parser qwen3
```

The response message will have a field named `reasoning_content` in addition to `content`, containing the thinking content generated by the model.

:::{note}
Please note that this feature is not OpenAI API compatible.
:::

:::{important}
`enable_thinking=False` may not be compatible with this feature.
If you need to pass `enable_thinking=False` to the API, please consider disabling parsing thinking content.
:::

### Parsing Tool Calls

SGLang supports parsing the tool calling content from the model generation into structured messages:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tool-call-parser qwen25
```

For more information, please refer to [our guide on Function Calling](../framework/function_call.md).

### Structured/JSON Output

SGLang supports structured/JSON output. 
Please refer to [SGLang's documentation](https://docs.sglang.ai/backend/structured_outputs.html#OpenAI-Compatible-API).
Besides, it is also recommended to instruct the model to generate the specific format in the system message or in your prompt.

### Serving Quantized models

Qwen3 comes with two types of pre-quantized models, FP8 and AWQ.

The command serving those models are the same as the original models except for the name change:
```shell
# For FP8 quantized model
python -m sglang.launch_server --model-path Qwen/Qwen3-8B-FP8

# For AWQ quantized model
python -m sglang.launch_server --model-path Qwen/Qwen3-8B-AWQ
```

### Context Length

The context length for Qwen3 models in pretraining is up to 32,768 tokens.
To handle context length substantially exceeding 32,768 tokens, RoPE scaling techniques should be applied.
We have validated the performance of [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

SGLang supports YaRN, which can be configured as
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' --context-length 131072
```

:::{note}
SGLang implements static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
We advise adding the `rope_scaling` configuration only when processing long contexts is required. 
It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 65,536 tokens, it would be better to set `factor` as 2.0. 
:::

:::{note}
The default `max_position_embeddings` in `config.json` is set to 40,960, which is used by SGLang.
This allocation includes reserving 32,768 tokens for outputs and 8,192 tokens for typical prompts, which is sufficient for most scenarios involving short text processing and leave adequate room for model thinking.
If the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN in this scenario, as it may potentially degrade model performance.
:::
