# SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.

To learn more about SGLang, please refer to the [documentation](https://docs.sglang.ai/).

## Environment Setup

By default, you can install `sglang` with pip in a clean environment:

```shell
pip install "sglang[all]>=0.4.6"
```

Please note that `sglang` relies on `flashinfer-python` and has strict dependencies on `torch` and its CUDA versions.
Check the note in the official document for installation ([link](https://docs.sglang.ai/start/install.html)) for more help.

## API Service

It is easy to build an OpenAI-compatible API service with SGLang, which can be deployed as a server that implements OpenAI API protocol.
By default, it starts the server at `http://localhost:30000`. 
You can specify the address with `--host` and `--port` arguments. 
Run the command as shown below:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B
```

By default, if the `--model-path` does not point to a valid local directory, it will download the model files from the HuggingFace Hub.
To download model from ModelScope, set the following before running the above command:
```shell
export SGLANG_USE_MODELSCOPE=true
```

For distrbiuted inference with tensor parallelism, it is as simple as
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
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768,
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

:::{important}
This feature has not been released.
For more information, please see this [pull request](https://github.com/sgl-project/sglang/pull/5551).
:::

Qwen3 models will think before respond.
This behaviour could be controled by either the hard switch, which could disable thinking completely, or the soft switch, where the model follows the instruction of the user on whether or not it should think.

The hard switch is availabe in SGLang through the following configuration to the API call.
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
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print("Chat response:", chat_response)
```
::::

:::{tip}
It is recommended to set sampling parameters differently for thinking and non-thinking modes.
:::

### Parsing Thinking Content

SGLang supports parsing the thinking content from the model generation into structured messages:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --reasoning-parser deepseek-r1
```

The response message will have a field named `reasoning_content` in addition to `content`, containing the thinking content generated by the model.

:::{note}
Please note that this feature is not OpenAI API compatible.
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
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B-FP8

# For AWQ quantized model
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B-AWQ
```

### Context Length

The context length for Qwen3 models in pretraining is up to 32,768 tokenns.
To handle context length substantially exceeding 32,768 tokens, RoPE scaling techniques should be applied.
We have validated the performance of [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

SGLang supports YaRN, which can be configured as
```shell
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B --json-model-override-args '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
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
