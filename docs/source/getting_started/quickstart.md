# Quickstart

This guide helps you quickly start using Qwen3. 
We provide examples of [Hugging Face Transformers](https://github.com/huggingface/transformers) as well as [ModelScope](https://github.com/modelscope/modelscope), and [vLLM](https://github.com/vllm-project/vllm) for deployment.

You can find Qwen3 models in [the Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) at HuggingFace Hub and [the Qwen3 collection](https://www.modelscope.cn/collections/Qwen3-9743180bdc6b48) at ModelScope.

## Transformers

To get a quick start with Qwen3, you can try the inference with `transformers` first.
Make sure that you have installed `transformers>=4.51.0`.
We advise you to use Python 3.10 or higher, and PyTorch 2.6 or higher.

The following is a very simple code snippet showing how to run Qwen3-8B:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

Qwen3 will think before respond, similar to QwQ models.
This means the model will use its reasoning abilities to enhance the quality of generated responses.
The model will first generate thinking content wrapped in a `<think>...</think>` block, followed by the final response.

-   Hard Switch:
    To strictly disable the model's thinking behavior, aligning its functionality with the previous Qwen2.5-Instruct models, you can set `enable_thinking=False` when formatting the text. 
    ```python
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Setting enable_thinking=False disables thinking mode
    )
    ```
    It can be particularly useful in scenarios where disabling thinking is essential for enhancing efficiency.

-   Soft Switch:
    Qwen3 also understands the user's instruction on its thinking behaviour, in particular, the soft switch `/think` and `/no_think`.
    You can add them to user prompts or system messages to switch the model's thinking mode from turn to turn. 
    The model will follow the most recent instruction in multi-turn conversations.

:::{note}
For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in `generation_config.json`).
DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions. 
For more detailed guidance, please refer to the Best Practices section.

For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0. 
:::


## ModelScope

To tackle with downloading issues, we advise you to try [ModelScope](https://github.com/modelscope/modelscope).
Before starting, you need to install `modelscope` with `pip`. 

`modelscope` adopts a programmatic interface similar (but not identical) to `transformers`.
For basic usage, you can simply change the first line of code above to the following:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

For more information, please refer to [the documentation of `modelscope`](https://www.modelscope.cn/docs).

## vLLM 

To deploy Qwen3, we advise you to use vLLM. 
vLLM is a fast and easy-to-use framework for LLM inference and serving. 
In the following, we demonstrate how to build a OpenAI-API compatible API service with vLLM.

First, make sure you have installed `vllm>=0.8.5`.

Run the following code to build up a vLLM service. 
Here we take Qwen3-8B as an example:

```bash
vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1
```

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

While the soft switch is always available, the hard switch is also availabe in vLLM through the following configuration to the API call.
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


## Next Step

Now, you can have fun with Qwen3 models. 
Would love to know more about its usage? 
Feel free to check other documents in this documentation.
