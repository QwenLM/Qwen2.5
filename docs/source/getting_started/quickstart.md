# Quickstart

This guide helps you quickly start using Qwen3. 
We provide examples of [Hugging Face Transformers](https://github.com/huggingface/transformers) as well as [ModelScope](https://github.com/modelscope/modelscope), and [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) for deployment.

You can find Qwen3 models in [the Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) at Hugging Face Hub and [the Qwen3 collection](https://www.modelscope.cn/collections/Qwen3-9743180bdc6b48) at ModelScope.

## Transformers

To get a quick start with Qwen3, you can try the inference with `transformers` first.
Make sure that you have installed `transformers>=4.51.0`.
We advise you to use Python 3.10 or higher, and PyTorch 2.6 or higher.

:::::{tab-set}
:sync-group: model

::::{tab-item} Qwen3-Instruct-2507
:sync: instruct

:::{important}
Qwen3-Instruct-2507 supports **only non-thinking mode** and **does not generate ``<think></think>`` blocks** in its output. 
Different from Qwen3-2504, **specifying `enable_thinking=False` is no longer required or supported**.
:::


The following contains a code snippet illustrating how to use Qwen3-235B-A22B-Instruct-2507 to generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

:::{Note}
We recommend `temperature=0.7`, `top_p=0.8`, `top_k=20`, and `min_p=0` for Qwen3-Instruct-2507 models.
For supported frameworks, adjust `presence_penalty` between 0 and 2 to reduce repetitions.
However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.
:::

:::{Note}
Qwen3-Instruct-2507 may use CoT (chain-of-thoughts) automatically for complex tasks.
We recommend using an output length of 16,384 tokens for most queries.
:::


::::


::::{tab-item} Qwen3-Thinking-2507
:sync: thinking

:::{important}
Qwen3-Thinking-2507 supports **only thinking mode**.
Additionally, to enforce model thinking, the default chat template automatically includes `<think>`. 
Therefore, it is normal for the model's output to contain only `</think>` without an explicit opening `<think>` tag.
:::

The following contains a code snippet illustrating how to use Qwen3-235B-A22B-Thinking-2507 to generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)  # no opening <think> tag
print("content:", content)
```

:::{note}
We recommend `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `min_p=0` for Qwen3-Thinking-2507 models.
For supported frameworks, adjust `presence_penalty` between 0 and 2 to reduce repetitions.
However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.
:::

:::{note}
Qwen3-Thinking-2507 features increased thinking depth. 
We strongly recommend its use in highly complex reasoning tasks with adequate maximum generation length.
:::

::::

::::{tab-item} Qwen3
:sync: hybrid

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
    Qwen3 also understands the user's instruction on its thinking behavior, in particular, the soft switch `/think` and `/no_think`.
    You can add them to user prompts or system messages to switch the model's thinking mode from turn to turn. 
    The model will follow the most recent instruction in multi-turn conversations.

:::{note}
For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in `generation_config.json`).
DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions. 

For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0. 
:::

::::
:::::


## ModelScope

To tackle with downloading issues, we advise you to try [ModelScope](https://github.com/modelscope/modelscope).
Before starting, you need to install `modelscope` with `pip`. 

`modelscope` adopts a programmatic interface similar (but not identical) to `transformers`.
For basic usage, you can simply change the first line of code above to the following:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

For more information, please refer to [the documentation of `modelscope`](https://www.modelscope.cn/docs).

## OpenAI API Compatibility 

You can serve Qwen3 via OpenAI-compatible APIs using frameworks such as vLLM, SGLang, and interact with the API using common HTTP clients or the OpenAI SDKs.

:::::{tab-set}
:sync-group: model

::::{tab-item} Qwen3-Instruct-2507
:sync: instruct

Here we take Qwen3-235B-A22B-Instruct-2507 as an example to start the API:

- SGLang (`sglang>=0.4.6.post1` is required):

    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 --port 8000 --tp 8 --context-length 262144
    ```

- vLLM (`vllm>=0.9.0` is recommended):

    ```shell
    vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --port 8000 --tensor-parallel-size 8 --max-model-len 262144
    ```

:::{note}
Consider adjusting the context length according to the available GPU memory.
:::

::::


::::{tab-item} Qwen3-Thinking-2507
:sync: thinking

Here we take Qwen3-235B-A22B-Thinking-2507 as an example to start the API:

- SGLang (`sglang>=0.4.6.post1` is required):

    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Thinking-2507 --port 8000 --tp 8 --context-length 262144  --reasoning-parser deepseek-r1
    ```

- vLLM (`vllm>=0.9.0` is recommended):

    ```shell
    vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507 --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --enable-reasoning --reasoning-parser deepseek_r1
    ```

:::{note}
Consider adjusting the context length according to the available GPU memory.
:::

:::{important}
We are currently working on adapting the `qwen3` reasoning parsers to the new behavior.
Please follow the command above at the moment.
:::

::::


::::{tab-item} Qwen3
:sync: hybrid

Here we take Qwen3-8B as an example to start the API:

- SGLang (`sglang>=0.4.6.post1` is required):

    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 8000 --reasoning-parser qwen3
    ```

- vLLM (`vllm>=0.9.0` is recommended):

    ```shell
    vllm serve Qwen/Qwen3-8B --port 8000 --enable-reasoning --reasoning-parser qwen3
    ```
::::

:::::


Then, you can use the [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) to communicate with Qwen:


::::::{tab-set}
:sync-group: model

:::::{tab-item} Qwen3-Instruct-2507
:sync: instruct

Here we show the basic command to interact with the chat completion API using Qwen3-235B-A22B-Instruct-2507.

::::{tab-set}
:sync-group: api

:::{tab-item} curl
:sync: curl

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 16384
}'
```
:::

:::{tab-item} Python
:sync: python

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
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=16384,
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "top_k": 20,
    }
)
print("Chat response:", chat_response)
```
::::
:::::

:::::{tab-item} Qwen3-Thinking-2507
:sync: thinking

Here we show the basic command to interact with the chat completion API using Qwen3-235B-A22B-Thinking-2507.

::::{tab-set}
:sync-group: api

:::{tab-item} curl
:sync: curl

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
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
:sync: python

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
    model="Qwen/Qwen3-235B-A22B-Thinking-2507",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    }
)
print("Chat response:", chat_response)
```
::::
:::::

:::::{tab-item} Qwen3
:sync: hybrid

Here we show the basic command to interact with the chat completion API using Qwen3-8B.

The default is with thinking enabled:

::::{tab-set}
:sync-group: api

:::{tab-item} curl
:sync: curl

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
:sync: python

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
    }
)
print("Chat response:", chat_response)
```
:::
::::

To disable thinking, one could use the soft switch (e.g., appending `/nothink` to the user query).
The hard switch can also be used as follows:

::::{tab-set}
:sync-group: api

:::{tab-item} curl
:sync: curl

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
:sync: python

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
    }
)
print("Chat response:", chat_response)
```
:::
:::::
::::::


For more usage, please refer to our document on [SGLang](../deployment/sglang) and [vLLM](../deployment/vllm).


## Thinking Budget

Qwen3 supports the configuration of thinking budget.
It is achieved by ending the thinking process once the budget is reached and guiding the model to generate the "summary" with an early-stopping prompt.

Since this feature involves customization specific to each model, it is currently not available in the open-source frameworks and only implemented by [the Alibaba Cloud Model Studio API](https://www.alibabacloud.com/help/en/model-studio/deep-thinking#6f0633b9cdts1).

However, with existing open-source frameworks, one can generate twice to implement this feature as follows:
1. For the first time, generate tokens up to the thinking budget and check if the thinking process is finished. If the thinking process is not finished, append the early-stopping prompt. 
2. For the second time, continue generation until the end of the content or the upper length limit is fulfilled.

The following snippet shows the implementation with Hugging Face Transformers:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

thinking_budget = 16
max_new_tokens = 32768

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
input_length = model_inputs.input_ids.size(-1)

# first generation until thinking budget
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=thinking_budget
)
output_ids = generated_ids[0][input_length:].tolist()

# check if the generation has already finished (151645 is <|im_end|>)
if 151645 not in output_ids:
    # check if the thinking process has finished (151668 is </think>)
    # and prepare the second model input
    if 151668 not in output_ids:
        print("thinking budget is reached")
        early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        early_stopping_ids = tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)
        input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
    else:
        input_ids = generated_ids
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    # second generation
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=input_length + max_new_tokens - input_ids.size(-1)  # could be negative if max_new_tokens is not large enough (early stopping text is 24 tokens)
    )
    output_ids = generated_ids[0][input_length:].tolist()

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

You should see the output in the console like the following
```text
thinking budget is reached
thinking content: <think>
Okay, the user is asking for a short introduction to large language models

Considering the limited time by the user, I have to give the solution based on the thinking directly now.
</think>
content: Large language models (LLMs) are advanced artificial intelligence systems trained on vast amounts of text data to understand and generate human-like language. They can perform tasks such as answering questions, writing stories, coding, and translating languages. LLMs are powered by deep learning techniques and have revolutionized natural language processing by enabling more context-aware and versatile interactions with text. Examples include models like GPT, BERT, and others developed by companies like OpenAI and Alibaba.
```

:::{note}
For purpose of demonstration only, `thinking_budget` is set to 16.
However, `thinking_budget` should not be set to that low in practice.
We recommend tuning `thinking_budget` based on the latency users can accept and setting it higher than 1024 for meaningful improvements across tasks.

If thinking is not desired at all, developers should make use of the hard switch instead.
:::

## Next Step

Now, you can have fun with Qwen3 models. 
Would love to know more about its usage? 
Feel free to check other documents in this documentation.
