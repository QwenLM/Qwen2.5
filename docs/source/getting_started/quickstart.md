# Quickstart

This guide helps you quickly start using Qwen2.5. 
We provide examples of [Hugging Face Transformers](https://github.com/huggingface/transformers) as well as [ModelScope](https://github.com/modelscope/modelscope), and [vLLM](https://github.com/vllm-project/vllm) for deployment.

You can find Qwen2.5 models in the [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) at Hugging Face Hub.

## Hugging Face Transformers & ModelScope

To get a quick start with Qwen2.5, we advise you to try with the inference with `transformers` first.
Make sure that you have installed `transformers>=4.37.0`.
We advise you to use Python 3.10 or higher, and PyTorch 2.3 or higher.

:::{dropdown} Install `transformers`
* Install with `pip`:

    ```bash
    pip install transformers -U
    ```

* Install with `conda`:

    ```bash
    conda install conda-forge::transformers
    ```

* Install from source:

    ```bash
    pip install git+https://github.com/huggingface/transformers
    ```
:::

The following is a very simple code snippet showing how to run Qwen2.5-7B-Instruct:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
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

As you can see, it's just standard usage for casual LMs in `transformers`!

### Streaming Generation

Streaming mode for model chat is simple with the help of `TextStreamer`. 
Below we show you an example of how to use it:

```python
...
# Reuse the code before `model.generate()` in the last code snippet
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer,
)
```

It will print the text to the console or the terminal as being generated.

### ModelScope

To tackle with downloading issues, we advise you to try [ModelScope](https://github.com/modelscope/modelscope).
Before starting, you need to install `modelscope` with `pip`. 

`modelscope` adopts a programmatic interface similar (but not identical) to `transformers`.
For basic usage, you can simply change the first line of code above to the following:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

For more information, please refer to [the documentation of `modelscope`](https://www.modelscope.cn/docs).

## vLLM for Deployment

To deploy Qwen2.5, we advise you to use vLLM. 
vLLM is a fast and easy-to-use framework for LLM inference and serving. 
In the following, we demonstrate how to build a OpenAI-API compatible API service with vLLM.

First, make sure you have installed `vllm>=0.4.0`:

```bash
pip install vllm
```

Run the following code to build up a vLLM service. 
Here we take Qwen2.5-7B-Instruct as an example:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
```

with `vllm>=0.5.3`, you can also use

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Then, you can use the [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) to communicate with Qwen:

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

or you can use Python client with `openai` Python package as shown below:

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

For more information, please refer to [the documentation of `vllm`](https://docs.vllm.ai/en/stable/).

## Next Step

Now, you can have fun with Qwen2.5 models. 
Would love to know more about its usages? 
Feel free to check other documents in this documentation.
