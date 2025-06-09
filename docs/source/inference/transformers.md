# Transformers

Transformers is a library of pretrained natural language processing for inference and training. 
Developers can use Transformers to train models on their data, build inference applications, and generate texts with large language models.

## Environment Setup

- `transformers>=4.51.0`
- `torch>=2.6` is recommended
- GPU is recommended


## Basic Usage

You can use the `pipeline()` interface or the `generate()` interface to generate texts with Qwen3 in transformers.

In general, the pipeline interface requires less boilerplate code, which is shown here.
The following shows a basic example using pipeline for multi-turn conversations:

```python
from transformers import pipeline

model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]
messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
# print(messages[-1]["content"])

messages.append({"role": "user", "content": "In a single sentence."})
messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
# print(messages[-1]["content"])
```


There are some important parameters creating the pipeline:
-   **Model**: `model_name_or_path` could be a model ID like `Qwen/Qwen3-8B` or a local path.

    To download model files to a local directory, you could use
    ```shell
    huggingface-cli download --local-dir ./Qwen3-8B Qwen/Qwen3-8B
    ```  
    You can also download model files using ModelScope if you are in mainland China
    ```shell
    modelscope download --local_dir ./Qwen3-8B Qwen/Qwen3-8B
    ```
-   **Device Placement**: `device_map="auto"` will load the model parameters to multiple devices automatically, if available. 
    It relies on the `accelerate` package.
    If you would like to use a single device, you can pass `device` instead of device_map.
    `device=-1` or `device="cpu"` indicates using CPU, `device="cuda"` indicates using the current GPU, and `device="cuda:1"` or `device=1` indicates using the second GPU.
    Do not use `device_map` and `device` at the same time!
-   **Compute Precision**: `torch_dtype="auto"` will determine automatically the data type to use based on the original precision of the checkpoint and the precision your device supports.
    For modern devices, the precision determined will be `bfloat16`.
    
    If you don't pass `torch_dtype="auto"`, the default data type is `float32`, which will take double the memory and be slower in computation.


Calls to the text generation pipeline will use the generation configuration from the model file, e.g., `generation_config.json`. 
This configuration could be overridden by passing arguments directly to the call.
The default is equivalent to
```python
messages = generator(messages, do_sample=True, temperature=0.6, top_k=2, top_p=0.95, eos_token_id=[151645, 151643])[0]{"generated_text"}
```

For the best practices in configuring generation parameters, please see the model card.

## Thinking & Non-Thinking Mode

By default, Qwen3 model will think before response.
It is also true for the `pipeline()` interface.
To switch between thinking and non-thinking mode, two methods can be used
-   Append a final assistant message, containing only `<think>\n\n</think>\n\n`. 
    This method is stateless, meaning it will only work for that single turn. 
    It will also strictly prevent the model from generating thinking content.
    For example, 
    ```python
    messages = [
        {"role": "user", "content": "Give me a short introduction to large language models."},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
    ]
    messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
    # print(messages[-1]["content"])

    messages.append({"role": "user", "content": "In a single sentence."})
    messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
    # print(messages[-1]["content"])
    ```
    
-   Add to the user (or the system) message, `/no_think` to disable thinking and `/think` to enable thinking.
    This method is stateful, meaning the model will follow the most recent instruction in multi-turn conversations.

    ```python
    messages = [
        {"role": "user", "content": "Give me a short introduction to large language models./no_think"},
    ]
    messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
    # print(messages[-1]["content"])

    messages.append({"role": "user", "content": "In a single sentence./think"})
    messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
    # print(messages[-1]["content"])
    ```


## Parsing Thinking Content

If you would like a more structured assistant message format, you can use the following function to extract the thinking content into a field named `reasoning_content` which is similar to the format used by vLLM, SGLang, etc.

```python
import copy
import re

def parse_thinking_content(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "assistant" and (m := re.match(r"<think>\n(.+)</think>\n\n", message["content"], flags=re.DOTALL)):
            message["content"] = message["content"][len(m.group(0)):]
            if thinking_content := m.group(1).strip():
                message["reasoning_content"] = thinking_content
    return messages
```

## Parsing Tool Calls

For tool calling with Transformers, please refer to [our guide on Function Calling](../framework/function_call.md#hugging-face-transformers).

## Serving Quantized models

Qwen3 comes with two types of pre-quantized models, FP8 and AWQ.
The command serving those models are the same as the original models except for the name change:

```python
from transformers import pipeline

model_name_or_path = "Qwen/Qwen3-8B-FP8" # FP8 models
# model_name_or_path = "Qwen/Qwen3-8B-AWQ" # AWQ models

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)
```

:::{note}
FP8 computation is supported on NVIDIA GPUs with compute capability > 8.9, that is, Ada Lovelace, Hopper, and later GPUs.

For better performance, make sure `triton` and a CUDA compiler compatible with the CUDA version of `torch` in your environment are installed. 
:::

:::{important}
As of 4.51.0, there are issues with Transformers when running those checkpoints **across GPUs**.
The following method could be used to work around those issues:
- Set the environment variable `CUDA_LAUNCH_BLOCKING=1` before running the script; or
- Uncomment [this line](https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/integrations/finegrained_fp8.py#L340) in your local installation of `transformers`.
:::


## Enabling Long Context

The maximum context length in pre-training for Qwen3 models is 32,768 tokens.
It can be extended to 131,072 tokens with RoPE scaling techniques.
We have validated the performance with YaRN.

Transformers supports YaRN, which can be enabled either by modifying the model files or overriding the default arguments when loading the model.

-   Modifying the model files: In the `config.json` file, add the rope_scaling fields:
    ```json
    {
        ...,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }
    }
    ```
-   Overriding the default arguments:
    ```python
    from transformers import pipeline

    model_name_or_path = "Qwen/Qwen3-8B"

    generator = pipeline(
        "text-generation", 
        model_name_or_path, 
        torch_dtype="auto", 
        device_map="auto",
        model_kwargs={
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        }
    )
    ```

:::{attention}
As of Transformers 4.52.3, it will use `max_position_embeddings/rope_scaling.original_max_position_embeddings` as the `rope_scaling.factor` regradless of the specified `rope_scaling.factor`. See [this issue](https://github.com/huggingface/transformers/issues/38224) for more information.
:::

:::{note}
Transformers implements static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
We advise adding the `rope_scaling` configuration only when processing long contexts is required. 
It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 65,536 tokens, it would be better to set `factor` as 2.0. 
:::


## Streaming Generation

With the help of `TextStreamer`, you can modify your chatting with Qwen3 to streaming mode. 
It will print the response as being generated to the console or the terminal.

```python
from transformers import pipeline, TextStreamer

model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)

streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

messages= generator(messages, max_new_tokens=32768, streamer=streamer)[0]["generated_text"]
```

Besides using `TextStreamer`, we can also use `TextIteratorStreamer` which stores print-ready text in a queue, to be used by a downstream application as an iterator:
```python
from transformers import pipeline, TextIteratorStreamer

model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)

streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

# Use Thread to run generation in background
# Otherwise, the process is blocked until generation is complete
# and no streaming effect can be observed.
from threading import Thread
generation_kwargs = dict(text_inputs=messages, max_new_tokens=32768, streamer=streamer)
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()

generated_text = ""
for new_text in streamer:
    generated_text += new_text
    print(generated_text)
```

## Batch Generation

:::{note}
Batching is not automatically a win for performance.
:::


```python
from transformers import pipeline

model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)
generator.tokenizer.padding_side="left"

batch = [
    [{"role": "user", "content": "Give me a short introduction to large language models."}],
    [{"role": "user", "content": "Give me a detailed introduction to large language models."}],
]

results = generator(batch, max_new_tokens=32768, batch_size=2)
batch = [result[0]["generated_text"] for result in results]
```

## FAQ

You may find distributed inference with Transformers is not as fast as you would imagine.
Transformers with `device_map="auto"` does not apply tensor parallelism, and it only uses one GPU at a time.
For Transformers with tensor parallelism, please refer to [its documentation](https://huggingface.co/docs/transformers/v4.51.3/en/perf_infer_gpu_multi).
