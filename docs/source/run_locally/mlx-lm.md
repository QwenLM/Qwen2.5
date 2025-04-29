# MLX LM

:::{attention}
To be updated for Qwen3.
:::

[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) helps you run LLMs locally on Apple Silicon. 
It is available at macOS. 
It has already supported Qwen models and this time, we have also provided checkpoints that you can directly use with it.

## Prerequisites

The easiest way to get started is to install the `mlx-lm` package:

- with `pip`:

  ```bash
  pip install mlx-lm
  ```

- with `conda`:

  ```bash
  conda install -c conda-forge mlx-lm
  ```

## Running with Qwen MLX Files

We provide model checkpoints with `mlx-lm` in our Hugging Face organization, and to search for what you need you can search the repo names with `-MLX`.

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from mlx_lm import load, generate

model, tokenizer = load('Qwen/Qwen2.5-7B-Instruct-MLX', tokenizer_config={"eos_token": "<|im_end|>"})

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=text, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)
```

## Make Your MLX files

You can make MLX files with just one command:

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path mlx/Qwen2.5-7B-Instruct/ -q
```

where

- `--hf-path`: the model name on Hugging Face Hub or the local path
- `--mlx-path`: the path for output files
- `-q`: enable quantization
