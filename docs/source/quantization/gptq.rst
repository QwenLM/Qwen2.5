GPTQ
======================

`GPTQ <https://arxiv.org/abs/2210.17323>`__ is a quantization method for
GPT-like LLMs, which uses one-shot weight quantization based on
approximate second-order information. In this document, we show you how
to use the quantized model with transformers and also how to quantize
your own model with `AutoGPTQ <https://github.com/AutoGPTQ/AutoGPTQ>`__.

Usage of GPTQ Models with Transformers
--------------------------------------

Now, Transformers has officially supported AutoGPTQ, which means that
you can directly use the quantized model with Transformers. The
following is a very simple code snippet showing how to run
``Qwen2-7B-Instruct-GPTQ-Int4`` (note that for each size of Qwen2, we
provide both Int4 and Int8 quantized models) with the quantized model:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2-7B-Instruct-GPTQ-Int4", # the quantized model
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-GPTQ-Int4")

   prompt = "Give me a short introduction to large language model."
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": prompt},
   ]
   text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True,
   )
   model_inputs = tokenizer([text], return_tensors="pt").to(device)

   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512,
   )
   generated_ids = [
       output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
   ]

   response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


Usage of GPTQ Quantized Models with vLLM
----------------------------------------

.. attention:: 

   ``vllm`` does not support GPTQ quantized Qwen2 MoE models at the moment (version 0.5.2).


vLLM has supported GPTQ, which means that you can directly use our
provided GPTQ models or those trained with ``AutoGPTQ`` with vLLM.
Actually, the usage is the same with the basic usage of vLLM. We provide
a simple example of how to launch OpenAI-API compatible API with vLLM
and ``Qwen2-7B-Instruct-GPTQ-Int4``:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct-GPTQ-Int4


.. code:: bash

    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."}
      ],
      "temperature": 0.7,
      "top_p": 0.8,
      "repetition_penalty": 1.05,
      "max_tokens": 512
    }'

or you can use python client with ``openai`` python package as shown
below:

.. code:: python

    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me something about large language models."},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )
    print("Chat response:", chat_response)



Quantize Your Own Model with AutoGPTQ
-------------------------------------

.. important:: 

    AutoGPTQ does not officially support quantizing Qwen2 MoE models at the moment (version 0.7.1). 
    Consider using `this fork <https://github.com/bozheng-hit/AutoGPTQ/tree/qwen2_moe>`__.


If you want to quantize your own model to GPTQ quantized models, we
advise you to use AutoGPTQ. It is suggested installing the latest
version of the package by installing from source code:

.. code:: bash

   git clone https://github.com/AutoGPTQ/AutoGPTQ
   cd AutoGPTQ
   pip install -e .

Suppose you have finetuned a model based on ``Qwen2-7B``, which is
named ``Qwen2-7B-finetuned``, with your own dataset, e.g., Alpaca. To
build your own GPTQ quantized model, you need to use the training data
for calibration. Below, we provide a simple demonstration for you to
run:

.. code:: python

   from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
   from transformers import AutoTokenizer

   # Specify paths and hyperparameters for quantization
   model_path = "your_model_path"
   quant_path = "your_quantized_model_path"
   quantize_config = BaseQuantizeConfig(
       bits=8, # 4 or 8
       group_size=128,
       damp_percent=0.01,
       desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
       static_groups=False,
       sym=True,
       true_sequential=True,
       model_name_or_path=None,
       model_file_base_name="model"
   )
   max_len = 8192

   # Load your tokenizer and model with AutoGPTQ
   # To learn about loading model to multiple GPUs,
   # visit https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/tutorial/02-Advanced-Model-Loading-and-Best-Practice.md
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)




However, if you would like to load the model on multiple GPUs, you need to use ``max_memory`` instead of ``device_map``. 
Here is an example:

.. code:: python

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config,
        max_memory={i: "20GB" for i in range(4)}
    )

Then you need to prepare your data for calibration. What you need to do
is just put samples into a list, each of which is a text. As we directly
use our finetuning data for calibration, we first format it with ChatML
template. For example:

.. code:: python

   import torch

   data = []
   for msg in dataset:
       text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
       model_inputs = tokenizer([text])
       input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
       data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

where each ``msg`` is a typical chat message as shown below:

.. code:: json

   [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Tell me who you are."},
       {"role": "assistant", "content": "I am a large language model named Qwen..."}
   ]

Then just run the calibration process by one line of code:

.. code:: python

   import logging

   logging.basicConfig(
       format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
   )
   model.quantize(data, cache_examples_on_gpu=False)

Finally, save the quantized model:

.. code:: python

   model.save_quantized(quant_path, use_safetensors=True)
   tokenizer.save_pretrained(quant_path)

It is unfortunate that the ``save_quantized`` method does not support
sharding. For sharding, you need to load the model and use
``save_pretrained`` from transformers to save and shard the model.
Except for this, everything is so simple. Enjoy!



Troubleshooting
---------------

**Issue:** 
With ``transformers`` and ``auto_gptq``, the logs suggest ``CUDA extension not installed.`` and the inference is slow.

``auto_gptq`` fails to find a fused CUDA kernel compatible with your environment and falls back to a plain implementation.
Follow its `installation guide <https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/INSTALLATION.md>`__ to install a pre-built wheel or try installing ``auto_gptq`` from source.

----

**Issue:** 
Qwen2-7B-Instruct-GPTQ-Int8 and Qwen2-1.5B-Instruct-GPTQ-Int8 inferencing with ``transformers`` and ``auto_gptq``, ``RuntimeError: probability tensor contains either `inf`, `nan` or element < 0`` is raised or endless of ``!!!!...`` is generated, depending on the PyTorch version.

The fused CUDA kernels for 8-bit quantized models in ``auto_gptq`` that are also accessible to ``transformers`` is the one called ``cuda_old``. 
It is not numerically stable for Qwen2 models. 
There are two workarounds:

1. Use ``vllm``:

   ``vllm`` uses a custom kernel for 8-bit GPTQ quantized models based on ``exllama_v2``.

2. Use the ``triton`` kernel if ``auto_gptq`` must be used:

   The ``triton`` kernel in ``auto_gptq`` is not accessible to ``transformers``. 
   Follow these steps:

   1. Copy the content of ``quantization_config`` in ``config.json`` to ``quantize_config.json`` in the model files;
   2. Use ``AutoGPTQForCausalLM.from_quantized`` from ``auto_gptq`` instead of ``AutoModelForCausalLM.from_pretrained`` from ``transformers`` to load the model;
   3. Pass ``use_triton`` to ``from_quantized`` (and make sure you have ``triton`` and ``nvcc`` installed).

----

**Issue:** 
Self-quantized Qwen2-72B-Instruct-GPTQ with ``vllm``, ``ValueError: ... must be divisible by ...`` is raised. 
The intermediate size of the self-quantized model is different from the official Qwen2-72B-Instruct-GPTQ models.


After quantization the size of the quantized weights are divided by the group size, which is typically 128. 
The intermediate size for the FFN blocks in Qwen2-72B is 29568. 
Unfortunately, :math:`29568 \div 128 = 231`. 
Since the number of attention heads and the dimensions of the weights must be divisible by the tensor parallel size, it means you can only run the quantized model with ``tensor_parallel_size=1``, i.e., one GPU card.

A workaround is to make the intermediate size divisible by :math:`128 \times 8 = 1024`. 
To achieve that, the weights should be padded with zeros.
While it is mathematically equivalent before and after zero-padding the weights, the results may be slightly different in reality.

Try the following:

.. code:: python

    import torch
    from torch.nn import functional as F

    from transformers import AutoModelForCausalLM

    # must use AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-72B-Instruct", torch_dtype="auto")

    # this size is Qwen2-72B only
    pad_size = 128 

    sd = model.state_dict()

    for i, k in enumerate(sd):
        v = sd[k]
        print(k, i)
        # interleaving the padded zeros
        if ('mlp.up_proj.weight' in k) or ('mlp.gate_proj.weight' in k):
            prev_v = F.pad(v.unsqueeze(1), (0, 0, 0, 1, 0, 0)).reshape(29568*2, -1)[:pad_size*2]
            new_v = torch.cat([prev_v, v[pad_size:]], dim=0)
            sd[k] = new_v
        elif 'mlp.down_proj.weight' in k:
            prev_v= F.pad(v.unsqueeze(2), (0, 1)).reshape(8192, 29568*2)[:, :pad_size*2]
            new_v = torch.cat([prev_v, v[:, pad_size:]], dim=1)
            sd[k] = new_v

    # this is a very large file; make sure your RAM is enough to load the model
    torch.save(sd, '/path/to/padded_model/pytorch_model.bin')

This will save the padded checkpoint to the specified directory. 
Then, copy other files from the original checkpoint to the new directory and modify the ``intermediate_size`` in ``config.json`` to ``29696``. 
Finally, you can quantize the saved model checkpoint.
