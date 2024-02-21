AWQ
=====================

For quantized models, one of our recommendations is the usage of
`AWQ <https://arxiv.org/abs/2306.00978>`__ with
`AutoAWQ <https://github.com/casper-hansen/AutoAWQ>`__. AWQ refers to
Activation-aware Weight Quantization, a hardware-friendly approach for
LLM low-bit weight-only quantization. AutoAWQ is an easy-to-use package
for 4-bit quantized models. AutoAWQ speeds up models by 3x and reduces
memory requirements by 3x compared to FP16. AutoAWQ implements the
Activation-aware Weight Quantization (AWQ) algorithm for quantizing
LLMs. In this document, we show you how to use the quantized model with
Transformers and also how to quantize your own model.

Usage of AWQ Quantized Models with Transformers
-----------------------------------------------

Now, Transformers has officially supported AutoAWQ, which means that you
can directly use the quantized model with Transformers. The following is
a very simple code snippet showing how to run ``Qwen1.5-7B-Chat-AWQ``
with the quantized model:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen1.5-7B-Chat-AWQ", # the quantized model
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-AWQ")

   prompt = "Give me a short introduction to large language model."
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": prompt}
   ]
   text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True
   )
   model_inputs = tokenizer([text], return_tensors="pt").to(device)

   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512
   )
   generated_ids = [
       output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
   ]

   response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

Usage of AWQ Quantized Models with vLLM
---------------------------------------

vLLM has supported AWQ, which means that you can directly use our
provided AWQ models or those trained with ``AutoAWQ`` with vLLM.
Actually, the usage is the same with the basic usage of vLLM. We provide
a simple example of how to launch OpenAI-API compatible API with vLLM
and ``Qwen1.5-7B-Chat-AWQ``:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat-AWQ

.. code:: bash

    curl http://localhost:8000/v1/chat/completions  -H "Content-Type: application/json" -d '{
       "model": "Qwen/Qwen1.5-7B-Chat-AWQ",
       "messages": [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Tell me something about large language models."}
       ],
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
       model="Qwen/Qwen1.5-7B-Chat-AWQ",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Tell me something about large language models."},
       ]
   )
   print("Chat response:", chat_response)

Quantize Your Own Model with AutoAWQ
------------------------------------

If you want to quantize your own model to AWQ quantized models, we
advise you to use AutoAWQ. It is suggested installing the latest version
of the package by installing from source code:

.. code:: bash

   git clone https://github.com/casper-hansen/AutoAWQ.git
   cd AutoAWQ
   pip install -e .

Suppose you have finetuned a model based on ``Qwen1.5-7B``, which is
named ``Qwen1.5-7B-finetuned``, with your own dataset, e.g., Alpaca. To
build your own AWQ quantized model, you need to use the training data
for calibration. Below, we provide a simple demonstration for you to
run:

.. code:: python

   from awq import AutoAWQForCausalLM
   from transformers import AutoTokenizer

   # Specify paths and hyperparameters for quantization
   model_path = "your_model_path"
   quant_path = "your_quantized_model_path"
   quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

   # Load your tokenizer and model with AutoAWQ
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

Then you need to prepare your data for calibaration. What you need to do
is just put samples into a list, each of which is a text. As we directly
use our finetuning data for calibration, we first format it with ChatML
template. For example:

.. code:: python

   data = []
   for msg in messages:
       msg = c['messages']
       text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
       data.append(text.strip())

where each ``msg`` is a typical chat message as shown below:

.. code:: json

   [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Tell me who you are."},
       {"role": "assistant", "content": "I am a large language model named Qwen..."}
   ]

Then just run the calibration process by one line of code:

.. code:: python

   model.quantize(tokenizer, quant_config=quant_config, calib_data=data)

Finally, save the quantized model:

.. code:: python

   model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
   tokenizer.save_pretrained(quant_path)

Then you can obtain your own AWQ quantized model for deployment. Enjoy!
