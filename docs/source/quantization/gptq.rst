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
``Qwen1.5-7B-Chat-GPTQ-Int8`` (note that for each size of Qwen1.5, we
provide both Int4 and Int8 quantized models) with the quantized model:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", # the quantized model
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int8")

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

Usage of GPTQ Quantized Models with vLLM
----------------------------------------

vLLM has supported GPTQ, which means that you can directly use our
provided GPTQ models or those trained with ``AutoGPTQ`` with vLLM.
Actually, the usage is the same with the basic usage of vLLM. We provide
a simple example of how to launch OpenAI-API compatible API with vLLM
and ``Qwen1.5-7B-Chat-GPTQ-Int8``:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int8

.. code:: bash

    curl http://localhost:8000/v1/chat/completions  -H "Content-Type: application/json" -d '{
       "model": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
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
       model="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Tell me something about large language models."},
       ]
   )
   print("Chat response:", chat_response)

Quantize Your Own Model with AutoGPTQ
-------------------------------------

If you want to quantize your own model to GPTQ quantized models, we
advise you to use AutoGPTQ. It is suggested installing the latest
version of the package by installing from source code:

.. code:: bash

   git clone https://github.com/AutoGPTQ/AutoGPTQ
   cd AutoGPTQ
   pip install -e .

Suppose you have finetuned a model based on ``Qwen1.5-7B``, which is
named ``Qwen1.5-7B-finetuned``, with your own dataset, e.g., Alpaca. To
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
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoGPTQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

Then you need to prepare your data for calibaration. What you need to do
is just put samples into a list, each of which is a text. As we directly
use our finetuning data for calibration, we first format it with ChatML
template. For example:

.. code:: python

   data = []
   for msg in messages:
       msg = c['messages']
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
