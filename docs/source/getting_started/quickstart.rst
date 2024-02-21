Quickstart
==========

This guide helps you quickly start using Qwen1.5. We provide examples of
`Hugging Face Transformers <https://github.com/huggingface/transformers>`__ 
as well as `ModelScope <https://github.com/modelscope/modelscope>`__, and 
`vLLM <https://github.com/vllm-project/vllm>`__ for deployment.

Hugging Face Transformers & ModelScope
--------------------------------------

To get a quick start with Qwen1.5, we advise you to try with the
inference with ``transformers`` first. Make sure that you have installed
``transformers>=4.37.0``. The following is a very simple code snippet
showing how to run Qwen1.5-Chat, with an example of Qwen1.5-7B-Chat:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   # Now you do not need to add "trust_remote_code=True"
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen1.5-7B-Chat",
       torch_dtype="auto",
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

   # Instead of using model.chat(), we directly use model.generate()
   # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
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

   # Directly use generate() and tokenizer.decode() to get the output.
   # Use `max_new_tokens` to control the maximum output length.
   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512
   )
   generated_ids = [
       output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
   ]

   response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

Previously, we use ``model.chat()`` (see ``modeling_qwen.py`` in
previous Qwen models for more information). Now, we follow the practice
of ``transformers`` and directly use ``model.generate()`` with
``apply_chat_template()`` in tokenizer.

To tackle with downloading issues, we advise you to try with from
ModelScope, just changing the first line of code above to the following:

.. code:: python

   from modelscope import AutoModelForCausalLM, AutoTokenizer

Streaming mode for model chat is simple with the help of
``TextStreamer``. Below we show you an example of how to use it:

.. code:: python

   ...
   # Reuse the code before `model.generate()` in the last code snippet
   from transformers import TextStreamer
   streamer = Texstreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512,
       streamer=streamer,
   )

vLLM for Deployment
-------------------

To deploy Qwen1.5, we advise you to use vLLM. vLLM is a fast
and easy-to-use framework for LLM inference and serving. In the
following, we demonstrate how to build a OpenAI-API compatible API
service with vLLM.

First, make sure you have installed ``vLLM>=0.3.0``:

.. code:: bash

   pip install vllm

Run the following code to build up a vllm service. Here we take
Qwen1.5-7B-Chat as an example:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat

Then, you can use the `create chat
interface <https://platform.openai.com/docs/api-reference/chat/completions/create>`__
to communicate with Qwen:

.. code:: bash

    curl http://localhost:8000/v1/chat/completions  -H "Content-Type: application/json" -d '{
       "model": "Qwen/Qwen1.5-7B-Chat",
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
       model="Qwen/Qwen1.5-7B-Chat",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Tell me something about large language models."},
       ]
   )
   print("Chat response:", chat_response)

Next Step
---------

Now, you can have fun with Qwen models. Would love to know more about
its usages? Feel free to check other documents in this documentation.
