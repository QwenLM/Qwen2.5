Quickstart
==========

This guide helps you quickly start using Qwen2. We provide examples of
`Hugging Face Transformers <https://github.com/huggingface/transformers>`__ 
as well as `ModelScope <https://github.com/modelscope/modelscope>`__, and 
`vLLM <https://github.com/vllm-project/vllm>`__ for deployment.

You can find Qwen2 models in the `Qwen2 collection <https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f>`__.

Hugging Face Transformers & ModelScope
--------------------------------------

To get a quick start with Qwen2, we advise you to try with the inference with ``transformers`` first. 
Make sure that you have installed ``transformers>=4.40.0``. 
We advise you to use Python 3.8 or higher, and Pytorch 2.2 or higher.

* Install with ``pip``:

  .. code:: bash

      pip install transformers -U

* Install with ``conda``:

  .. code:: bash
    
      conda install conda-forge::transformers

* Install from source:

  .. code:: bash

      pip install git+https://github.com/huggingface/transformers


The following is a very simple code snippet showing how to run Qwen2-Instruct, with an example of Qwen2-7B-Instruct:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   # Now you do not need to add "trust_remote_code=True"
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2-7B-Instruct",
       torch_dtype="auto",
       device_map="auto",
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

   # Instead of using model.chat(), we directly use model.generate()
   # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
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

   # Directly use generate() and tokenizer.decode() to get the output.
   # Use `max_new_tokens` to control the maximum output length.
   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512,
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
   streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
   generated_ids = model.generate(
       model_inputs.input_ids,
       max_new_tokens=512,
       streamer=streamer,
   )

vLLM for Deployment
-------------------

To deploy Qwen2, we advise you to use vLLM. vLLM is a fast
and easy-to-use framework for LLM inference and serving. In the
following, we demonstrate how to build a OpenAI-API compatible API
service with vLLM.

First, make sure you have installed ``vllm>=0.4.0``:

.. code:: bash

   pip install vllm

Run the following code to build up a vllm service. Here we take
Qwen2-7B-Instruct as an example:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct

Then, you can use the `create chat
interface <https://platform.openai.com/docs/api-reference/chat/completions/create>`__
to communicate with Qwen:

.. code:: bash

    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "Qwen/Qwen2-7B-Instruct",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."}
      ],
      "temperature": 0.7,
      "top_p": 0.8,
      "repetition_penalty": 1.05,
      "max_tokens": 512
    }'

or you can use Python client with ``openai`` Python package as shown
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
        model="Qwen/Qwen2-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me something about large language models."},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )
    print("Chat response:", chat_response)


Next Step
---------

Now, you can have fun with Qwen models. Would love to know more about
its usages? Feel free to check other documents in this documentation.
