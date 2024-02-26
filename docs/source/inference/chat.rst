Using Transformers to Chat
==========================

The most significant but also the simplest usage of Qwen1.5 is to chat
with it using the ``transformers`` library. In this document, we show
how to chat with ``Qwen1.5-7B-Chat``, in either streaming mode or not.

Basic Usage
-----------

You can just write several lines of code with ``transformers`` to chat with
Qwen1.5-Chat. Essentially, we build the tokenizer and the model with
``from_pretrained`` method, and we use ``generate`` method to perform
chatting with the help of chat template provided by the tokenizer.
Below is an example of how to chat with Qwen1.5-7B-Chat:

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

Note that the previous method in the original Qwen repo ``chat()`` is
now replaced by ``generate()``. The ``apply_chat_template()`` function
is used to convert the messages into a format that the model can
understand. The ``add_generation_prompt`` argument is used to add a
generation prompt, which refers to ``<|im_start|>assistant\n`` to the input. 
Notably, we apply ChatML template for chat models following our previous 
practice. The ``max_new_tokens`` argument is used to set the maximum length 
of the response. The ``tokenizer.batch_decode()`` function is used to 
decode the response. In terms of the input, the above ``messages`` is an 
example to show how to format your dialog history and system prompt. By 
default, if you do not specify system prompt, we directly use ``You are 
a helpful assistant.``.

Streaming Mode
--------------

With the help of ``TextStreamer``, you can modify your chatting with
Qwen to streaming mode. Below we show you an example of how to use it:

.. code:: python

   # Repeat the code above before model.generate()
   # Starting here, we add streamer for text generation.
   from transformers import TextStreamer
   streamer = Texstreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

   # This will print the output in the streaming mode.
   generated_ids = model.generate(
       model_inputs,
       max_new_tokens=512,
       streamer=streamer,
   )

Besides using ``TextStreamer``, we can also use ``TextIteratorStreamer``
which stores print-ready text in a queue, to be used by a downstream
application as an iterator:

.. code:: python

   # Repeat the code above before model.generate()
   # Starting here, we add streamer for text generation.
   from transformers import TextIteratorStreamer
   streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

   from threading import Thread
   generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
   thread = Thread(target=model.generate, kwargs=generation_kwargs)

   thread.start()
   generated_text = ""
   for new_text in streamer:
       generated_text += new_text
   print(generated_text)

Next Step
---------

Now you can chat with Qwen1.5 in either streaming mode or not. Continue
to read the documentation and try to figure out more advanced usages of
model inference!
