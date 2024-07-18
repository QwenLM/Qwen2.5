Using Transformers to Chat
==========================

The most significant but also the simplest usage of Qwen2 is to chat
with it using the ``transformers`` library. In this document, we show
how to chat with ``Qwen2-7B-Instruct``, in either streaming mode or not.

Basic Usage
-----------

You can just write several lines of code with ``transformers`` to chat with
Qwen2-Instruct. Essentially, we build the tokenizer and the model with
``from_pretrained`` method, and we use ``generate`` method to perform
chatting with the help of chat template provided by the tokenizer.
Below is an example of how to chat with Qwen2-7B-Instruct:

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   device = "cuda" # the device to load the model onto

   # Now you do not need to add "trust_remote_code=True"
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2-7B-Instruct",
       torch_dtype="auto",
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

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


To continue the chat, simply append the response to the messages with the role assistant and repeat the procedure.
The following shows and example:

.. code:: python

   messages.append({"role": "assistant", "content": response})
   
   prompt = "Tell me more."
   messages.append({"role": "user", "content": prompt})

   text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True,
   )

   model_inputs = tokenizer([text], return_tensors="pt").to(device)

   # Directly use generate() and tokenizer.decode() to get the output.
   # Use `max_new_tokens` to control the maximum output length.
   generated_ids = model.generate(
       **model_inputs,
       max_new_tokens=512,
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


Pipeline Usage
--------------

``transformers`` also provides a functionality called "pipeline" that encapsulates the many operations in common tasks. 
You can chat with the model in just 4 lines of code:

.. code:: python

    from transformers import pipeline

    pipe = pipeline("text-generation", "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")

    # the default system message will be used
    messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

    response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]


Batching
--------

All common ``transformers`` methods support batched input and output.
For basic usage, the following is an example:

.. code:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", padding_side="left")

    message_batch = [
        [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
        [{"role": "user", "content": "Hello!"}],
    ]
    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

    generated_ids_batch = model.generate(
        **model_inputs_batch,
        max_new_tokens=512,
    )
    generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
    response_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

With pipeline, it is simpler:

.. code:: python

    from transformers import pipeline

    pipe = pipeline("text-generation", "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")
    pipe.tokenizer.padding_side="left"

    message_batch = [
        [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
        [{"role": "user", "content": "Hello!"}],
    ]

    result_batch = pipe(messages, max_new_tokens=512, batch_size=2)
    response_message_batch = [result[0]["generated_text"][-1] for result in result_batch]

.. warning::

    Batching is not automatically a win for performance. 


Streaming Mode
--------------

With the help of ``TextStreamer``, you can modify your chatting with
Qwen to streaming mode. Below we show you an example of how to use it:

.. code:: python

   # Repeat the code above before model.generate()
   # Starting here, we add streamer for text generation.
   from transformers import TextStreamer
   streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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

   # Use Thread to run generation in background
   # Otherwise, the process is blocked until generation is complete 
   # and no streaming effect can be observed.
   from threading import Thread
   generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
   thread = Thread(target=model.generate, kwargs=generation_kwargs)
   thread.start()

   generated_text = ""
   for new_text in streamer:
       generated_text += new_text
   print(generated_text)



Using Flash Attention 2 to Accelerate Generation
------------------------------------------------

.. tip::

    With the latest ``transformers`` and ``torch``, Flash Attention 2 will be applied by default if applicable. 
    You do not need to request the use of Flash Attention 2 in ``transformers`` or install the ``flash_attn`` package. 
    The following is intended for users that cannot use latest versions for various reasons.


If you would like to apply Flash Attention 2, you need to install an appropriate version of ``flash_attn``. 
You can find pre-built wheels at `its GitHub repository <https://github.com/Dao-AILab/flash-attention/releases>`__, 
and you should make sure the Python version, the torch version, and the CUDA version of torch are a match. 
Otherwise, you need to install from source. 
Please follow the guides at `its GitHub README <https://github.com/Dao-AILab/flash-attention>`__. 

After a successful installation, you can load the model as shown below:

.. code:: python

    model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2-7B-Instruct",
       torch_dtype="auto",
       device_map="auto",
       attn_implementation="flash_attention_2",
    )


.. note::
    
    The attention module for a model in ``transformers`` typically has three variants: ``sdpa``, ``flash_attention_2``, and ``eager``.
    The first two are wrappers around related functions in the ``torch`` and the ``flash_attn`` packages. 
    It defaults to ``sdpa`` if available.
    
    In addition, ``torch`` has integrated three implementations for ``sdpa``: ``FLASH_ATTENTION`` (indicating Flash Attention 2 since version 2.2), ``EFFICIENT_ATTENTION`` (Memory Efficient Attention), and ``MATH``. 
    It attempts to automatically select the most optimal implementation based on the inputs. 
    You don't need to install extra packages to use them. 

    Hence, if applicable, by default, ``transformers`` uses ``sdpa`` and ``torch`` selects ``FLASH_ATTENTION``.
    
    If you wish to explicitly select the implementations in ``torch``, refer to `this tutorial <https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`__.



Troubleshooting
---------------

**Issue:** Loading models takes a lot of memory.

Normally, memory usage after loading the model can be roughly taken as twice the parameter count. 
For example, a 7B model will take 14GB memory to load. 
It is because for large language models, the compute dtype is often 16-bit floating point number. 
Of course, you will need more memory in inference to store the activations.

For ``transformers``, ``torch_dtype="auto"`` is recommended and the model will be loaded in ``bfloat16`` automatcially. 
Otherwise, the model will be loaded in ``float32`` and it will need double the memory. 
You can also pass ``torch.bfloat16`` as ``torch_dtype`` explicitly. 


----

**Issue:** Multi-GPU inference is slow.

``transformers`` relies on ``accelerate`` for multi-GPU inference and the implementation is a kind of naive model parallelism: 
different GPUs computes different layers of the model. 
It is enabled by the use of ``device_map="auto"`` or a customized ``device_map`` for multiple GPUs.

However, this kind of implementation is not efficienct as for a single request, only one GPU computes at the same time and the other GPUs just wait. 
To use all the GPUs, you need to arrange multiple sequences as on a pipeline, making sure each GPU has some work to do. 
However, that will require concurrency management and load balancing, which is out of the scope of ``transformers``. 
Even if all things are implemented, you can make use of concurrency to improve the total throughput but the latency for each request is not great.

For Multi-GPU inference, we recommend using specialized inference framework, such as vllm and TGI, which support tensor parallelism. 


----

**Issue:** The inference of Qwen2 MoE models is slow.

All MoE models in `transformers` compute the results of the expert FFNs in loops, and it is less efficient for GPUs by nature. 
The performance is even worse for model with finegrained experts, where the model has a lot of experts and each experts is relatively small, which is the case for Qwen2 MoE. 
To optimize that, a fused kernel implementation (as in ``vllm``) or methods like expert parallel (as in ``mcore``) is needed.
For now, we recommend using ``vllm`` for Qwen2 MoE.

----

**Issue:** With Qwen2-7B in ``float16``, ``RuntimeError: probability tensor contains either `inf`, `nan` or element < 0`` is raised or endless of ``!!!!...`` is generated, depending on the PyTorch version.

We don't recommend using ``float16`` for Qwen2 models or numerical instability may occur, especially for cards without support of fp16 matmul with fp32 accumulate.
If you have to use ``float16``, consider using `this fork <https://github.com/jklj077/transformers/tree/qwen2-patch>`__ and force `attn_implementation="eager"`.


Next Step
---------

Now you can chat with Qwen2 in either streaming mode or not. Continue
to read the documentation and try to figure out more advanced usages of
model inference!
