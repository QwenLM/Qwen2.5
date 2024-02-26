vLLM
=====================

We recommend you trying with
`vLLM <https://github.com/vllm-project/vllm>`__ for your deployement
of Qwen. It is simple to use, and it is fast with state-of-the-art
serving throughtput, efficienct management of attention key value memory
with PagedAttention, continuous batching of input requests, optimized
CUDA kernels, etc. To learn more about vLLM, please refer to the
`paper <https://arxiv.org/abs/2309.06180>`__ and
`documentation <https://vllm.readthedocs.io/>`__.

Installation
------------

By default, you can install ``vLLM`` by pip:
``pip install vLLM>=0.3.0``, but if you are using CUDA 11.8, check the
note in the official document for installation
(`link <https://docs.vllm.ai/en/latest/getting_started/installation.html>`__)
for some help. We also advise you to install ray by ``pip install ray``
for distributed serving.

Offline Batched Inference
-------------------------

Models supported by Qwen2 codes, e.g., Qwen1.5, are supported by vLLM.
The simplest usage of vLLM is offline batched inference as demonstrated
below.

.. code:: python

   from transformers import AutoTokenizer
   from vllm import LLM, SamplingParams

   # Initialize the tokenizer
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

   # Pass the default decoding hyperparameters of Qwen1.5-7B-Chat
   # max_tokens is for the maximum length for generation.
   sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

   # Input the model name or path. Can be GPTQ or AWQ models.
   llm = LLM(model="Qwen/Qwen1.5-7B-Chat")

   # Prepare your prompts
   prompt = "Tell me something about large language models."
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": prompt}
   ]
   text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True
   )

   # generate outputs
   outputs = llm.generate([text], sampling_params)

   # Print the outputs.
   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

OpenAI-API Compatible API Service
---------------------------------

It is easy to build an OpenAI-API compatible API service with vLLM,
which can be deployed as a server that implements OpenAI API protocol.
By default, it starts the server at ``http://localhost:8000``. You can
specify the address with ``--host`` and ``--port`` arguments. Run the
command as shown below:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen1.5-7B-Chat

You don’t need to worry about chat template as it by default uses the
chat template provided by the tokenizer.

Then, you can use the `create chat
interface <https://platform.openai.com/docs/api-reference/chat/completions/create>`__
to communicate with Qwen:

.. code:: bash

   curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
       "model": "Qwen/Qwen1.5-7B-Chat",
       "messages": [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Tell me something about large language models."}
       ]
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

Multi-GPU Distributred Serving
------------------------------

To scale up your serving throughputs, distributed serving helps you by
leveraging more GPU devices. Besides, for large models like
``Qwen1.5-72B-Chat``, it is impossible to serve it on a single GPU.
Here, we demonstrate how to run ``Qwen1.5-72B-Chat`` with tensor
parallelism just by passing in the argument ``tensor_parallel_size``:

.. code:: python

   from vllm import LLM, SamplingParams
   llm = LLM(model="Qwen/Qwen1.5-72B-Chat", tensor_parallel_size=4)

You can run multi-GPU serving by passing in the argument
``--tensor-parallel-size``:

.. code:: bash

   python -m vllm.entrypoints.api_server \
       --model Qwen1.5-72B-Chat \
       --tensor-parallel-size 4

Serving Quantized Models
------------------------

vLLM supports different types of quantized models, including AWQ, GPTQ,
SqueezeLLM, etc. Here we show how to deploy AWQ and GPTQ models. The
usage is almost the same as above except for an additional argument for
quantization. For example, to run an AWQ model. e.g.,
``Qwen1.5-7B-Chat-AWQ``:

.. code:: python

   from vllm import LLM, SamplingParams
   llm = LLM(model="Qwen1.5-7B-Chat-AWQ", quantization="awq")

or GPTQ models like ``Qwen1.5-7B-Chat-GPTQ-Int8``:

.. code:: python

   llm = LLM(model="Qwen1.5-7B-Chat-GPTQ-Int4", quantization="gptq")

Similarly, you can run serving adding the argument ``--quantization`` as
shown below:

.. code:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen1.5-7B-Chat-AWQ \
       --quantization awq

or

.. code:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int8 \
       --quantization gptq

Additionally, vLLM supports the combination of AWQ or GPTQ models with
KV cache quantization, namely FP8 E5M2 KV Cache. For example:

.. code:: python

   llm = LLM(model="Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", quantization="gptq", kv_cache_dtype="fp8_e5m2")

.. code:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen1.5-7B-Chat-GPTQ-Int8 \
       --quantization gptq \
       --kv-cache-dtype fp8_e5m2

Troubleshooting
---------------

You may encounter OOM issues that are pretty annoying. We recommend two
arguments for you to make some fix. The first one is
``--max-model-len``. Our provided default ``max_postiion_embedding`` is
``32768`` and thus the maximum length for the serving is also this
value, leading to higher requirements of memory. Reducing it to a proper
length for yourself often helps with the OOM issue. Another argument you
can pay attention to is ``--gpu-memory-utilization``. By default it is
``0.9`` and you can level it up to tackle the OOM problem. This is also
why you find a vLLM service always takes so much memory.
