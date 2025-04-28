TGI
=====================

.. attention:: 
    To be updated for Qwen3.

Hugging Face's Text Generation Inference (TGI) is a production-ready framework specifically designed for deploying and serving large language models (LLMs) for text generation tasks. It offers a seamless deployment experience, powered by a robust set of features:

* `Speculative Decoding <Speculative Decoding_>`_: Accelerates generation speeds.
* `Tensor Parallelism`_: Enables efficient deployment across multiple GPUs.
* `Token Streaming`_: Allows for the continuous generation of text.
* Versatile Device Support: Works seamlessly with `AMD`_, `Gaudi`_ and `AWS Inferentia`_.

.. _AMD: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/deploy-your-model.html#serving-using-hugging-face-tgi
.. _Gaudi: https://github.com/huggingface/tgi-gaudi
.. _AWS Inferentia: https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/#:~:text=Get%20started%20with%20TGI%20on%20SageMaker%20Hosting
.. _Tensor Parallelism: https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism
.. _Token Streaming: https://huggingface.co/docs/text-generation-inference/conceptual/streaming

Installation
-----------------

The easiest way to use TGI is via the TGI docker image. In this guide, we show how to use TGI with docker.

It's possible to run it locally via Conda or build locally. Please refer to `Installation Guide <https://huggingface.co/docs/text-generation-inference/installation>`_  and `CLI tool <https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_cli>`_ for detailed instructions.

Deploy Qwen2.5 with TGI
-----------------------

1. **Find a Qwen2.5 Model:** Choose a model from `the Qwen2.5 collection <https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e>`_.
2. **Deployment Command:** Run the following command in your terminal, replacing ``model`` with your chosen Qwen2.5 model ID and ``volume`` with the path to your local data directory:

.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model


Using TGI API
-------------

Once deployed, the model will be available on the mapped port (8080).

TGI comes with a handy API for streaming response:

.. code:: bash

   curl http://localhost:8080/generate_stream -H 'Content-Type: application/json' \
           -d '{"inputs":"Tell me something about large language models.","parameters":{"max_new_tokens":512}}'


It's also available on OpenAI style API:

.. code:: bash

    curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "",
      "messages": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."}
      ],
      "temperature": 0.7,
      "top_p": 0.8,
      "repetition_penalty": 1.05,
      "max_tokens": 512
    }'


.. note::

   The model field in the JSON is not used by TGI, you can put anything. 

Refer to the `TGI Swagger UI <https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/completions>`_ for a complete API reference.

You can also use Python API:

.. code:: python

   from openai import OpenAI
   
   # initialize the client but point it to TGI
   client = OpenAI(
      base_url="http://localhost:8080/v1/",  # replace with your endpoint url
      api_key="",  # this field is not used when running locally
   )
   chat_completion = client.chat.completions.create(
      model="",  # it is not used by TGI, you can put anything
      messages=[
         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
         {"role": "user", "content": "Tell me something about large language models."},
      ],
      stream=True,
      temperature=0.7,
      top_p=0.8,
      max_tokens=512,
   )

   # iterate and print stream
   for message in chat_completion:
      print(message.choices[0].delta.content, end="")


Quantization for Performance
----------------------------

1. Data-dependent quantization (GPTQ and AWQ)

Both GPTQ and AWQ models are data-dependent. The official quantized models can be found from `the Qwen2.5 collection`_ and you can also quantize models with your own dataset to make it perform better on your use case. 

The following shows the command to start TGI with Qwen2.5-7B-Instruct-GPTQ-Int4:

.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --quantize gptq


If the model is quantized with AWQ, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ, please use ``--quantize awq``.

2. Data-agnostic quantization

EETQ on the other side is not data dependent and can be used with any model. Note that we're passing in the original model (instead of a quantized model) with the ``--quantize eetq`` flag.

.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --quantize eetq



Multi-Accelerators Deployment
-----------------------------

Use the ``--num-shard`` flag to specify the number of accelerators. Please also use ``--shm-size 1g`` to enable shared memory for optimal NCCL performance (`reference <https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#a-note-on-shared-memory-shm>`__):

.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --num-shard 2


Speculative Decoding
--------------------

Speculative decoding can reduce the time per token by speculating on the next token. Use the ``--speculative-decoding`` flag, setting the value to the number of tokens to speculate on (default: 0 for no speculation):


.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --speculate 2


The overall performance of speculative decoding highly depends on the type of task. It works best for code or highly repetitive text.

More context on speculative decoding can be found `here <https://huggingface.co/docs/text-generation-inference/conceptual/speculation>`__.


Zero-Code Deployment with HF Inference Endpoints
---------------------------------------------------

For effortless deployment, leverage Hugging Face Inference Endpoints:

- **GUI interface:** `<https://huggingface.co/inference-endpoints/dedicated>`__
- **Coding interface:** `<https://huggingface.co/blog/tgi-messages-api>`__

Once deployed, the endpoint can be used as usual.


Common Issues
----------------

Qwen2.5 supports long context lengths, so carefully choose the values for ``--max-batch-prefill-tokens``, ``--max-total-tokens``, and ``--max-input-tokens`` to avoid potential out-of-memory (OOM) issues. If an OOM occurs, you'll receive an error message upon startup. The following shows an example to modify those parameters:

.. code:: bash

   model=Qwen/Qwen2.5-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --max-batch-prefill-tokens 4096 --max-total-tokens 4096 --max-input-tokens 2048