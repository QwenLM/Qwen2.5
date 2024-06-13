TGI
=====================

Hugging Face's Text Generation Inference (TGI) is a production-ready framework specifically designed for deploying and serving large language models (LLMs) for text generation tasks. It offers a seamless deployment experience, powered by a robust set of features:

* `Speculative Decoding <Speculative Decoding_>`_:  Accelerates generation speeds.
* `Tensor Parallelism`_:  Enables efficient deployment across multiple GPUs.
* `Token Streaming`_:  Allows for the continuous generation of text.
* Versatile Device Support:  Works seamlessly with `AMD`_, `Gaudi`_ and `AWS Inferentia`_.

.. _AMD: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/deploy-your-model.html#serving-using-hugging-face-tgi
.. _Gaudi: https://github.com/huggingface/tgi-gaudi
.. _AWS Inferentia: https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/#:~:text=Get%20started%20with%20TGI%20on%20SageMaker%20Hosting
.. _Tensor Parallelism: https://huggingface.co/docs/text-generation-inference/conceptual/speculation
.. _Token Streaming: https://huggingface.co/docs/text-generation-inference/conceptual/streaming

Installation
-----------------

The easiest way to use TGI is via the TGI docker image, but it's possible to run it locally via Conda or build locally. Refer to the `Installation Guide <https://huggingface.co/docs/text-generation-inference/installation>`_  and `CLI tool <https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_cli>`_ for detailed instructions.

Deploy Qwen with TGI
-----------------------

1. **Find a Qwen Model:** Choose a model from the Qwen2 collection `here <https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f>`_.
2. **Deployment Command:** Run the following command in your terminal, replacing ``model`` with your chosen Qwen model ID and ``volume`` with the path to your local data directory:

.. code:: bash

   model=Qwen/Qwen2-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model


Using TGI API
-------------

Once deployed, the model will be available on the selected port

.. code:: bash

   curl 127.0.0.1:8080/generate_stream \
           -X POST \
           -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
           -H 'Content-Type: application/json'

It's also available on OpenAI style API.

.. code:: bash

   curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
         "model": "",  # Note that this field is not used by TGI, you can put anything
         "messages": [
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "Tell me something about large language models."}
         ],
   }'


Refer to the `TGI Swagger UI <https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/completions>`_ for a complete API reference (ignore any errors at the top).

You can also use Python API:

.. code:: python
   from openai import OpenAI
   
   # initialize the client but point it to TGI
   client = OpenAI(
      base_url="localhost:8080" + "/v1/",  # replace with your endpoint url
      api_key="",  # this field is not used when running locally
   )
   chat_completion = client.chat.completions.create(
      model="",
      messages=[
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Tell me something about large language models."},
      ],
      stream=True,
      max_tokens=500
   )

   # iterate and print stream
   for message in chat_completion:
      print(message.choices[0].delta.content, end="")`


Quantization for Performance
----------------------------


1. Data dependent quantization (GPTQ and AWQ)

Both GPTQ and AWQ models are data-dependent. The official quantized model can be found from https://huggingface.co/Qwen and you can also quantize models with your own dataset to make it perform better on your use case. 

.. code:: bash

   # via Docker
   model=Qwen/Qwen2-7B-Instruct-GPTQ-Int4
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --quantize gptq

If the model is quantized with AWQ, e.g. Qwen/Qwen2-7B-Instruct-AWQ please use `--quantize awq`


2. Data agnostic quantization.

EETQ on the other side is not data dependent and can be used with any model. It can be used with the `--quantize eetq` flag. Note that we're passing in the original model with `--quantize eetq` flag.


.. code:: bash

   # via Docker
   model=Qwen/Qwen2-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --quantize eetq


3. Latency metrics

Here are some time_per_token metrics for the quantized qwen 7B instruct models on 4090 GPU:
- gptq int4 6.8ms
- awq int4 7.9ms
- eetq int8 9.7ms


Multi accelerators Deployment
---------------------
Use the ``--num-shard`` flag to specify the number of accelerators. Please also use ``--shm-size 1g`` to enable shared memory for optimal NCCL performance: `Reference <https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#a-note-on-shared-memory-shm>`_

.. code:: bash

   # via Docker
   model=Qwen/Qwen2-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --num-shard 2


Speculative Decoding
-----

Speculative decoding can reduce the time per token by speculating on the next token. Use the ``--speculative-decoding`` flag, setting the value to the number of tokens to speculate on (default: 0 for no speculation):


.. code:: bash

   # via Docker
   model=Qwen/Qwen2-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --speculate 2


Time Per Token Metrics (Qwen 2-7B-Instruct, No Quantization, 4090 GPU)
- no speculation 17.4ms
- speculation with n=2 16.6ms

In this particular use case (code generation), speculative decoding is 10% faster than the default configuration. The overall perfomrance of speculative decoding highly depends on the type of task. It works best for code, or highly repetitive text.

More content on speculative decoding can be found from https://huggingface.co/docs/text-generation-inference/conceptual/speculation


Zero-Code Deployment with HF Inference Endpoints
---------------------------------------------------

For effortless deployment, leverage Hugging Face Inference Endpoints.

- **GUI interface** `https://huggingface.co/inference-endpoints/dedicated`
- **Coding interface:** `https://huggingface.co/blog/tgi-messages-api`

Once deployed, the endpoint can be used as usual.


Common Issue
----------------

Qwen2 supports long context lengths, so carefully choose the values for `--max-batch-prefill-tokens`, `--max-total-tokens`, and `--max-input-tokens` to avoid potential out-of-memory (OOM) issues. If an OOM occurs, you'll receive an error message upon startup.

.. code:: bash

   # via Docker
   model=Qwen/Qwen2-7B-Instruct
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --max-batch-prefill-tokens 4096 --max-total-tokens 4096 --max-input-tokens 2048