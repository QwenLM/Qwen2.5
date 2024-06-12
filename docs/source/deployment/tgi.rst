TGI
=====================

Hugging Face's Text Generation Inference (TGI) is a production-ready framework specifically designed for deploying and serving large language models (LLMs) for text generation tasks. It offers a seamless deployment experience, powered by a robust set of features:

* **Speculative Decoding:**  Accelerates generation speeds.
* **Tensor Parallelism:**  Enables efficient deployment across multiple GPUs.
* **Token Streaming:**  Allows for the continuous generation of text.
* **Versatile Device Support:**  Works seamlessly with AMD, Gaudi, AWS Inferentia, and Google TPU.


Installation
-----------------

The easiest way to use TGI is via the TGI docker image, but it's possible to run it locally via Conda. Please refer this doc for instructions on the installation: `https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#local-install`


Deploy Qwen with TGI
-----------------------

Find a model from qwen2 collection and get the model path: `https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f`

You can either run the model via CLI or via Docker


.. code:: bash

   # CLI
   text-generation-launcher --model-id Qwen/Qwen2-0.5B-Instruct --port 8080

   # via Docker
   model=Qwen/Qwen2-0.5B-Instruct
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
           "model": "Qwen/Qwen2-0.5B-Instruct",
           "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Tell me something about large language models."}
           ],
           }'

The full API is listed in this Swagger UI: `https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/completions`


Quantization for Performance
----------------------------

Qwen2 come with both GPTQ and AWQ model. You can run them with additional --quantize flag
. 
TGI can also serve quantized qwen models, for example Qwen/Qwen2-0.5B-Instruct can be ran with "--quantize gptq"

.. code:: bash

   # CLI
   text-generation-launcher --model-id Qwen/Qwen2-72B-Instruct-GPTQ-Int4 --port 8080 --quantize gptq

   # via Docker
   model=Qwen/Qwen2-72B-Instruct-GPTQ-Int4
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --quantize gptq

If the model is quantized with awq, e.g. Qwen/Qwen2-7B-Instruct-AWQ please use --quantize awq


Multi accelerators Deployment
---------------------

Please use --num-shard flag and set it value to the number of accelerators.


Zero-Code Deployment with HF Inference Endpoints
---------------------------------------------------

For effortless deployment, leverage Hugging Face Inference Endpoints.

- **GUI interface** `https://huggingface.co/inference-endpoints/dedicated`
- **Coding interface:** `https://huggingface.co/blog/tgi-messages-api`

Once deployed, the endpoint can be used as usual.


Common Issue
----------------

Qwen2 supports long context lengths, so carefully choose the values for `--max-batch-prefill-tokens`, `--max-total-tokens`, and `--max-input-tokens` to avoid potential out-of-memory (OOM) issues. If an OOM occurs, you'll receive an error message upon startup.
