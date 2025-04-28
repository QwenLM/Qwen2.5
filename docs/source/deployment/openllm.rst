OpenLLM
=======

.. attention:: 
    To be updated for Qwen3.

OpenLLM allows developers to run Qwen2.5 models of different sizes as OpenAI-compatible APIs with a single command. It features a built-in chat UI, state-of-the-art inference backends, and a simplified workflow for creating enterprise-grade cloud deployment with Qwen2.5. Visit `the OpenLLM repository <https://github.com/bentoml/OpenLLM/>`_ to learn more.

Installation
------------

Install OpenLLM using ``pip``.

.. code:: bash

   pip install openllm

Verify the installation and display the help information:

.. code:: bash

   openllm --help

Quickstart
----------

Before you run any Qwen2.5 model, ensure your model repository is up to date by syncing it with OpenLLM's latest official repository.

.. code:: bash

   openllm repo update

List the supported Qwen2.5 models:

.. code:: bash

   openllm model list --tag qwen2.5

The results also display the required GPU resources and supported platforms:

.. code:: bash

   model    version                repo     required GPU RAM    platforms
   -------  ---------------------  -------  ------------------  -----------
   qwen2.5  qwen2.5:0.5b           default  12G                 linux
            qwen2.5:1.5b           default  12G                 linux
            qwen2.5:3b             default  12G                 linux
            qwen2.5:7b             default  24G                 linux
            qwen2.5:14b            default  80G                 linux
            qwen2.5:14b-ggml-q4    default                      macos
            qwen2.5:14b-ggml-q8    default                      macos
            qwen2.5:32b            default  80G                 linux
            qwen2.5:32b-ggml-fp16  default                      macos
            qwen2.5:72b            default  80Gx2               linux
            qwen2.5:72b-ggml-q4    default                      macos

To start a server with one of the models, use ``openllm serve`` like this:

.. code:: bash

   openllm serve qwen2.5:7b

By default, the server starts at ``http://localhost:3000/``.

Interact with the model server
------------------------------

With the model server up and running, you can call its APIs in the following ways:

.. tab-set::

    .. tab-item:: CURL

       Send an HTTP request to its ``/generate`` endpoint via CURL:

       .. code-block:: bash

            curl -X 'POST' \
               'http://localhost:3000/api/generate' \
               -H 'accept: text/event-stream' \
               -H 'Content-Type: application/json' \
               -d '{
               "prompt": "Tell me something about large language models.",
               "model": "Qwen/Qwen2.5-7B-Instruct",
               "max_tokens": 2048,
               "stop": null
            }'

    .. tab-item:: Python client

       Call the OpenAI-compatible endpoints with frameworks and tools that support the OpenAI API protocol. Here is an example:

       .. code-block:: python

            from openai import OpenAI

            client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

            # Use the following func to get the available models
            # model_list = client.models.list()
            # print(model_list)

            chat_completion = client.chat.completions.create(
               model="Qwen/Qwen2.5-7B-Instruct",
               messages=[
                  {
                        "role": "user",
                        "content": "Tell me something about large language models."
                  }
               ],
               stream=True,
            )
            for chunk in chat_completion:
               print(chunk.choices[0].delta.content or "", end="")

    .. tab-item:: Chat UI

       OpenLLM provides a chat UI at the ``/chat`` endpoint for the LLM server at http://localhost:3000/chat.

       .. image:: ../../source/assets/qwen-openllm-ui-demo.png

Model repository
----------------

A model repository in OpenLLM represents a catalog of available LLMs. You can add your own repository to OpenLLM with custom Qwen2.5 variants for your specific needs. See our `documentation to learn details <https://github.com/bentoml/OpenLLM?tab=readme-ov-file#model-repository>`_.