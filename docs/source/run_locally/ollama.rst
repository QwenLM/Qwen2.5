Ollama
===========================

`Ollama <https://ollama.com/>`__ helps you run LLMs locally with only a
few commands. It is available at MacOS, Linux, and Windows. Now, Qwen1.5
is officially on Ollama, and you can run it with one command:

.. code:: bash

   ollama run qwen

Next, we introduce more detailed usages of Ollama for running Qwen
models.

Quickstart
----------

Visit the official website `Ollama <https://ollama.com/>`__ and click
download to install Ollama on your device. You can also search models in
the website, where you can find the Qwen1.5 models. Except for the
default one, you can choose to run Qwen1.5-Chat models of different
sizes by:

-  ``ollama run qwen:0.5b``
-  ``ollama run qwen:1.8b``
-  ``ollama run qwen:4b``
-  ``ollama run qwen:7b``
-  ``ollama run qwen:14b``
-  ``ollama run qwen:72b``

Run Ollama with Your GGUF Files
-------------------------------

Sometimes you don't want to pull models and you just want to use Ollama
with your own GGUF files. Suppose you have a GGUF file of Qwen,
``qwen1_5-7b-chat-q4_0.gguf``. For the first step, you need to create a
file called ``Modelfile``. The content of the file is shown below:

.. code:: text

   FROM qwen1_5-7b-chat-q4_0.gguf

   # set the temperature to 1 [higher is more creative, lower is more coherent]
   PARAMETER temperature 0.7
   PARAMETER top_p 0.8
   PARAMETER repeat_penalty 1.05
   PARAMETER top_k 20

   TEMPLATE """{{ if and .First .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}<|im_start|>user
   {{ .Prompt }}<|im_end|>
   <|im_start|>assistant
   {{ .Response }}"""

   # set the system message
   SYSTEM """
   You are a helpful assistant.
   """

Then create the ollama model by running:

.. code:: bash

   ollama create qwen7b -f Modelfile

Once it is finished, you can run your ollama model by:

.. code:: bash

   ollama run qwen7b
