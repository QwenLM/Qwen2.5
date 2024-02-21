GGUF
===========================

Recently, running LLMs locally is popular in the community, and running
GGUF files with llama.cpp is a typical example. With llama.cpp, you can
not only build GGUF files for your models but also perform low-bit
quantization. In GGUF, you can directly quantize your models without
calibration, or apply the AWQ scale for better quality, or use imatrix
with calibration data. In this document, we demonstrate the simplest way
to quantize your model as well as the way to apply AWQ scale to your
Qwen model quantization.

Quantize Your Models and Make GGUF Files
----------------------------------------

Before you move to quantization, make sure you have followed the
instruction and started to use llama.cpp. The following guidance will
NOT provide instructions about installation and building. Now, suppose
you would like to quantize ``Qwen1.5-7B-Chat``. You need to first make a
GGUF file for the fp16 model as shown below:

.. code:: bash

   python convert-hf-to-gguf.py Qwen/Qwen1.5-7B-Chat --outfile models/7B/qwen1_5-7b-chat-fp16.gguf

where the first argument refers to the path to the HF model directory or
the HF model name, and the second argument refers to the path of your
output GGUF file (here I just put it under the directory ``models/7B``.
Remember to create the directory before you run the command). In this
way, you have generated a GGUF file for your fp16 model, and you then
need to quantize it to low bits based on your requirements. An example
of quantizing the model to 4 bits is shown below:

.. code:: bash

   ./quantize models/7B/qwen1_5-7b-chat-fp16.gguf models/7B/qwen1_5-7b-chat-q4_0.gguf q4_0

where we use ``q4_0`` for the 4-bit quantization. Until now, you have
finished quantizing a model to 4 bits and putting it into a GGUF file,
which can be run directly with llama.cpp.

Quantize Your Models With AWQ Scales
------------------------------------

To improve the quality of your quantized models, one possible solution
is to apply the AWQ scale, following `this
script <https://github.com/casper-hansen/AutoAWQ/blob/main/docs/examples.md>`__.
First of all, when you run ``model.quantize()`` with AutoAWQ, remember
to add ``export_compatible=True`` as shown below:

.. code:: python

   ...
   model.quantize(
       tokenizer,
       quant_config=quant_config,
       export_compatible=True
   )
   model.save_pretrained(quant_path)
   ...

With ``model.save_quantzed()`` as shown above, a fp16 model with AWQ
scales is saved. Then, when you run ``convert-hf-to-gguf.py``, remember
to replace the model path with the path to the fp16 model with AWQ
scales, e.g.,

.. code:: bash

   python convert-hf-to-gguf.py ${quant_path} --outfile models/7B/qwen1_5-7b-chat-fp16-awq.gguf

In this way, you can apply the AWQ scales to your quantized models in
GGUF formats, which helps improving the model quality.

We usually quantize the fp16 model to 2, 3, 4, 5, 6, and 8-bit models.
To perform different low-bit quantization, just replace the quantization
method in your command. For example, if you want to quantize your model
to 2-bit model, you can replace ``q4_0`` to ``q2_k`` as demonstrated
below:

.. code:: bash

   ./quantize models/7B/qwen1_5-7b-chat-fp16.gguf models/7B/qwen1_5-7b-chat-q2_k.gguf q2_k

We now provide GGUF models in the following quantization levels:
``q2_k``, ``q3_k_m``, ``q4_0``, ``q4_k_m``, ``q5_0``, ``q5_k_m``,
``q6_k``, and ``q8_0``. For more information, please visit
`llama.cpp <https://github.com/ggerganov/llama.cpp>`__.
