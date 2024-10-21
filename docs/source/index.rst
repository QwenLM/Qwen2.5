Welcome to Qwen!
================

.. figure:: https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png
  :width: 60%
  :align: center
  :alt: Qwen2.5
  :class: no-scaled-link


Qwen is the large language model and large multimodal model series of the Qwen Team, Alibaba Group. Now the large language models have been upgraded to Qwen2.5. Both language models and multimodal models are pretrained on large-scale multilingual and multimodal data and post-trained on quality data for aligning to human preferences. 
Qwen is capable of natural language understanding, text generation, vision understanding, audio understanding, tool use, role play, playing as AI agent, etc. 

The latest version, Qwen2.5, has the following features:

- Dense, easy-to-use, decoder-only language models, available in **0.5B**, **1.5B**, **3B**, **7B**, **14B**, **32B**, and **72B** sizes, and base and instruct variants.
- Pretrained on our latest large-scale dataset, encompassing up to **18T** tokens.
- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. 
- More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots. 
- Context length support up to **128K** tokens and can generate up to **8K** tokens. 
- Multilingual support for over **29** languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more. 

For more information, please visit our:

* `Blog <https://qwenlm.github.io/>`__
* `GitHub <https://github.com/QwenLM>`__
* `Hugging Face <https://huggingface.co/Qwen>`__
* `ModelScope <https://modelscope.cn/organization/qwen>`__
* `Qwen2.5 Collection <https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e>`__

Join our community by joining our `Discord <https://discord.gg/yPEP2vHTu4>`__ and `WeChat <https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png>`__ group. We are looking forward to seeing you there!


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/quickstart
   getting_started/concepts
   
.. toctree::
   :maxdepth: 1
   :caption: Inference
   :hidden:

   inference/chat

.. toctree::
   :maxdepth: 1
   :caption: Run Locally
   :hidden:

   run_locally/ollama
   run_locally/mlx-lm
   run_locally/llama.cpp
   
.. toctree::
   :maxdepth: 1
   :caption: Web UI
   :hidden:

   web_ui/text_generation_webui

.. toctree::
   :maxdepth: 1
   :caption: Quantization
   :hidden:

   quantization/awq
   quantization/gptq
   quantization/llama.cpp

.. toctree::
   :maxdepth: 1
   :caption: Deployment
   :hidden:

   deployment/vllm
   deployment/tgi
   deployment/skypilot
   deployment/openllm

.. toctree::
   :maxdepth: 2
   :caption: Training
   :hidden:

   training/SFT/index

.. toctree::
   :maxdepth: 1
   :caption: Framework
   :hidden:

   framework/function_call
   framework/qwen_agent
   framework/LlamaIndex
   framework/Langchain

.. toctree::
   :maxdepth: 1
   :caption: Benchmark
   :hidden:

   benchmark/quantization_benchmark
   benchmark/speed_benchmark
