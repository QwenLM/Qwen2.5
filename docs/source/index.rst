Welcome to Qwen!
================

.. figure:: https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png
  :width: 60%
  :align: center
  :alt: Qwen3
  :class: no-scaled-link


Qwen is the large language model and large multimodal model series of the Qwen Team, Alibaba Group. Both language models and multimodal models are pretrained on large-scale multilingual and multimodal data and post-trained on quality data for aligning to human preferences. 
Qwen is capable of natural language understanding, text generation, vision understanding, audio understanding, tool use, role play, playing as AI agent, etc. 

The latest version, Qwen3, has the following features:

- **Dense and Mixture-of-Experts (MoE) models**, available in 0.6B, 1.7B, 4B, 8B, 14B, 32B and 30B-A3B, 235B-A22B.
- **Seamless switching between thinking mode** (for complex logical reasoning, math, and coding) and **non-thinking mode** (for efficient, general-purpose chat) **within a single model**, ensuring optimal performance across various scenarios.
- **Significantly enhancement in reasoning capabilities**, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- **Superior human preference alignment**, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- **Expertise in agent capabilities**, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- **Support of 100+ languages and dialects** with strong capabilities for **multilingual instruction following** and **translation**.

For more information, please visit our:

* `Blog <https://qwenlm.github.io/>`__
* `GitHub <https://github.com/QwenLM>`__
* `Hugging Face <https://huggingface.co/Qwen>`__
* `ModelScope <https://modelscope.cn/organization/qwen>`__
* `Qwen3 Collection <https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f>`__

Join our community by joining our `Discord <https://discord.gg/yPEP2vHTu4>`__ and `WeChat <https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png>`__ group. We are looking forward to seeing you there!


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/quickstart
   getting_started/concepts
   getting_started/speed_benchmark
   getting_started/quantization_benchmark
   
.. toctree::
   :maxdepth: 1
   :caption: Inference
   :hidden:

   inference/transformers

.. toctree::
   :maxdepth: 1
   :caption: Run Locally
   :hidden:

   run_locally/llama.cpp
   run_locally/ollama
   run_locally/mlx-lm

.. toctree::
   :maxdepth: 1
   :caption: Deployment
   :hidden:

   deployment/sglang
   deployment/vllm
   deployment/tgi
   deployment/dstack
   deployment/skypilot
   deployment/openllm

.. toctree::
   :maxdepth: 1
   :caption: Quantization
   :hidden:

   quantization/awq
   quantization/gptq
   quantization/llama.cpp

.. toctree::
   :maxdepth: 1
   :caption: Training
   :hidden:

   training/llama_factory
   training/ms_swift
   training/verl
   training/axolotl

.. toctree::
   :maxdepth: 1
   :caption: Framework
   :hidden:

   framework/function_call
   framework/qwen_agent
   framework/LlamaIndex
   framework/Langchain
