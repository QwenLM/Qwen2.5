Axolotl Qwen3 Training Guide
=============================

This guide will help you get started with post-training (SFT, RLHF, RM, PRM) for Qwen3 / Qwen3_MOE using Axolotl, and covers optimizations to enable for better performance.

Requirements
============

- **GPU:** NVIDIA Ampere (or newer) for ``bf16`` and ``Flash Attention``, or AMD GPU
- **Python:** ≥3.11
- **CUDA:** ≥12.4 (for NVIDIA GPUs)

Installation
============

You can install Axolotl using PyPI, Conda, Git, Docker, or launch a cloud environment.

.. important::

   Install PyTorch *before* installing Axolotl to ensure CUDA compatibility.

For the latest instructions, see the official `Axolotl Installation Guide <https://docs.axolotl.ai/docs/installation.html>`_.

Quickstart
==========

SFT
---

We have provided a sample YAML config for SFT with Qwen/Qwen3-32B: `SFT 32B QLoRA config <https://github.com/axolotl-ai-cloud/axolotl/blob/v0.9.2/examples/qwen3/32b-qlora.yaml>`_.

.. code:: bash

   # Train the model
   axolotl train path/to/32b-qlora.yaml

   # Merge LoRA weights with the base model
   # This will create a new `merged` directory under `{output_dir}`
   axolotl merge-lora path/to/32b-qlora.yaml

.. tip::

   To train a smaller model, edit the ``base_model`` in your config:

   .. code:: yaml

      base_model: Qwen/Qwen3-8B

Qwen3 works with all Axolotl features including ``Flash Attention``, ``bf16``, ``LoRA``, ``torch_compile``, and ``QLoRA``.

To run on more than single GPU, please take a look at the `Multi-GPU Training Guide <https://docs.axolotl.ai/docs/multi-gpu.html>`_ or `Multi-node Training Guide <https://docs.axolotl.ai/docs/multi-node.html>`_.

RLHF
----

See the `RLHF Guide <https://docs.axolotl.ai/docs/rlhf.html>`_ for required dataset formats and examples for each method.

RM/PRM
------

Please refer to the `Reward Modelling Guide <https://docs.axolotl.ai/docs/reward_modelling.html>`_ for required dataset formats and config examples.

Dataset
=======

By default, the example config uses the ``mlabonne/FineTome-100k`` dataset (from HuggingFace Hub). You can substitute any dataset of your own.

SFT Dataset Format
------------------

Axolotl handles various SFT dataset formats, but the current **recommended** format (for use with ``chat_template``) is the OpenAI Messages format:

.. code:: json

   [
     {
       "messages": [
         {
           "role": "user",
           "content": "What is Qwen3?"
         },
         {
           "role": "assistant",
           "content": "Qwen3 is a language model..."
         }
       ]
     }
   ]

Use this in your config:

.. code:: yaml

   datasets:
     - path: path/to/your/dataset.json
       type: chat_template

You can also load datasets from multiple sources: HuggingFace Hub, local files, directories, S3, GCS, Azure, etc.

See the `Dataset Loading Guide <https://docs.axolotl.ai/docs/dataset_loading.html>`_ for more details.

To load different dataset formats, refer to the `SFT Dataset Formats Guide <https://docs.axolotl.ai/docs/dataset-formats/#supervised-fine-tuning-sft>`_.

Optimizations
=============

With Qwen3/Qwen3_MOE, you can leverage Axolotl's custom optimizations for improved speed and reduced memory usage:

- `Cut Cross Entropy <https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy>`_
- `Liger Kernels <https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels>`_
- (LoRA/QLoRA only): `LoRA Kernels Optimization <https://docs.axolotl.ai/docs/lora_optims.html>`_

Additional Suggestions
=======================

Troubleshooting
---------------

- Ensure your CUDA version matches your GPU and PyTorch version.
- If running into out-of-memory issues, try reducing your batch size, enable the optimizations above, or reduce sequence length.
- Qwen3 MoE may have slower training due to the upstream transformer's handling of MoE layers.
- For help, check the help channel on `Axolotl Discord <https://discord.gg/7m9sfhzaf3>`_ or create a Discussion on `Axolotl GitHub <https://github.com/axolotl-ai-cloud/axolotl>`_.

Links
-----

- `Axolotl Documentation <https://docs.axolotl.ai/>`_
- `Axolotl Discord <https://discord.gg/7m9sfhzaf3>`_
- `Axolotl GitHub <https://github.com/axolotl-ai-cloud/axolotl>`_
- `Axolotl Website <https://axolotl.ai>`_
