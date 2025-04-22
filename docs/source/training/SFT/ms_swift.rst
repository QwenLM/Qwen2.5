ms-swift
===========================================

Introduction to ms-swift SFT
----------------------------


ms-swift is the official large model and multimodal model training and deployment framework provided by the ModelScope community. 

GitHub repository: `ms-swift <https://github.com/modelscope/ms-swift>`__

The SFT script in ms-swift has the following features:

- Flexible training options: single-GPU and multi-GPU support
- Efficient tuning methods: full-parameter, LoRA, Q-LoRA, and Dora
- Broad model compatibility: supports various LLM and MLLM architectures

For detailed model compatibility, see: `Supported Models <https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html>`__

Environment Setup
-----------------

1. Follow the instructions of `ms-swift <https://github.com/modelscope/ms-swift>`__, and build the environment.

2. Optional packages for advanced features::

      pip install deepspeed  # For multi-GPU training
      pip install flash-attn --no-build-isolation

Data Preparation
----------------

ms-swift supports multiple dataset formats:

.. code-block:: text

   # Standard messages format
   {"messages": [
       {"role": "system", "content": "<system-prompt>"},
       {"role": "user", "content": "<query1>"},
       {"role": "assistant", "content": "<response1>"}
   ]}

   # ShareGPT conversation format
   {"system": "<system-prompt>", "conversation": [
       {"human": "<query1>", "assistant": "<response1>"},
       {"human": "<query2>", "assistant": "<response2>"}
   ]}

   # Instruction tuning format
   {"system": "<system-prompt>", 
    "instruction": "<task-instruction>", 
    "input": "<additional-context>", 
    "output": "<expected-response>"}

   # Multimodal format (supports images, audio, video)
   {"messages": [
       {"role": "user", "content": "<image>Describe this image"},
       {"role": "assistant", "content": "<description>"}
   ], "images": ["/path/to/image.jpg"]}

For complete dataset formatting guidelines, see: `Custom Dataset Documentation <https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html>`__

Pre-built datasets are available at: `Supported Datasets <https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html#datasets>`__

Training Examples
-----------------

Single-GPU Training
-------------------

**LLM Example (Qwen2.5-7B-Instruct):**

.. code-block:: bash

    # 19GB
    CUDA_VISIBLE_DEVICES=0 \
    swift sft \
       --model Qwen/Qwen2.5-7B-Instruct \
       --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
       --train_type lora \
       --lora_rank 8 \
       --lora_alpha 32 \
       --target_modules all-linear \
       --torch_dtype bfloat16 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 16 \
       --learning_rate 1e-4 \
       --max_length 2048 \
       --eval_steps 50 \
       --save_steps 50 \
       --save_total_limit 2 \
       --logging_steps 5 \
       --output_dir output \
       --system 'You are a helpful assistant.' \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4 \
        --attn_impl flash_attn


**MLLM Example (Qwen2.5-VL-7B-Instruct):**

.. code-block:: bash

   # 18GB
    CUDA_VISIBLE_DEVICES=0 \
    MAX_PIXELS=602112 \
    swift sft \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
       --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite' \
       --train_type lora \
       --torch_dtype bfloat16 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 16 \
       --learning_rate 1e-4 \
       --max_length 2048 \
       --eval_steps 200 \
       --save_steps 200 \
       --save_total_limit 5 \
       --logging_steps 5 \
       --output_dir output \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4

Multi-GPU Training
------------------

**LLM Example with DeepSpeed:**

.. code-block:: bash

    # 18G*8
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=8 \
    nohup swift sft \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
        --train_type lora \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --torch_dtype bfloat16 \
        --deepspeed zero2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-4 \
        --max_length 2048 \
        --num_train_epochs 1 \
        --output_dir output \
        --attn_impl flash_attn

**MLLM Example with DeepSpeed:**

.. code-block:: bash

    # 17G*8
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=8 \
    MAX_PIXELS=602112 \
    nohup swift sft \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
       --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite' \
       --train_type lora \
       --deepspeed zero2 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 8 \
       --learning_rate 2e-5 \
       --max_length 4096 \
       --num_train_epochs 2 \
       --output_dir output \
        --attn_impl flash_attn

Model Export
-------------------------

**Merge LoRA Adapters:**

.. code-block:: bash

    swift export \
       --adapters output/checkpoint-xxx \
       --merge_lora true

**Push to ModelScope Hub:**

.. code-block:: bash

    swift export \
       --adapters output/checkpoint-xxx \
       --push_to_hub true \
       --hub_model_id '<your-namespace>/<model-name>' \
       --hub_token '<your-access-token>'