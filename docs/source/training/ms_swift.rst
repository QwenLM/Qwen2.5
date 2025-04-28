SWIFT
===========================================

.. attention:: 
    To be updated for Qwen3.

ModelScope SWIFT (ms-swift) is the official large model and multimodal model training and deployment framework provided by the ModelScope community. 

GitHub repository: `ms-swift <https://github.com/modelscope/ms-swift>`__

Supervised Fine-Tuning (SFT)
-----------------------------

The SFT script in ms-swift has the following features:

- Flexible training options: single-GPU and multi-GPU support
- Efficient tuning methods: full-parameter, LoRA, Q-LoRA, and Dora
- Broad model compatibility: supports various LLM and MLLM architectures

For detailed model compatibility, see: `Supported Models <https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html>`__

Environment Setup
+++++++++++++++++

1. Follow the instructions of `ms-swift <https://github.com/modelscope/ms-swift>`__, and build the environment.

2. Optional packages for advanced features::

      pip install deepspeed  # For multi-GPU training
      pip install flash-attn --no-build-isolation

Data Preparation
+++++++++++++++++

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
+++++++++++++++++

Single-GPU Training
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

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



Reinforcement Learning (RL)
-----------------------------

The RL script in ms-swift has the following features:

- Support single-GPU and multi-GPU training
- Support full-parameter tuning, LoRA, Q-LoRA, and Dora
- Supports multiple RL algorithms including GRPO, DAPO, PPO, DPO, KTO, ORPO, CPO, and SimPO
- Supports both large language models (LLM) and multimodal models (MLLM)

For detailed support information, please refer to: `Supported Features <https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html#pre-training-and-fine-tuning>`__


Environment Setup
++++++++++++++++++

1. Follow the instructions of `ms-swift <https://github.com/modelscope/ms-swift>`__, and build the environment.
2. Install these packages (Optional)::

      pip install deepspeed
      pip install math_verify==0.5.2
      pip install flash-attn --no-build-isolation
      pip install vllm


Data Preparation
++++++++++++++++

ms-swift has built-in preprocessing logic for several datasets, which can be directly used for training via the ``--dataset`` parameter. For supported datasets, please refer to: `Supported Datasets <https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html#datasets>`__

You can also use local custom datasets by providing the local dataset path to the ``--dataset`` parameter.

Example Dataset Formats:

.. code-block:: text

   # llm
   {"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}]}
   {"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
   {"messages": [{"role": "user", "content": "What is your name?"}]}

   # mllm
   {"messages": [{"role": "user", "content": "<image>What is the difference between the two images?"}], "images": ["/xxx/x.jpg"]}
   {"messages": [{"role": "user", "content": "<image><image>What is the difference between the two images?"}], "images": ["/xxx/y.jpg", "/xxx/z.png"]}

Notes on Dataset Requirements

1. Reward Function Calculation: Depending on the reward function being used, additional columns may be required in the dataset. For example:

      When using the built-in accuracy/cosine reward, the dataset must include a ``solution`` column to compute accuracy.
      The other columns in the dataset will also be passed to the `kwargs` of the reward function. 

2. Customizing the Reward Function: To tailor the reward function to your specific needs, you can refer to the following resource: `external reward plugin <https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin>`__


GRPO Training Examples
+++++++++++++++++++++++

Single-GPU Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

**LLM (Qwen2.5-7B):**

.. code-block:: bash
   
   # 42G
   CUDA_VISIBLE_DEVICES=0 \
   nohup swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-7B \
       --vllm_gpu_memory_utilization 0.5 \
       --use_vllm true \
       --sleep_level 1 \
       --offload_model true \
       --offload_optimizer true \
       --gc_collect_after_offload true \
       --reward_funcs accuracy format \
       --train_type lora \
       --lora_rank 8 \
       --lora_alpha 32 \
       --target_modules all-linear \
       --torch_dtype bfloat16 \
       --dataset 'AI-MO/NuminaMath-TIR' \
       --max_completion_length 1024 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 4 \
       --learning_rate 1e-5 \
       --gradient_accumulation_steps 1 \
       --eval_steps 100 \
       --save_steps 100 \
       --save_total_limit 2 \
       --logging_steps 5 \
       --max_length 2048 \
       --output_dir output \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4 \
       --dataset_num_proc 4 \
       --num_generations 4 \
       --temperature 0.9 \
       --system 'examples/train/grpo/prompt.txt' \
       --log_completions true

**MLLM (Qwen2.5-VL-7B-Instruct):**

.. code-block:: bash

   # 55G
   CUDA_VISIBLE_DEVICES=0 \
   MAX_PIXELS=602112 \
   swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
       --vllm_gpu_memory_utilization 0.5 \
       --use_vllm true \
       --sleep_level 1 \
       --offload_model true \
       --offload_optimizer true \
       --gc_collect_after_offload true \
       --external_plugins examples/train/grpo/plugin/plugin.py \
       --reward_funcs external_r1v_acc format \
       --train_type lora \
       --lora_rank 8 \
       --lora_alpha 32 \
       --target_modules all-linear \
       --torch_dtype bfloat16 \
       --dataset 'lmms-lab/multimodal-open-r1-8k-verified' \
       --vllm_max_model_len 4196 \
       --max_completion_length 1024 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 4 \
       --learning_rate 1e-5 \
       --gradient_accumulation_steps 1 \
       --eval_steps 100 \
       --save_steps 100 \
       --save_total_limit 2 \
       --logging_steps 5 \
       --output_dir output \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4 \
       --dataset_num_proc 4 \
       --num_generations 4 \
       --temperature 0.9 \
       --system 'examples/train/grpo/prompt.txt' \
       --log_completions true

Multi-GPU Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

**LLM Example with DeepSpeed:**

.. code-block:: bash

   # 60G*8
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   NPROC_PER_NODE=8 \
   swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-7B \
       --reward_funcs accuracy format \
       --use_vllm true \
       --vllm_device auto \
       --vllm_gpu_memory_utilization 0.7 \
       --vllm_max_model_len 8192 \
       --num_infer_workers 8 \
       --train_type lora \
       --torch_dtype bfloat16 \
       --dataset 'AI-MO/NuminaMath-TIR' \
       --max_completion_length 2048 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --learning_rate 1e-6 \
       --gradient_accumulation_steps 2 \
       --eval_steps 200 \
       --save_steps 200 \
       --save_total_limit 2 \
       --logging_steps 5 \
       --max_length 4096 \
       --output_dir output \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4 \
       --dataset_num_proc 4 \
       --num_generations 8 \
       --temperature 0.9 \
       --system 'examples/train/grpo/prompt.txt' \
       --deepspeed zero2 \
       --log_completions true \
       --sleep_level 1 \
       --offload_model true \
       --offload_optimizer true \
       --gc_collect_after_offload true \
       --log_completions true

**MLLM Example with DeepSpeed:**

.. code-block:: bash

   # 60G*8
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   NPROC_PER_NODE=8 \
   nohup swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
       --external_plugins examples/train/grpo/plugin/plugin.py \ 
       --reward_funcs external_r1v_acc format \
       --use_vllm true \
       --vllm_device auto \
       --vllm_gpu_memory_utilization 0.7 \
       --vllm_max_model_len 8192 \
       --num_infer_workers 8 \
       --train_type lora \
       --torch_dtype bfloat16 \
       --dataset 'lmms-lab/multimodal-open-r1-8k-verified' \
       --max_completion_length 2048 \
       --num_train_epochs 1 \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --learning_rate 1e-6 \
       --gradient_accumulation_steps 2 \
       --eval_steps 200 \
       --save_steps 200 \
       --save_total_limit 2 \
       --logging_steps 5 \
       --vllm_max_model_len 4196 \
       --output_dir output \
       --warmup_ratio 0.05 \
       --dataloader_num_workers 4 \
       --dataset_num_proc 4 \
       --num_generations 8 \
       --temperature 0.9 \
       --system 'examples/train/grpo/prompt.txt' \
       --deepspeed zero2 \
       --log_completions true \
       --sleep_level 1 \
       --offload_model true \
       --offload_optimizer true \
       --gc_collect_after_offload true \
       --log_completions true


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