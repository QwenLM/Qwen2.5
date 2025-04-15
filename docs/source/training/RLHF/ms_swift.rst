ms-swift
============================

Introduction to ms-swift
------------------------

ms-swift is the official large model and multimodal model training and deployment framework provided by the ModelScope community. 

GitHub repository: `ms-swift <https://github.com/modelscope/ms-swift>`__

The RLHF script in ms-swift has the following features:

- Support single-GPU and multi-GPU training
- Support full-parameter tuning, LoRA, Q-LoRA, and Dora
- Supports multiple RLHF algorithms including GRPO, DAPO, PPO, DPO, KTO, ORPO, CPO, and SimPO
- Supports both large language models (LLM) and multimodal models (MLLM)

For detailed support information, please refer to: `Supported Features <https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html#pre-training-and-fine-tuning>`__


Environment Setup
-----------------

1. Follow the instructions of `ms-swift <https://github.com/modelscope/ms-swift>`__, and build the environment.
2. Install these packages (Optional)::

      pip install deepspeed
      pip install math_verify==0.5.2
      pip install flash-attn --no-build-isolation


Data Preparation
----------------

ms-swift has built-in preprocessing logic for several datasets, which can be directly used for training via the ``--dataset`` parameter. For supported datasets, please refer to: `Supported Datasets <https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html#datasets>`__

You can also use local custom datasets by providing the local dataset path to the ``--dataset`` parameter.

Example dataset formats:

.. code-block:: text

   # llm
   {"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}]}
   {"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
   {"messages": [{"role": "user", "content": "What is your name?"}]}

   # mllm
   {"messages": [{"role": "user", "content": "<image>What is the difference between the two images?"}], "images": ["/xxx/x.jpg"]}
   {"messages": [{"role": "user", "content": "<image><image>What is the difference between the two images?"}], "images": ["/xxx/.yjpg", "/xxx/z.png"]}

GRPO Training Examples
----------------------

Single-GPU Configuration
------------------------

**LLM (Qwen2.5-7B):**

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   swift rlhf \
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
       --dataset 'AI-MO/NuminaMath-TIR#1000' \
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

   CUDA_VISIBLE_DEVICES=0 \
   swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
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
       --dataset 'lmms-lab/multimodal-open-r1-8k-verified#1000' \
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

Multi-GPU Training
------------------

**LLM Example with DeepSpeed:**

.. code-block:: bash

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
       --dataset 'AI-MO/NuminaMath-TIR#5000' \
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


Multi-GPU Deployment
--------------------

**MLLM Example with DeepSpeed:**

.. code-block:: bash


   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   NPROC_PER_NODE=8 \
   swift rlhf \
       --rlhf_type grpo \
       --model Qwen/Qwen2.5-VL-7B-Instruct \
       --reward_funcs accuracy format \
       --use_vllm true \
       --vllm_device auto \
       --vllm_gpu_memory_utilization 0.7 \
       --vllm_max_model_len 8192 \
       --num_infer_workers 8 \
       --train_type lora \
       --torch_dtype bfloat16 \
       --dataset 'lmms-lab/multimodal-open-r1-8k-verified#5000' \
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

Model Export & Deployment
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