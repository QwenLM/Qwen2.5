# MS-SWIFT

ModelScope SWIFT (**ms-swift**) is the large model and multimodal large model training and deployment framework provided by the [ModelScope community](https://modelscope.cn/).

GitHub repository: [ms-swift](https://github.com/modelscope/ms-swift)

Features of using ms-swift for training LLM:

- **Model Types**: Supports 500+ plain-text large models and 200+ multimodal large models, covering the entire process from training to deployment.
- **Hardware Support**: Compatible with CPUs, RTX series GPUs, T4/V100, A10/A100/H100, Ascend NPUs, MPS, and more.
- **Training Methods**: Supports full-parameter fine-tuning, LoRA, QLoRA, DoRA, and other techniques.
- **Distributed Training**: Supports distributed training technologies such as DDP, device_map, DeepSpeed ZeRO-2/ZeRO-3, FSDP, and integrates parallelism techniques from Megatron, including Tensor Parallelism, Pipeline Parallelism, Sequence Parallelism, and Expert Parallelism.
- **RLHF Training**: Supports human alignment methods like DPO, GRPO, DAPO, RM, PPO, KTO, etc., for both plain-text and multimodal large models.

This article will demonstrate runnable training demos and provide the format for custom datasets. It includes how to use ms-swift for SFT and GRPO on Qwen3-8B, as well as using Megatron-SWIFT (ms-swift's integration of Megatron-LM) for SFT on Qwen3-30B-A3B. Through expert parallelism technology, MoE model training can be accelerated by nearly 10 times.

Before starting fine-tuning, ensure your environment is properly set up.

```shell
pip install ms-swift -U
# Install from source
pip install git+https://github.com/modelscope/ms-swift.git

pip install transformers -U

# Optional packages
pip install deepspeed # multi-GPU training
pip install liger-kernel # save GPU memory resources
pip install flash-attn --no-build-isolation
```

## Supervised Fine-Tuning (SFT)

### Data Preparation

The custom dataset format for SFT using ms-swift is as follows (the system field is optional). You can organize it into formats such as JSON, JSONL, or CSV. Specify `--dataset <dataset_path>` in the training script.

For complete dataset formatting guidelines, see: [Custom Dataset Documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)

- General format
    ```json
    {"messages": [
        {"role": "system", "content": "<system-prompt>"},
        {"role": "user", "content": "<query1>"},
        {"role": "assistant", "content": "<response1>"}
    ]}
    ```
- Format with think
    ```json
    {"messages": [
        {"role": "user", "content": "Where is the capital of Zhejiang?"},
        {"role": "assistant", "content": "<think>\n...\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
    ]}
    ```

If you want to train using data without a chain of thought but retain the model's reasoning ability, there are two approaches to minimize disruption during fine-tuning:

**Option 1**: During training, specify `--loss_scale ignore_empty_think` to ignore the loss calculation for `<think>\n\n</think>\n\n`, preventing the loss of reasoning ability. Refer to the training script [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh). The custom dataset format is as follows:

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

**Option 2**: Add `/no_think` to the query in the dataset to avoid the loss of reasoning ability. Refer to the training script [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo2.sh). The custom dataset format is as follows:

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang? /no_think"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

### 30-Minute Self-Cognition Fine-Tuning

This section introduces a 30-minute self-cognition fine-tuning process for the Qwen3-8B model. The required GPU memory is 22GB, and it can be run on the A10 provided by [ModelScope's free compute resources](https://modelscope.cn/my/mynotebook).

After training, the model will identify itself as "swift-robot," trained by "swift," instead of its original self-cognition as "Qwen," trained by Alibaba Cloud.

If you need to train in an offline environment, you can manually download the model and dataset and specify `--model <model-path>` and `--dataset <dataset-dir>`. The dataset can be found on [Modelscope Hub](https://modelscope.cn/datasets/swift/self-cognition).

For the meaning of each parameter in the training script, please refer to the [Command-line parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:qwen3#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

After fine-tuning, you can use the following script to test the fine-tuning results. Note that the `--adapters` section needs to be modified to the directory path of the last saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

```text
<<< who are you?
<think>
Okay, the user asked, "who are you?" I need to introduce myself. Let me start by stating my name, swift-robot. Then, I should mention that I'm an AI assistant developed by swift. I should explain my purpose, which is to provide information and assistance. I should also highlight my capabilities, like answering questions, generating text, and engaging in conversation. It's important to keep the tone friendly and approachable. Maybe add something about being here to help and encourage the user to ask anything. Let me check if I covered all the key points: name, developer, purpose, capabilities, and a welcoming statement. Yeah, that should do it. Now, let me put that into a concise and friendly response.
</think>

Hello! I am swift-robot, an artificial intelligence assistant developed by swift. My purpose is to provide information and assistance to users like you. I can answer questions, generate text, and engage in conversations on a wide range of topics. I am here to help, so feel free to ask me anything you need!
```

By default, ms-swift will use the ModelScope community to download models and datasets. If you want to use the HuggingFace community, you need to additionally specify `--use_hf true`.

Merge LoRA weights:

```shell
swift export \
    --adapters output/checkpoint-xxx \
    --merge_lora true
```

Push the model to ModelScope/HuggingFace:

```shell
# If you are pushing the complete weights, you need to change `--adapters` to `--model`.
# The Modelscope hub_token can be found here: https://modelscope.cn/my/myaccesstoken
swift export \
    --adapters output/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<hub-model-id>' \
    --hub_token '<hub-token>' \
    --use_hf false
```

If you want to use multiple GPUs for training, the following provides a demo for multi-GPU training:

```shell
# 4 * 60GB
# You can run the experiment by setting `--dataset AI-ModelScope/alpaca-gpt4-data-en`.
# Note: If you want to specify `--packing true`, you must additionally set `--attn_impl flash_attn`.

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset '<your-dataset>' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
```

## Reinforcement Learning (RL)

ms-swift supports RLHF methods such as DPO, GRPO, DAPO, PPO, KTO, and more. This section will focus on an example of using ms-swift to perform GRPO training for Qwen3-8B.

For detailed RLHF support information, please refer to: [Supported Features](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html).

### Environment Setup

In addition to installing the ms-swift related dependencies introduced above, the following dependencies also need to be installed:

```shell
pip install "math_verify==0.5.2"
pip install vllm
```

### Data Preparation

The dataset format for GRPO training using ms-swift is similar to that of SFT, except that the assistant part of the last round is not required. If using accuracy as a reward, a `solution` column is needed to calculate the accuracy.

Example Dataset Formats:

```json
{"messages": [{"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```

For dataset preparation for other RLHF algorithms, see: [Custom Dataset Documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#rlhf).

Notes on Dataset Requirements:

- **Reward Function Calculation**: The dataset format depends on the reward function being used. Additional columns may be required to support specific reward calculations. For instance:

  - When using the built-in accuracy or cosine similarity reward, the dataset must include a `solution` column to calculate the accuracy of the responses.
  - Other columns in the dataset will be passed as `**kwargs` to the reward function for additional customization.

- **Customizing the Reward Function**: To adapt the reward function to your specific needs, you can refer to the following resource: [External Reward Plugin](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin). This plugin provides examples and templates for implementing custom reward functions.

During the training process, we use vLLM to accelerate the sampling process. By setting `num_infer_workers=8`, we deploy a vLLM engine for each device to speed up the sampling process.

```shell
# 70G*8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 4096 \
    --vllm_max_model_len 8192 \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --deepspeed zero3 \
    --num_infer_workers 8 \
    --tensor_parallel_size 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --log_completions true \
    --overlong_filter true
```

## Megatron-SWIFT

ms-swift incorporates Megatron parallelism techniques to accelerate the training of large models. The supported models can be found in the [Supported Models Documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

For environment preparation and the conversion between HF and MCore model weights, you can refer to the [Megatron-SWIFT Training Documentation](https://swift.readthedocs.io/en/latest/Instruction/Megatron-SWIFT-Training.html). These topics will not be elaborated here.

We will use Alibaba Cloud DLC to start the training The training environment consists of 2 machines with 8 * 80GiB A800 GPUs. For more information on multi-node startup methods, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node).

```shell
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# Ensure that the weight-saving paths on the two nodes are identical.
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true
```

The custom dataset format is the same as `swift sft`, which can be found in the previous section. Simply specify `--dataset <dataset_path>`.

The following is a comparison of training speed and GPU memory usage between `megatron sft` and `swift sft` for full-parameter fine-tuning of the Qwen3-30B-A3B model:

|                  | Megatron-LM | DeepSpeed-ZeRO2 | DeepSpeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 9.6s/it     | -               | 91.2s/it        |
| GPU Memory Usage | 16 * 60GiB  | OOM             | 16 * 80GiB      |

## Conclusion

The above is the best practice for training Qwen3 series models using ms-swift. If you encounter any difficulties during use, please join the discussion in [this issue](https://github.com/modelscope/ms-swift/issues/4030).
