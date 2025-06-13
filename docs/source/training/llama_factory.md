# LLaMA-Factory

:::{attention}
To be updated for Qwen3.
:::

Here we provide a script for supervised finetuning Qwen2.5 with
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). This
script for supervised finetuning (SFT) has the following features:

- Support single-GPU and multi-GPU training;
- Support full-parameter tuning, LoRA, Q-LoRA, Dora.

In the following, we introduce more details about the usage of the
script.

## Installation

Before you start, make sure you have installed the following packages:

1. Follow the instructions of
   [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and build
   the environment.
2. Install these packages (Optional):

```
pip install deepspeed
pip install flash-attn --no-build-isolation
```

3. If you want to use
   [FlashAttention-2](https://github.com/Dao-AILab/flash-attention),
   make sure your CUDA is 11.6 and above.

## Data Preparation

LLaMA-Factory provides several training datasets in `data` folder, you
can use it directly. If you are using a custom dataset, please prepare
your dataset as follows.

1. Organize your data in a **json** file and put your data in `data`
   folder. LLaMA-Factory supports dataset in `alpaca` or `sharegpt`
   format.

- The dataset in `alpaca` format should follow the below format:

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

- The dataset in `sharegpt` format should follow the below format:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

2. Provide your dataset definition in `data/dataset_info.json` in the
   following format .

- For `alpaca` format dataset, the columns in `dataset_info.json`
  should be:

```json
"dataset_name": {
  "file_name": "dataset_name.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

- For `sharegpt` format dataset, the columns in `dataset_info.json`
  should be:

```json
"dataset_name": {
    "file_name": "dataset_name.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
```

## Training

Execute the following training command:

```bash
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path $MODEL_PATH \
    --dataset your_dataset \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16
```

and enjoy the training process. To make changes to your training, you
can modify the arguments in the training command to adjust the
hyperparameters. One argument to note is `cutoff_len`, which is the
maximum length of the training data. Control this parameter to avoid OOM
error.

## Merge LoRA

If you train your model with LoRA, you probably need to merge adapter
parameters to the main branch. Run the following command to perform the
merging of LoRA adapters.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path path_to_base_model \
    --adapter_name_or_path path_to_adapter \
    --template qwen \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```

## Conclusion

The above content is the simplest way to use LLaMA-Factory to train
Qwen. Feel free to dive into the details by checking the official repo!
