# verl

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2) paper.

GitHub repository: [verl](https://github.com/volcengine/verl)

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex Post-Training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.
- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc
- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.
- **Ready integration with popular HuggingFace models**: verl supports popular LLM models, including Qwen, Llama, and more.

verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.
- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

Next, we will introduce how to use verl for training Qwen3 models.

## Reinforcement Learning (RL)

Now, verl supports various combinations of training frameworks and inference frameworks, including FSDP, Megatron-LM, vLLM, SGLang, etc. verl also supports training with multiple algorithms such as PPO, GRPO, DAPO, etc.

### Step1: Environment and Training Preparation

You can follow verl's [installation guide](https://verl.readthedocs.io/en/latest/start/install.html) to complete the environment configuration.

Data preparation can be done by running the following command:

```shell
git clone https://github.com/volcengine/verl.git
cd verl
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

Model download can be done using the following command:

```shell
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-1.7B')"
```

### Step2: Start Training

In verl, training frameworks and inference frameworks can be combined freely, as long as the training framework and inference framework themselves support model training and inference tasks, so that verl can support RL-related training.

Below is an example using FSDP and vLLM to demonstrate how to train Qwen3 models in verl. We chose Qwen3-1.7B as the example, as it only requires a single 80GB GPU and a machine with more than 64GB of memory to start training.

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=80 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_1_7b_function_rm' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```

## Finally

If you encounter any difficulties during use, please join the discussion at [GitHub](https://github.com/volcengine/verl/discussions).
