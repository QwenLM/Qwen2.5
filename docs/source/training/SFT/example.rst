Example
====================================================

Here we provide a very simple script for supervised finetuning, which is revised from the training
script in ```Fastchat`` <https://github.com/lm-sys/FastChat>`__. The
script is used to finetune Qwen with Hugging Face Trainer. You can check
the script
`here <https://github.com/QwenLM/Qwen1.5/blob/main/finetune.py>`__. This
script for supervised finetuning (SFT) has the following features:

-  Support single-GPU and multi-GPU training;
-  Support full-parameter tuning,
   `LoRA <https://arxiv.org/abs/2106.09685>`__, and
   `Q-LoRA <https://arxiv.org/abs/2305.14314>`__.

In the following, we introduce more details about the usage of the
script.

Installation
------------

Before you start, make sure you have installed the following packages:

.. code:: bash

   pip install peft deepspeed optimum accelerate

Data Preparation
----------------

For data preparation, we advise you to organize the data in a jsonl
file, where each line is a dictionary as demonstrated below:

.. code:: json

   {
       "type": "chatml",
       "messages": [
           {
               "role": "system",
               "content": "You are a helpful assistant."
           },
           {
               "role": "user",
               "content": "Tell me something about large language models."
           },
           {
               "role": "assistant",
               "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
           }
       ],
       "source": "unknown"
   }

.. code:: json

   {
       "type": "chatml",
       "messages": [
           {
               "role": "system",
               "content": "You are a helpful assistant."
           },
           {
               "role": "user",
               "content": "What is your name?"
           },
           {
               "role": "assistant",
               "content": "My name is Qwen."
           }
       ],
       "source": "self-made"
   }

Above are two examples of each data sample in the dataset. Each sample
is a JSON object with the following fields: ``type``, ``messages`` and
``source``. ``messages`` is required while the others are optional for
you to label your data format and data source. The ``messages`` field is
a list of JSON objects, each of which has two fields: ``role`` and
``content``. ``role`` can be ``system``, ``user``, or ``assistant``.
``content`` is the text of the message. ``source`` is the source of the
data, which can be ``self-made``, ``alpaca``, ``open-hermes``, or any
other string.

To make the jsonl file, you can use ``json`` to save a list of
dictionaries to the jsonl file:

.. code:: python

   import json

   with open('data.jsonl', 'w') as f:
       for sample in samples:
           f.write(json.dumps(sample) + '\n')

Quickstart
----------

For you to start finetuning quickly, we directly provide a shell script
for you to run without paying attention to details. You need
different hyperparameters for different types of training, e.g.,
single-GPU / multi-GPU training, full-parameter tuning, LoRA, or Q-LoRA.



.. code:: bash

   cd examples/sft
   bash finetune.sh -m <model_path> -d <data_path> --deepspeed <config_path> [--use_lora True] [--q_lora True]


Specify the ``<model_path>`` for your model, ``<data_path>`` for your
data, and ``<config_path>`` for your deepspeed configuration. 
If you use LoRA or Q-LoRA, just add ``--use_lora True`` or
``--q_lora True`` based on your requirements.
This is the simplest way to start finetuning. If you want to change more
hyperparameters, you can dive into the script and modify those
parameters.

Advanced Usages
---------------

In this section, we introduce the details of the scripts, including the
core python script as well as the corresponding shell script.

Shell Script
~~~~~~~~~~~~~

Before we introduce the python code, we provide a brief introduction to
the shell script with commands. We provide some guidance inside the
shell script and here we take ``finetune.sh`` as an example.

To set up the environment variables for distributed training (or
single-GPU training), specify the following variables:
``GPUS_PER_NODE``, ``NNODES``, ``NODE_RANK``, ``MASTER_ADDR``, and
``MASTER_PORT``. No need to worry too much about them as we provide the
default settings for you. In the command, you can pass in the argument
``-m`` and ``-d`` to specify the model path and data path, respectively.
You can also pass in the argument ``--deepspeed`` to specify the
deepspeed configuration file. We provide two configuration files for
ZeRO2 and ZeRO3, and you can choose one based on your requirements. In
most cases, we recommend using ZeRO3 for multi-GPU training except for
Q-LoRA, where we recommend using ZeRO2.

There are a series of hyperparameters to tune. Passing in ``--bf16`` or
``--fp16`` to specify the precision for mixed precision training. 
The other significant hyperparameters include:

-  ``--output_dir``: the path of your output models or adapters.
-  ``--num_train_epochs``: the number of training epochs.
-  ``--gradient_accumulation_steps``: the number of gradient
   accumulation steps.
-  ``--per_device_train_batch_size``: the batch size per GPU for
   training, and the total batch size is equalt to
   ``per_device_train_batch_size`` :math:`\times` ``number_of_gpus``
   :math:`\times` ``gradient_accumulation_steps``.
-  ``--learning_rate``: the learning rate.
-  ``--warmup_steps``: the number of warmup steps.
-  ``--lr_scheduler_type``: the type of learning rate scheduler.
-  ``--weight_decay``: the value of weight decay.
-  ``--adam_beta2``: the value of :math:`\beta_2` in Adam.
-  ``--model_max_length``: the maximum sequence length.
-  ``--use_lora``: whether to use LoRA. Adding ``--q_lora`` can enable
   Q-LoRA.
-  ``--gradient_checkpointing``: whether to use gradient checkpointing.

Python Script
~~~~~~~~~~~~~

In this script, we mainly use ``trainer`` from HF and ``peft`` to train
our models. We also use ``deepspeed`` to accelerate the training
process. The script is very simple and easy to understand.

.. code:: python

   @dataclass
   @dataclass
   class ModelArguments:
       model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


   @dataclass
   class DataArguments:
       data_path: str = field(
           default=None, metadata={"help": "Path to the training data."}
       )
       eval_data_path: str = field(
           default=None, metadata={"help": "Path to the evaluation data."}
       )
       lazy_preprocess: bool = False


   @dataclass
   class TrainingArguments(transformers.TrainingArguments):
       cache_dir: Optional[str] = field(default=None)
       optim: str = field(default="adamw_torch")
       model_max_length: int = field(
           default=8192,
           metadata={
               "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
           },
       )
       use_lora: bool = False


   @dataclass
   class LoraArguments:
       lora_r: int = 64
       lora_alpha: int = 16
       lora_dropout: float = 0.05
       lora_target_modules: List[str] = field(
           default_factory=lambda: [
               "q_proj",
               "k_proj",
               "v_proj",
               "o_proj",
               "up_proj",
               "gate_proj",
               "down_proj",
           ]
       )
       lora_weight_path: str = ""
       lora_bias: str = "none"
       q_lora: bool = False

The classes for arguments allow you to specify hyperparameters for
model, data, training, and additionally LoRA if you use LoRA or Q-LoRA
to train your model. Specifically, ``model-max-length`` is a key
hyperparameter that determines your maximum sequence length of your
training data.

``LoRAArguments`` includes the hyperparameters for LoRA or Q-LoRA:

-  ``lora_r``: the rank for LoRA;
-  ``lora_alpha``: the alpha value for LoRA;
-  ``lora_dropout``: the dropout rate for LoRA;
-  ``lora_target_modules``: the target modules for LoRA. By default we
   tune all linear layers;
-  ``lora_weight_path``: the path to the weight file for LoRA;
-  ``lora_bias``: the bias for LoRA;
-  ``q_lora``: whether to use Q-LoRA.


.. code:: python

   def maybe_zero_3(param):
       if hasattr(param, "ds_id"):
           assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
           with zero.GatheredParameters([param]):
               param = param.data.detach().cpu().clone()
       else:
           param = param.detach().cpu().clone()
       return param


   # Borrowed from peft.utils.get_peft_model_state_dict
   def get_peft_state_maybe_zero_3(named_params, bias):
       if bias == "none":
           to_return = {k: t for k, t in named_params if "lora_" in k}
       elif bias == "all":
           to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
       elif bias == "lora_only":
           to_return = {}
           maybe_lora_bias = {}
           lora_bias_names = set()
           for k, t in named_params:
               if "lora_" in k:
                   to_return[k] = t
                   bias_name = k.split("lora_")[0] + "bias"
                   lora_bias_names.add(bias_name)
               elif "bias" in k:
                   maybe_lora_bias[k] = t
           for k, t in maybe_lora_bias:
               if bias_name in lora_bias_names:
                   to_return[bias_name] = t
       else:
           raise NotImplementedError
       to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
       return to_return


   def safe_save_model_for_hf_trainer(
       trainer: transformers.Trainer, output_dir: str, bias="none"
   ):
       """Collects the state dict and dump to disk."""
       # check if zero3 mode enabled
       if deepspeed.is_deepspeed_zero3_enabled():
           state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
       else:
           if trainer.args.use_lora:
               state_dict = get_peft_state_maybe_zero_3(
                   trainer.model.named_parameters(), bias
               )
           else:
               state_dict = trainer.model.state_dict()
       if trainer.args.should_save and trainer.args.local_rank == 0:
           trainer._save(output_dir, state_dict=state_dict)

The method ``safe_save_model_for_hf_trainer``, which uses
``get_peft_state_maybe_zero_3``, helps tackle the problems in saving
models trained either with or without ZeRO3.

.. code:: python

   def preprocess(
       messages,
       tokenizer: transformers.PreTrainedTokenizer,
       max_len: int,
   ) -> Dict:
       """Preprocesses the data for supervised fine-tuning."""

       texts = []
       for i, msg in enumerate(messages):
           texts.append(
               tokenizer.apply_chat_template(
                   msg,
                   tokenize=True,
                   add_generation_prompt=False,
                   padding=True,
                   max_length=max_len,
                   truncation=True,
               )
           )
       input_ids = torch.tensor(texts, dtype=torch.int)
       target_ids = input_ids.clone()
       target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
       attention_mask = input_ids.ne(tokenizer.pad_token_id)

       return dict(
           input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
       )

For data preprocessing, we use ``preprocess`` to organize the data.
Specifically, we apply our ChatML template to the texts. If you prefer
other chat templates, you can use others, e.g., by still applying
``apply_chat_template()`` with another tokenizer. The chat template is
stored in the ``tokenizer_config.json`` in the HF repo. Additionally, we
pad the sequence of each sample to the maximum length for training.

.. code:: python

   class SupervisedDataset(Dataset):
       """Dataset for supervised fine-tuning."""

       def __init__(
           self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
       ):
           super(SupervisedDataset, self).__init__()

           rank0_print("Formatting inputs...")
           messages = [example["messages"] for example in raw_data]
           data_dict = preprocess(messages, tokenizer, max_len)

           self.input_ids = data_dict["input_ids"]
           self.target_ids = data_dict["target_ids"]
           self.attention_mask = data_dict["attention_mask"]

       def __len__(self):
           return len(self.input_ids)

       def __getitem__(self, i) -> Dict[str, torch.Tensor]:
           return dict(
               input_ids=self.input_ids[i],
               labels=self.labels[i],
               attention_mask=self.attention_mask[i],
           )


   class LazySupervisedDataset(Dataset):
       """Dataset for supervised fine-tuning."""

       def __init__(
           self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
       ):
           super(LazySupervisedDataset, self).__init__()
           self.tokenizer = tokenizer
           self.max_len = max_len

           rank0_print("Formatting inputs...Skip in lazy mode")
           self.tokenizer = tokenizer
           self.raw_data = raw_data
           self.cached_data_dict = {}

       def __len__(self):
           return len(self.raw_data)

       def __getitem__(self, i) -> Dict[str, torch.Tensor]:
           if i in self.cached_data_dict:
               return self.cached_data_dict[i]

           ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer, self.max_len)
           ret = dict(
               input_ids=ret["input_ids"][0],
               labels=ret["target_ids"][0],
               attention_mask=ret["attention_mask"][0],
           )
           self.cached_data_dict[i] = ret

           return ret


   def make_supervised_data_module(
       tokenizer: transformers.PreTrainedTokenizer,
       data_args,
       max_len,
   ) -> Dict:
       """Make dataset and collator for supervised fine-tuning."""
       dataset_cls = (
           LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
       )
       rank0_print("Loading data...")

       train_data = []
       with open(data_args.data_path, "r") as f:
           for line in f:
               train_data.append(json.loads(line))
       train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)

       if data_args.eval_data_path:
           eval_data = []
           with open(data_args.eval_data_path, "r") as f:
               for line in f:
                   eval_data.append(json.loads(line))
           eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
       else:
           eval_dataset = None

       return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

Then we utilize ``make_supervised_data_module`` by using
``SupervisedDataset`` or ``LazySupervisedDataset`` to build the dataset.

.. code:: python

   def train():
       global local_rank

       parser = transformers.HfArgumentParser(
           (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
       )
       (
           model_args,
           data_args,
           training_args,
           lora_args,
       ) = parser.parse_args_into_dataclasses()

       # This serves for single-gpu qlora.
       if (
           getattr(training_args, "deepspeed", None)
           and int(os.environ.get("WORLD_SIZE", 1)) == 1
       ):
           training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

       local_rank = training_args.local_rank

       device_map = None
       world_size = int(os.environ.get("WORLD_SIZE", 1))
       ddp = world_size != 1
       if lora_args.q_lora:
           device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
           if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
               logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

       model_load_kwargs = {
           "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
       }

       compute_dtype = (
           torch.float16
           if training_args.fp16
           else (torch.bfloat16 if training_args.bf16 else torch.float32)
       )

       # Load model and tokenizer
       config = transformers.AutoConfig.from_pretrained(
           model_args.model_name_or_path,
           cache_dir=training_args.cache_dir,
       )
       config.use_cache = False

       model = AutoModelForCausalLM.from_pretrained(
           model_args.model_name_or_path,
           config=config,
           cache_dir=training_args.cache_dir,
           device_map=device_map,
           quantization_config=BitsAndBytesConfig(
               load_in_4bit=True,
               bnb_4bit_use_double_quant=True,
               bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=compute_dtype,
           )
           if training_args.use_lora and lora_args.q_lora
           else None,
           **model_load_kwargs,
       )
       tokenizer = AutoTokenizer.from_pretrained(
           model_args.model_name_or_path,
           cache_dir=training_args.cache_dir,
           model_max_length=training_args.model_max_length,
           padding_side="right",
           use_fast=False,
       )

       if training_args.use_lora:
           lora_config = LoraConfig(
               r=lora_args.lora_r,
               lora_alpha=lora_args.lora_alpha,
               target_modules=lora_args.lora_target_modules,
               lora_dropout=lora_args.lora_dropout,
               bias=lora_args.lora_bias,
               task_type="CAUSAL_LM",
           )
           if lora_args.q_lora:
               model = prepare_model_for_kbit_training(
                   model, use_gradient_checkpointing=training_args.gradient_checkpointing
               )

           model = get_peft_model(model, lora_config)

           # Print peft trainable params
           model.print_trainable_parameters()

           if training_args.gradient_checkpointing:
               model.enable_input_require_grads()

       # Load data
       data_module = make_supervised_data_module(
           tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
       )

       # Start trainer
       trainer = Trainer(
           model=model, tokenizer=tokenizer, args=training_args, **data_module
       )

       # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
       # loading the checkpoint when using LoRA with DeepSpeed.
       # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
       if (
           list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
           and not training_args.use_lora
       ):
           trainer.train(resume_from_checkpoint=True)
       else:
           trainer.train()
       trainer.save_state()

       safe_save_model_for_hf_trainer(
           trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
       )

The ``train`` method is the key to the training. In general, it loads
the tokenizer and model with ``AutoTokenizer.from_pretrained()`` and
``AutoModelForCausalLM.from_pretrained()``. If we use LoRA, the method
will initialize LoRA configuration with ``LoraConfig``. If we apply
Q-LoRA, we should use ``prepare_model_for_kbit_training``. Note that for
now it still does not support resume for LoRA. Then we leave the
following efforts to ``trainer`` and have a cup of coffee!

Next Step
---------

Now, you are able to use a very simple script to perform different types
of SFT. Alternatively, you can use more advanced training libraries,
such as
`Axolotl <https://github.com/OpenAccess-AI-Collective/axolotl>`__ or
`LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`__, to enjoy
more functionalities. To take a step forward, after SFT, you can
consider RLHF to align your model to human preferences! Stay tuned for
our next tutorial on RLHF!
