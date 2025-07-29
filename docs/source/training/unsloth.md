# Unsloth

This guide will teach you how to easily train Qwen3 models with Unsloth. Unsloth simplifies local model training, handling everything from loading and quantization to training, evaluation, running, and deployment with inference engines (Ollama, llama.cpp, vLLM). **Train Qwen** models 2× faster using 70% less VRAM.

**GitHub repo:** [Unsloth](https://github.com/unslothai/unsloth)

## ⭐ Key Features
- Supports full fine-tuning, pretraining, LoRA, QLoRA, 8-bit training & more  
- Single and multi-GPU support (Linux, Windows, Colab, Kaggle; NVIDIA GPUs, soon AMD & Intel)  
- Compatible with all transformer models: TTS, multimodal, STT, BERT, RL  
- RLHF support: GRPO, DPO, DAPO, RM, PPO, KTO, etc.  
- Hand-written Triton kernels and a manual backprop engine ensure no accuracy degradation (0% approximation).

## Quickstart
**Local Installation (Linux recommended):**

```bash
pip install unsloth
```

You can view Unsloth’s full [installation instructions here.](https://docs.unsloth.ai/get-started/installing-+-updating)

## Fine-tuning Qwen3 with Unsloth
Unsloth makes Qwen3 fine-tuning 2× faster, uses 70% less VRAM, with 8× longer contexts. Qwen3 (14B) fits in a free 16 GB Colab Tesla T4 GPU.

To retain Qwen3's reasoning capabilities, use a 75% reasoning to 25% non-reasoning dataset ratio (e.g., NVIDIA’s math‑reasoning dataset + Maxime’s FineTome).

For more details, see Unsloth’s full [Qwen3 fine-tuning guide](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune#fine-tuning-qwen3-with-unsloth).

### Colab Notebooks
- [Qwen3 (14B) Reasoning + Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb)  
- [Qwen3 (4B) Advanced GRPO LoRA](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)  
- [Qwen3 (14B) Alpaca (Base model)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Alpaca.ipynb)

**Update Unsloth locally:**

```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### Fine-tuning Qwen3 MoE Models
Supported MoE models include 30B‑A3B and 235B‑A22B. Unsloth fine-tunes the 30B‑A3B model with just 17.5 GB VRAM. Router-layer fine-tuning is disabled by default.

Use `FastModel` for MoE fine-tuning:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-30B-A3B",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)
```

### Notebook Guide
For an end-to-end walkthrough, see Unsloth’s [full end-to-end fine-tuning guide](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide).

- Open the notebook → click **Runtime ▸ Run all**.  
- Adjust settings (e.g., model name, context length) directly in the notebook:  
  - `max_seq_length`: Recommended 2048 (Qwen3 supports up to 40960).  
  - `load_in_4bit=True`: reduces memory usage by 4×.  
  - Enable full fine-tuning (`full_finetuning=True`) or 8-bit training (`load_in_8bit=True`).

If you want to use models directly from [ModelScope](https://modelscope.cn/organization/unsloth), use:

```bash
pip install modelscope -qqq
```

```python
import os
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=2048,
)
```

## RL & GRPO with Qwen3
You can also train Qwen models with reinforcement learning (RL) using Unsloth. Explore Unsloth’s advanced GRPO notebook, featuring proximity-based reward scoring and Hugging Face's Open‑R1 math dataset: [Qwen3 (4B) Advanced GRPO LoRA notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb).  
- Proximity-based rewards for closer answers  
- Custom GRPO formatting and templates  
- Enhanced evaluation accuracy with regex matching

## Resources & Links
That’s how you can easily train Qwen models with Unsloth. If you need any help, join the discussion on Unsloth’s [Discord](https://discord.com/invite/unsloth) or [GitHub](https://github.com/unslothai/unsloth) pages.

**Links:**  
- [Unsloth Documentation](https://docs.unsloth.ai/)  
- [Unsloth Discord](https://discord.com/invite/unsloth)  
- [Unsloth Website](https://unsloth.ai/)  
- [Unsloth Reddit](https://www.reddit.com/r/unsloth/)  
