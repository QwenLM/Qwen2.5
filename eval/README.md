This folder provides scripts to reproduce evaluation results across various benchmarks for the **Qwen** series of large language models.

## Supported Benchmarks

Currently, we support the following benchmark:

| Model | Dataset | Config | Reproduced Score |
|-------|--------|--------|------------------|
| Qwen3-235B-A22B-Instruct-2507 | ARC-AGI 1 (pass@1) | [./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml](./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml) | 40.75 |

In the meantime, you can find the model outputs and final evaluation results in the [`./output`](./output) and [`./eval_res`](./eval_res) directories, respectively.

Additional benchmarks will be added in future updates. 


## Evaluation Guide

Follow the steps below to reproduce the reported scores.

### Step 0: Prerequisites

Ensure you have:
- Python ≥ 3.9
- Either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sgl) installed

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Start vLLM Server

Launch the vLLM inference server using the command below:

```bash
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"  # Replace with desired model
export MODEL_PATH="$MODEL_NAME"  # Or path to local checkpoint
export NUM_GPUS=8

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $NUM_GPUS \
    --enforce-eager \
    --port 8030
```

> 💡 Adjust `tensor_parallel_size` according to your GPU setup.

### Optional: Start SGLang Router (Recommended for Faster Evaluation)

Since evaluations can take several days, we recommend using **SGLang** with data parallelism to accelerate inference. See the [SGLang Router documentation](https://docs.sglang.ai/router/router.html) for details.

Start the SGLang router server:

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

> ⚠️ Adjust `dp_size` based on available resources, and ensure consistency in port configuration for subsequent steps.


### Step 2: Run Inference

Once the inference server is running, generate model responses using the multithreaded inference script.

```bash
mkdir -p output

# Example: Evaluate on ARC-AGI
python generate_api_answers/infer_multithread.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
```

#### Resume Interrupted Inference

If the process is interrupted, simply re-run the same command. The script will automatically detect existing outputs and resume generation for incomplete prompts.

### Step 3: Compute Scores

After inference completes, evaluate the results using the scoring script:

```bash
mkdir -p eval_res

python eval/eval.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml \
    > eval_res/ARCAGI-Qwen3-235B-A22B-Instruct-2507_eval_result.txt
```

The final score will be saved to the specified output file.
