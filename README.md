# Qwen2.5

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" width="400"/>
<p>

<p align="center">
          💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2412.15115">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen2.5/">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>

<br>
💻 <a href="https://gallery.pai-ml.com/#/preview/deepLearning/nlp/qwen2-5_7b">PAI-DSW</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-72B-Instruct">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen2.5-` or visit the [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), and you will find all you need! Enjoy!

To learn more about Qwen2.5, feel free to read our documentation \[[EN](https://qwen.readthedocs.io/en/latest/)|[ZH](https://qwen.readthedocs.io/zh-cn/latest/)\]. Our documentation consists of the following sections:

- Quickstart: the basic usages and demonstrations;
- Inference: the guidance for the inference with transformers, including batch inference, streaming, etc.;
- Run Locally: the instructions for running LLM locally on CPU and GPU, with frameworks like `llama.cpp` and `Ollama`;
- Deployment: the demonstration of how to deploy Qwen for large-scale inference with frameworks like `vLLM`, `TGI`, etc.;
- Quantization: the practice of quantizing LLMs with GPTQ, AWQ, as well as the guidance for how to make high-quality quantized GGUF files;
- Training: the instructions for post-training, including SFT and RLHF (TODO) with frameworks like Axolotl, LLaMA-Factory, etc.
- Framework: the usage of Qwen with frameworks for application, e.g., RAG, Agent, etc.
- Benchmark: the statistics about inference speed and memory footprint (Available for Qwen2.5).

## Introduction

In the past three months since Qwen2's release, numerous developers have built new models on the Qwen2 language models, providing us with valuable feedback. During this period, we have focused on creating smarter and more knowledgeable language models. Today, we are excited to introduce the latest addition to the Qwen family: **Qwen2.5**. 

- Dense, easy-to-use, decoder-only language models, available in **0.5B**, **1.5B**, **3B**, **7B**, **14B**, **32B**, and **72B** sizes, and base and instruct variants.
- Pretrained on our latest large-scale dataset, encompassing up to **18T** tokens.
- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. 
- More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots. 
- Context length support up to **128K** tokens and can generate up to **8K** tokens. 
- Multilingual support for over **29** languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more. 

## News

- 2024.09.19: We released the Qwen2.5 series. This time there are 3 extra model sizes: 3B, 14B, and 32B for more possibilities. Check our [blog](https://qwenlm.github.io/blog/qwen2.5) for more!
- 2024.06.06: We released the Qwen2 series. Check our [blog](https://qwenlm.github.io/blog/qwen2/)!
- 2024.03.28: We released the first MoE model of Qwen: Qwen1.5-MoE-A2.7B! Temporarily, only HF transformers and vLLM support the model. We will soon add the support of llama.cpp, mlx-lm, etc. Check our [blog](https://qwenlm.github.io/blog/qwen-moe/) for more information!
- 2024.02.05: We released the Qwen1.5 series.

## Performance

Detailed evaluation results are reported in this <a href="https://qwenlm.github.io/blog/qwen2.5/"> 📑 blog</a>.

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) .

## Quickstart

### 🤗 Hugging Face Transformers


The latest version of `transformers` is recommended (at least 4.37.0).
Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

For quantized models, we advise you to use the GPTQ and AWQ correspondents, namely `Qwen2.5-7B-Instruct-GPTQ-Int8` and `Qwen2.5-7B-Instruct-AWQ`. 

### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

### 💻 Run locally

#### Ollama

After [installing ollama](https://github.com/ollama/ollama/blob/main/README.md), you can initiate the ollama service with the following command:
```shell
ollama serve
# You need to keep this service running whenever you are using ollama
```

To pull a model checkpoint and run the model, use the `ollama run` command. You can specify a model size by adding a suffix to `qwen2.5`, such as `:0.5b`, `:1.5b`, `:7b`, or `:72b`:
```shell
ollama run qwen2.5:7b
# To exit, type "/bye" and press ENTER
```

You can also access the ollama service via its OpenAI-compatible API. Please note that you need to (1) keep `ollama serve` running while using the API, and (2) execute `ollama run qwen2.5:7b` before utilizing this API to ensure that the model checkpoint is prepared.
```py
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # required but ignored
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='qwen2.5:7b',
)
```

For additional details, please visit [ollama.ai](https://ollama.ai/).

#### llama.cpp

Download our provided GGUF files or create them by yourself, and you can directly use them with the latest [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with a one-line command:
```shell
./llama-cli -m <path-to-file> -n 512 -co -sp -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
```

For additional guides, please refer to [our documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html).

#### MLX-LM

If you are running on Apple Silicon, we have also provided checkpoints compatible with [`mlx-lm`](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md). Look for models ending with MLX on HuggingFace Hub, like [Qwen2.5-7B-Instruct-MLX](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-MLX).

#### LMStudio

Qwen2.5 has already been supported by [lmstudio.ai](https://lmstudio.ai/). You can directly use LMStudio with our GGUF files.

#### OpenVINO

Qwen2.5 has already been supported by [OpenVINO toolkit](https://github.com/openvinotoolkit). You can install and run this [chatbot example](https://github.com/OpenVINO-dev-contest/Qwen2.openvino) with Intel CPU, integrated GPU or discrete GPU. 


## Web UI

#### Text generation web UI

You can directly use [`text-generation-webui`](https://github.com/oobabooga/text-generation-webui) for creating a web UI demo. If you use GGUF, remember to install the latest wheel of `llama.cpp` with the support of Qwen2.5.


#### llamafile

Clone [`llamafile`](https://github.com/Mozilla-Ocho/llamafile), run source install, and then create your own llamafile with the GGUF file following the guide [here](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#creating-llamafiles). You are able to run one line of command, say `./qwen.llamafile`, to create a demo.


## Deployment

Qwen2.5 is supported by multiple inference frameworks. Here we demonstrate the usage of `vLLM`, `SGLang` and `OpenLLM`.

### vLLM

We advise you to use the latest version of vLLM to build OpenAI-compatible API service, including tool use support. Start the server with a chat model, e.g. `Qwen2.5-7B-Instruct`:
```shell
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Then use the chat API as demonstrated below:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "max_tokens": 512
}'
```

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```

### SGLang

> [!Warning]
> The OpenAI-compatible APIs provided by SGLang currently do NOT support **tool use** or **function calling**. 

Please install `SGLang` from source. Similar to `vLLM`, you need to launch a server and use OpenAI-compatible API service. Start the server first:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
```
You can use it in Python as shown below:
```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of China?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

print(state["answer_1"])
```

### OpenLLM

[OpenLLM](https://github.com/bentoml/OpenLLM) allows you to easily run Qwen2.5 as OpenAI-compatible APIs. You can start a model server using `openllm serve`. For example:

```bash
openllm serve qwen2.5:7b
```

The server is active at `http://localhost:3000/`, providing OpenAI-compatible APIs. You can create an OpenAI client to call its chat API. For more information, refer to [our documentation](https://qwen.readthedocs.io/en/latest/deployment/openllm.html).

### Tool Use

For tool use capabilities, we recommend taking a look at [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), which provides a wrapper around these APIs to support tool use or function calling.
Tool use with Qwen2.5 can also be conducted with Hugging Face `transformers`, Ollama, and vLLM.
Follow guides in our documentation to see how to enable the support.


## Finetuning

We advise you to use training frameworks, including [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), [unsloth](https://github.com/unslothai/unsloth), [Swift](https://github.com/modelscope/swift), etc., to finetune your models with SFT, DPO, PPO, etc.


## License Agreement

All our open-source models, except for the 3B and 72B variants, are licensed under Apache 2.0. 
You can find the license files in the respective Hugging Face repositories.

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2.5,
    title   = {Qwen2.5 Technical Report}, 
    author  = {An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
    journal = {arXiv preprint arXiv:2412.15115},
    year    = {2024}
}

@article{qwen2,
    title   = {Qwen2 Technical Report}, 
    author  = {An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
    journal = {arXiv preprint arXiv:2407.10671},
    year    = {2024}
}
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](assets/wechat.png)!
