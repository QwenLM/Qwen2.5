# Qwen1.5

<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/assets/blog/qwen1.5/logo_qwen1.5.jpg" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen1.5-`, and you will find all you need! Enjoy!



## Introduction
This time, we upgrade Qwen to Qwen1.5, the beta version of Qwen2. Similar to Qwen, it is still a decoder-only transformer model with SwiGLU activation, RoPE, multi-head attention. At this moment, we have achieved:
* 7 model sizes: 0.5B, 1.8B, 4B, 7B, 14B, and 72B models, plus a 14B (A2.7B) MoE model;
* Significant model quality improvements in chat models;
* Strengthened multilingual capabilities in both base and chat models;
* All models support the context length of `32768` tokens;
* System prompts enabled for all models, which means roleplay is possible.
* No need of `trust_remote_code` anymore.

We have not integrated GQA and mixture of SWA and full attention in this version and we will add the features in the future version.


## News
* 2024.03.28: We released the first MoE model of Qwen: Qwen1.5-MoE-A2.7B! Temporarily, only HF transformers and vLLM support the model. We will soon add the support of llama.cpp, mlx-lm, etc. Check our [blog](https://qwenlm.github.io/blog/qwen-moe/) for more information!
* 2024.02.05: We released the Qwen1.5 series.

## Performance
Detailed evaluation results are reported in this <a href="https://qwenlm.github.io"> 📑 blog</a>.


## Requirements
* `transformers>=4.37.0` for Qwen1.5 dense models.
* For Qwen1.5-MoE models, you should clone `transformers` and install from source.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0`.
> </b>
> </div>

## Quickstart

### 🤗 Hugging Face Transformers

Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-72B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

For quantized models, we advise you to use the GPTQ and AWQ correspondents, namely `Qwen1.5-7B-Chat-GPTQ-Int8`, `Qwen1.5-7B-Chat-AWQ`. 

### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

### 💻 Run locally

#### Ollama

> [!NOTE]
> <div align="center">
> Ollama provides an <a href="https://github.com/ollama/ollama/blob/main/docs/openai.md">OpenAI-compatible API</a>, which however does NOT support <b>function calling</b>. For tool use capabilities, consider using <a href="https://github.com/QwenLM/Qwen-Agent">Qwen-Agent</a>, which offers a wrapper for function calling over the API.
> </div>

After [installing ollama](https://github.com/ollama/ollama/blob/main/README.md), you can initiate the ollama service with the following command:
```shell
ollama serve
# You need to keep this service running whenever you are using ollama
```

To pull a model checkpoint and run the model, use the `ollama run` command. You can specify a model size by adding a suffix to `qwen`, such as `:0.5b`, `:1.8b`, `:4b`, `:7b`, `:14b`, or `:72b`:
```shell
ollama run qwen:4b
# To exit, type "/bye" and press ENTER
```

You can also access the ollama service via its OpenAI-compatible API. Please note that you need to (1) keep `ollama serve` running while using the API, and (2) execute `ollama run qwen:4b` before utilizing this API to ensure that the model checkpoint is prepared.
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
    model='qwen:4b',
)
```

For additional details, please visit [ollama.ai](https://ollama.ai/).

#### llama.cpp
Download our provided GGUF files or create them by yourself, and you can directly use them with the latest [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with a one-line command:
```shell
./main -m <path-to-file> -n 512 --color -i -cml -f prompts/chat-with-qwen.txt
```

#### LMStudio
Qwen1.5 has already been supported by [lmstudio.ai](https://lmstudio.ai/). You can directly use LMStudio with our GGUF files.

#### OpenVINO
Qwen1.5 has already been supported by [OpenVINO toolkit](https://github.com/openvinotoolkit). You can install and run this [example notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) with Intel CPU, integrated GPU or discrete GPU. 


## Web UI

#### Text generation web UI
You can directly use [`text-generation-webui`](https://github.com/oobabooga/text-generation-webui) for creating a web UI demo. If you use GGUF, remember to install the latest wheel of `llama.cpp` with the support of Qwen1.5.


#### llamafile
Clone [`llamafile`](https://github.com/Mozilla-Ocho/llamafile), run source install, and then create your own llamafile with the GGUF file following the guide [here](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#creating-llamafiles). You are able to run one line of command, say `./qwen.llamafile`, to create a demo.


## Deployment
Now, Qwen1.5 is supported by multiple inference frameworks. Here we demonstrate the usage of `vLLM` and `SGLang`.

> [!Warning]
> <div align="center">
> The OpenAI-compatible APIs provided by vLLM and SGLang currently do NOT support <b>function calling</b>. For tool use capabilities, <a href="https://github.com/QwenLM/Qwen-Agent">Qwen-Agent</a> provides a wrapper around these APIs to support function calling.
> </div>

### vLLM
We advise you to use `vLLM>=0.3.0` to build OpenAI-compatible API service. Start the server with a chat model, e.g. `Qwen1.5-7B-Chat`:
```shell
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-7B-Chat --model Qwen/Qwen1.5-7B-Chat 
```

Then use the chat API as demonstrated below:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen1.5-7B-Chat",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
    ]
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
    model="Qwen1.5-7B-Chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ]
)
print("Chat response:", chat_response)
```

### SGLang
Please install `SGLang` from source. Similar to `vLLM`, you need to launch a server and use OpenAI-compatible API service. Start the server first:
```shell
python -m sglang.launch_server --model-path Qwen/Qwen1.5-7B-Chat --port 30000
```
You can use it in Python as shown below:
```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
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

## Finetuning
We advise you to use training frameworks, including [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), [Swift](https://github.com/modelscope/swift), etc., to finetune your models with SFT, DPO, PPO, etc.

## API
Qwen1.5 models are now deployed on both [DashScope](https://dashscope.aliyun.com/) and [Together](https://api.together.ai/). Check [this](https://api.together.xyz/playground/chat/Qwen/Qwen1.5-72B-Chat) out and have fun with Qwen1.5-72B-Chat!

## 🐳 Docker

To simplify the deployment process, we provide docker images with pre-built environments: [qwenllm/qwen](https://hub.docker.com/r/qwenllm/qwen). You only need to install the driver and download model files to launch demos and finetune the model.

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen1.5 -it qwenllm/qwen:1.5-cu121 bash
```

## License Agreement
Check the license of each model inside its HF repo. It is NOT necessary for you to submit a request for commercial usage.

## Citation
If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](assets/wechat.png)!

