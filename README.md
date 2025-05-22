# Qwen3

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="400"/>
<p>

<p align="center">
          💜 <a href="https://chat.qwen.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2505.09388">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen3/">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-Demo">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen3-` or visit the [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f), and you will find all you need! Enjoy!

To learn more about Qwen3, feel free to read our documentation \[[EN](https://qwen.readthedocs.io/en/latest/)|[ZH](https://qwen.readthedocs.io/zh-cn/latest/)\]. Our documentation consists of the following sections:

- Quickstart: the basic usages and demonstrations;
- Inference: the guidance for the inference with Transformers, including batch inference, streaming, etc.;
- Run Locally: the instructions for running LLM locally on CPU and GPU, with frameworks like llama.cpp and Ollama;
- Deployment: the demonstration of how to deploy Qwen for large-scale inference with frameworks like SGLang, vLLM, TGI, etc.;
- Quantization: the practice of quantizing LLMs with GPTQ, AWQ, as well as the guidance for how to make high-quality quantized GGUF files;
- Training: the instructions for post-training, including SFT and RLHF (TODO) with frameworks like Axolotl, LLaMA-Factory, etc.
- Framework: the usage of Qwen with frameworks for application, e.g., RAG, Agent, etc.

## Introduction

We are excited to announce the release of Qwen3, the latest addition to the Qwen family of large language models.
These models represent our most advanced and intelligent systems to date, improving from our experience in building QwQ and Qwen2.5.
We are making the weights of Qwen3 available to the public, including both dense and Mixture-of-Expert (MoE) models. 

The highlights from Qwen3 include:
- **Dense and Mixture-of-Experts (MoE) models of various sizes**, available in 0.6B, 1.7B, 4B, 8B, 14B, 32B and 30B-A3B, 235B-A22B.
- **Seamless switching between thinking mode** (for complex logical reasoning, math, and coding) and **non-thinking mode** (for efficient, general-purpose chat), ensuring optimal performance across various scenarios.
- **Significantly enhancement in reasoning capabilities**, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- **Superior human preference alignment**, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- **Expertise in agent capabilities**, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- **Support of 100+ languages and dialects** with strong capabilities for **multilingual instruction following** and **translation**.

> [!IMPORTANT]
> Qwen3 models adopt a different naming scheme.
>
> The post-trained models do not use the "-Instruct" suffix anymore. For example, Qwen3-32B is the newer version of Qwen2.5-32B-Instruct.
>
> The base models now have names ending with "-Base".


## News

- 2025.04.29: We released the Qwen3 series. Check our [blog](https://qwenlm.github.io/blog/qwen3) for more details!
- 2024.09.19: We released the Qwen2.5 series. This time there are 3 extra model sizes: 3B, 14B, and 32B for more possibilities. Check our [blog](https://qwenlm.github.io/blog/qwen2.5) for more!
- 2024.06.06: We released the Qwen2 series. Check our [blog](https://qwenlm.github.io/blog/qwen2/)!
- 2024.03.28: We released the first MoE model of Qwen: Qwen1.5-MoE-A2.7B! Temporarily, only HF transformers and vLLM support the model. We will soon add the support of llama.cpp, mlx-lm, etc. Check our [blog](https://qwenlm.github.io/blog/qwen-moe/) for more information!
- 2024.02.05: We released the Qwen1.5 series.

## Performance

Detailed evaluation results are reported in this [📑 blog](https://qwenlm.github.io/blog/qwen3/).

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html).

## Run Qwen3

### 🤗 Transformers

Transformers is a library of pretrained natural language processing for inference and training. 
The latest version of `transformers` is recommended and `transformers>=4.51.0` is required.

The following contains a code snippet illustrating how to use the model generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

By default, Qwen3 models will think before response.
This could be controlled by
- `enable_thinking=False`: Passing `enable_thinking=False` to `tokenizer.apply_chat_template` will strictly prevent the model from generating thinking content.
- `/think` and `/no_think` instructions: Use those words in the system or user message to signify whether Qwen3 should think. In multi-turn conversations, the latest instruction is followed.



### ModelScope

We strongly advise users especially those in mainland China to use ModelScope. 
ModelScope adopts a Python API similar to Transformers.
The CLI tool `modelscope download` can help you solve issues concerning downloading checkpoints.


### llama.cpp

[`llama.cpp`](https://github.com/ggml-org/llama.cpp) enables LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware.
`llama.cpp>=b5092` is required for the support of Qwen3 architecture.
`llama.cpp>=b5401` is recommended for the full support of the official Qwen3 chat template.

To use the CLI, run the following in a terminal:
```shell
./llama-cli -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --color -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift
# CTRL+C to exit
```

To use the API server, run the following in a terminal:
```shell
./llama-server -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --reasoning-format deepseek -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift --port 8080
```
A simple web front end will be at `http://localhost:8080` and an OpenAI-compatible API will be at `http://localhost:8080/v1`.

For additional guides, please refer to [our documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html).

> [!TIP]
> llama.cpp adopts "rotating context management" and infinite generation is made possible by evicting earlier tokens.
> It could configured by parameters and the commands above effectively disable it.
> For more details, please refer to [our documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html#llama-cli).

### Ollama

After [installing Ollama](https://ollama.com/), you can initiate the Ollama service with the following command (Ollama v0.6.6 or higher is required):
```shell
ollama serve
# You need to keep this service running whenever you are using ollama
```

To pull a model checkpoint and run the model, use the `ollama run` command. You can specify a model size by adding a suffix to `qwen3`, such as `:8b` or `:30b-a3b`:
```shell
ollama run qwen3:8b
# Setting parameters, type "/set parameter num_ctx 40960" and "/set parameter num_predict 32768"
# To exit, type "/bye" and press ENTER
```

You can also access the Ollama service via its OpenAI-compatible API. 
Please note that you need to (1) keep `ollama serve` running while using the API, and (2) execute `ollama run qwen3:8b` before utilizing this API to ensure that the model checkpoint is prepared.
The API is at `http://localhost:11434/v1/` by default.

For additional details, please visit [ollama.ai](https://ollama.com/).

> [!TIP]
> Ollama adopts the same "rotating context management" with llama.cpp.
> However, its default settings (`num_ctx` 2048 and `num_predict` -1), suggesting infinite generation with a 2048-token context,
> could lead to trouble for Qwen3 models.
> We recommend setting `num_ctx` and `num_predict` properly.

### LMStudio

Qwen3 has already been supported by [lmstudio.ai](https://lmstudio.ai/). You can directly use LMStudio with our GGUF files.

### ExecuTorch

To export and run on ExecuTorch (iOS, Android, Mac, Linux, and more), please follow this [example](https://github.com/pytorch/executorch/blob/main/examples/models/qwen3/README.md).

### MNN

To export and run on MNN, which supports Qwen3 on mobile devices, please visit [Alibaba MNN](https://github.com/alibaba/MNN).

### MLX LM

If you are running on Apple Silicon, [`mlx-lm`](https://github.com/ml-explore/mlx-lm) also supports Qwen3 (`mlx-lm>=0.24.0`). 
Look for models ending with MLX on Hugging Face Hub.


### OpenVINO

If you are running on Intel CPU or GPU, [OpenVINO toolkit](https://github.com/openvinotoolkit) supports Qwen3.
You can follow this [chatbot example](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb).


<!-- ### Text generation web UI

You can directly use [`text-generation-webui`](https://github.com/oobabooga/text-generation-webui) for creating a web UI demo. If you use GGUF, remember to install the latest wheel of `llama.cpp` with the support of Qwen2.5. -->


<!-- ### llamafile

Clone [`llamafile`](https://github.com/Mozilla-Ocho/llamafile), run source install, and then create your own llamafile with the GGUF file following the guide [here](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#creating-llamafiles). You are able to run one line of command, say `./qwen.llamafile`, to create a demo. -->


## Deploy Qwen3

Qwen3 is supported by multiple inference frameworks. 
Here we demonstrate the usage of `SGLang`, `vLLM` and `TensorRT-LLM`.
You can also find Qwen3 models from various inference providers, e.g., [Alibaba Cloud Model Studio](https://www.alibabacloud.com/en/product/modelstudio).

### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.
SGLang could be used to launch a server with OpenAI-compatible API service. 
`sglang>=0.4.6.post1` is required.
It is as easy as
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --reasoning-parser qwen3
```
An OpenAI-compatible API will be available at `http://localhost:30000/v1`.

### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference and serving engine for LLMs.
`vllm>=0.8.5` is recommended.

```shell
vllm serve Qwen/Qwen3-8B --port 8000 --enable-reasoning --reasoning-parser deepseek_r1
```
An OpenAI-compatible API will be available at `http://localhost:8000/v1`.

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is an open-source LLM inference engine from NVIDIA, which provides optimizations including custom attention kernels, quantization and more on NVIDIA GPUs. Qwen3 is supported in its re-architected [PyTorch backend](https://nvidia.github.io/TensorRT-LLM/torch.html). `tensorrt_llm>=0.20.0rc3` is recommended. Please refer to the [README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md#qwen3) page for more details.

```shell
trtllm-serve Qwen/Qwen3-8B --host localhost --port 8000 --backend pytorch
```
An OpenAI-compatible API will be available at `http://localhost:8000/v1`.

### MindIE

For deployment on Ascend NPUs, please visit [Modelers](https://modelers.cn/) and search for Qwen3.

<!-- 
### OpenLLM

[OpenLLM](https://github.com/bentoml/OpenLLM) allows you to easily run Qwen2.5 as OpenAI-compatible APIs. You can start a model server using `openllm serve`. For example:

```bash
openllm serve qwen2.5:7b
```

The server is active at `http://localhost:3000/`, providing OpenAI-compatible APIs. You can create an OpenAI client to call its chat API. For more information, refer to [our documentation](https://qwen.readthedocs.io/en/latest/deployment/openllm.html). -->


## Build with Qwen3

### Tool Use

For tool use capabilities, we recommend taking a look at [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), which provides a wrapper around these APIs to support tool use or function calling with MCP support.
Tool use with Qwen3 can also be conducted with SGLang, vLLM, Transformers, llama.cpp, Ollama, etc.
Follow guides in our documentation to see how to enable the support.


### Finetuning

We advise you to use training frameworks, including [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [UnSloth](https://github.com/unslothai/unsloth), [Swift](https://github.com/modelscope/swift), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), etc., to finetune your models with SFT, DPO, GRPO, etc.


## License Agreement

All our open-weight models are licensed under Apache 2.0. 
You can find the license files in the respective Hugging Face repositories.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@article{qwen3,
    title={Qwen3 Technical Report}, 
    author={An Yang and Anfeng Li and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Gao and Chengen Huang and Chenxu Lv and Chujie Zheng and Dayiheng Liu and Fan Zhou and Fei Huang and Feng Hu and Hao Ge and Haoran Wei and Huan Lin and Jialong Tang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jing Zhou and Jingren Zhou and Junyang Lin and Kai Dang and Keqin Bao and Kexin Yang and Le Yu and Lianghao Deng and Mei Li and Mingfeng Xue and Mingze Li and Pei Zhang and Peng Wang and Qin Zhu and Rui Men and Ruize Gao and Shixuan Liu and Shuang Luo and Tianhao Li and Tianyi Tang and Wenbiao Yin and Xingzhang Ren and Xinyu Wang and Xinyu Zhang and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yinger Zhang and Yu Wan and Yuqiong Liu and Zekun Wang and Zeyu Cui and Zhenru Zhang and Zhipeng Zhou and Zihan Qiu},
    journal = {arXiv preprint arXiv:2505.09388},
    year={2025}
}

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
