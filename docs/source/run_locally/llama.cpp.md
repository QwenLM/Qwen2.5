# llama.cpp

[^GGUF]: GPT-Generated Unified Format

:::{dropdown} llama.cpp as a C++ library
Before starting, let's first discuss what is llama.cpp and what you should expect, and why we say "use" llama.cpp, with "use" in quotes.
llama.cpp is essentially a different ecosystem with a different design philosophy that targets light-weight footprint, minimal external dependency, multi-platform, and extensive, flexible hardware support:
- Plain C/C++ implementation without external dependencies
- Support a wide variety of hardware:
  - AVX, AVX2 and AVX512 support for x86_64 CPU
  - Apple Silicon via Metal and Accelerate (CPU and GPU)
  - NVIDIA GPU (via CUDA), AMD GPU (via hipBLAS), Intel GPU (via SYCL), Ascend NPU (via CANN), and Moore Threads GPU (via MUSA)
  - Vulkan backend for GPU
- Various quantization schemes for faster inference and reduced memory footprint
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

It's like the Python frameworks `torch`+`transformers` or `torch`+`vllm` but in C++.
However, this difference is crucial: 
- Python is an interpreted language: 
  The code you write is executed line-by-line on-the-fly by an interpreter. 
  You can run the example code snippet or script with an interpreter or a natively interactive interpreter shell.
  In addition, Python is learner friendly, and even if you don't know much before, you can tweak the source code here and there.
- C++ is a compiled language: 
  The source code you write needs to be compiled beforehand, and it is translated to machine code and an executable program by a compiler.
  The overhead from the language side is minimal. 
  You do have source code for example programs showcasing how to use the library. 
  But it is not very easy to modify the source code if you are not verse in C++ or C.

To use llama.cpp means that you use the llama.cpp library in your own program, like writing the source code of [Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), [GPT4ALL](https://www.nomic.ai/gpt4all), [llamafile](https://llamafile.ai/) etc.
But that's not what this guide is intended or could do.
Instead, here we introduce how to use the `llama-cli` example program, in the hope that you know that llama.cpp does support Qwen2.5 models and how the ecosystem of llama.cpp generally works.
:::

In this guide, we will show how to "use" [llama.cpp](https://github.com/ggml-org/llama.cpp) to run models on your local machine, in particular, the `llama-cli` and the `llama-server` example program, which comes with the library.

The main steps are:
1. Get the programs
2. Get the Qwen3 models in GGUF[^GGUF] format
3. Run the program with the model

:::{note}
llama.cpp supports Qwen3 and Qwen3MoE from version `b5092`.
:::

## Getting the Program

You can get the programs in various ways. 
For optimal efficiency, we recommend compiling the programs locally, so you get the CPU optimizations for free.
However, if you don't have C++ compilers locally, you can also install using package managers or downloading pre-built binaries. 
They could be less efficient but for non-production example use, they are fine.

:::::{tab-set}
::::{tab-item} Compile Locally

Here, we show the basic command to compile `llama-cli` locally on **macOS** or **Linux**.
For Windows or GPU users, please refer to [the guide from llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

:::{rubric} Installing Build Tools
:heading-level: 5
:::

To build locally, a C++ compiler and a build system tool are required. 
To see if they have been installed already, type `cc --version` or `cmake --version` in a terminal window.
- If installed, the build configuration of the tool will be printed to the terminal, and you are good to go!
- If errors are raised, you need to first install the related tools:
  - On macOS, install with the command `xcode-select --install`
  - On Ubuntu, install with the command `sudo apt install build-essential`. 
    For other Linux distributions, the command may vary; the essential packages needed for this guide are `gcc` and `cmake`.

:::{rubric} Compiling the Program
:heading-level: 5
:::

For the first step, clone the repo and enter the directory:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Then, build llama.cpp using CMake:
```bash
cmake -B build
cmake --build build --config Release
```

The first command will check the local environment and determine which backends and features should be included.
The second command will actually build the programs.

To shorten the time, you can also enable parallel compiling based on the CPU cores you have, for example:
```bash
cmake --build build --config Release -j 8
```
This will build the programs with 8 parallel compiling jobs.

The built programs will be in `./build/bin/`.

::::

::::{tab-item} Package Managers
For **macOS** and **Linux** users, `llama-cli` and `llama-server` can be installed with package managers including Homebrew, Nix, and Flox.

Here, we show how to install `llama-cli` and `llama-server` with Homebrew. 
For other package managers, please check the instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

Installing with Homebrew is very simple:

1. Ensure that Homebrew is available on your operating system. 
   If you don't have Homebrew, you can install it as in [its website](https://brew.sh/).

2. Second, you can install the pre-built binaries, `llama-cli` and `llama-server` included, with a single command:
   ```bash
   brew install llama.cpp
   ```

Note that the installed binaries might not be built with the optimal compile options for your hardware, which can lead to poor performance.
They also don't support GPU on Linux systems.
::::

::::{tab-item} Binary Release

You can also download pre-built binaries from [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases).
Please note that those pre-built binaries files are architecture-, backend-, and os-specific. 
If you are not sure what those mean, you probably don't want to use them and running with incompatible versions will most likely fail or lead to poor performance.

The file name is like `llama-<version>-bin-<os>-<feature>-<arch>.zip`.

There are three simple parts:
- `<version>`: the version of llama.cpp. The latest is preferred, but as llama.cpp is updated and released frequently, the latest may contain bugs. If the latest version does not work, try the previous release until it works.
- `<os>`: the operating system. `win` for Windows; `macos` for macOS; `linux` for Linux.
- `<arch>`: the system architecture. `x64` for `x86_64`, e.g., most Intel and AMD systems, including Intel Mac; `arm64` for `arm64`, e.g., Apple Silicon or Snapdragon-based systems.

The `<feature>` part is somewhat complicated for Windows:
- Running on CPU
  - x86_64 CPUs: We suggest try the `avx2` one first.
    - `noavx`: No hardware acceleration at all.
    - `avx2`, `avx`, `avx512`: SIMD-based acceleration. Most modern desktop CPUs should support avx2, and some CPUs support `avx512`.
    - `openblas`: Relying on OpenBLAS for acceleration for prompt processing but not generation.
  - arm64 CPUs: We suggest try the `llvm` one first.
    - [`llvm` and `msvc`](https://github.com/ggml-org/llama.cpp/pull/7191) are different compilers
- Running on GPU: We suggest try the `cu<cuda_verison>` one for NVIDIA GPUs, `kompute` for AMD GPUs, and `sycl` for Intel GPUs first. Ensure that you have related drivers installed.
  - [`vulcan`](https://github.com/ggml-org/llama.cpp/pull/2059): support certain NVIDIA and AMD GPUs
  - [`kompute`](https://github.com/ggml-org/llama.cpp/pull/4456): support certain NVIDIA and AMD GPUs
  - [`sycl`](https://github.com/ggml-org/llama.cpp/discussions/5138): Intel GPUs, oneAPI runtime is included
  - `cu<cuda_verison>`: NVIDIA GPUs, CUDA runtime is not included. You can download the `cudart-llama-bin-win-cu<cuda_version>-x64.zip` and unzip it to the same directory if you don't have the corresponding CUDA toolkit installed.

You don't have much choice for macOS or Linux.
- Linux: only one prebuilt binary, `llama-<version>-bin-linux-x64.zip`, supporting CPU.
- macOS: `llama-<version>-bin-macos-x64.zip` for Intel Mac with no GPU support; `llama-<version>-bin-macos-arm64.zip` for Apple Silicon with GPU support.

After downloading the `.zip` file, unzip them into a directory and open a terminal at that directory.

::::
:::::


## Getting the GGUF

GGUF[^GGUF] is a file format for storing information needed to run a model, including but not limited to model weights, model hyperparameters, default generation configuration, and tokenizer.

You can use the official Qwen GGUFs from our Hugging Face Hub or prepare your own GGUF file.

### Using the Official Qwen3 GGUFs

We provide a series of GGUF models in our Hugging Face organization, and to search for what you need you can search the repo names with `-GGUF`. 

Download the GGUF model that you want with `huggingface-cli` (you need to install it first with `pip install huggingface_hub`):
```bash
huggingface-cli download <model_repo> <gguf_file> --local-dir <local_dir>
```

For example:
```bash
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir .
```

This will download the Qwen3-8B model in GGUF format quantized with the scheme Q4_K_M.

### Preparing Your Own GGUF

Model files from Hugging Face Hub can be converted to GGUF, using the `convert-hf-to-gguf.py` Python script.
It does require you to have a working Python environment with at least `transformers` installed.

Obtain the source file if you haven't already:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Suppose you would like to use Qwen3-8B you can make a GGUF file for the fp16 model as shown below:
```bash
python convert-hf-to-gguf.py Qwen/Qwen3-8B --outfile qwen3-8b-f16.gguf
```
The first argument to the script refers to the path to the HF model directory or the HF model name, and the second argument refers to the path of your output GGUF file.
Remember to create the output directory before you run the command.

The fp16 model could be a bit heavy for running locally, and you can quantize the model as needed.
We introduce the method of creating and quantizing GGUF files in [this guide](../quantization/llama.cpp). 
You can refer to that document for more information.

## Run Qwen with llama.cpp

:::{note}
Regarding switching between thinking and non-thinking modes,
while the soft switch is always available, the hard switch implemented in the chat template is not exposed in llama.cpp.
The quick workaround is to pass [a custom chat template](../../source/assets/qwen3_nonthinking.jinja) equivalent to always `enable_thinking=False` via `--chat-template-file`.
:::


### llama-cli

[llama-cli](https://github.com/ggml-org/llama.cpp/tree/master/tools/main) is a console program which can be used to chat with LLMs.
Simple run the following command where you place the llama.cpp programs:
```shell
./llama-cli -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --color -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift
```

Here are some explanations to the above command:
-   **Model**: llama-cli supports using model files from local path, remote URL, or Hugging Face hub.
    -   `-hf Qwen/Qwen3-8B-GGUF:Q8_0` in the above indicates we are using the model file from Hugging Face hub
    -   To use a local path, pass `-m qwen3-8b-q8_0.gguf` instead
    -   To use a remote URL, pass `-mu https://hf.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-Q8_0.gguf?download=true` instead

-   **Speed Optimization**: 
    - CPU: llama-cli by default will use CPU and you can change `-t` to specify how many threads you would like it to use, e.g., `-t 8` means using 8 threads.
    - GPU: If the programs are built with GPU support, you can use `-ngl`, which allows offloading some layers to the GPU for computation.
    If there are multiple GPUs, it will offload to all the GPUs.
    You can use `-dev` to control the devices used and `-sm` to control which kinds of parallelism is used.
    For example, `-ngl 99 -dev cuda0,cuda1 -sm row` means offload all layers to GPU 0 and GPU1 using the split mode row. 
    Adding `-fa` may also speed up the generation.

-   **Sampling Parameters**: llama.cpp supports [a variety of sampling methods](https://github.com/ggml-org/llama.cpp/tree/master/tools/main#generation-flags) and has default configuration for many of them.
    It is recommended to adjust those parameters according to the actual case and the recommended parameters from Qwen3 modelcard could be used as a reference.
    If you encounter repetition and endless generation, it is recommended to pass in addition `--presence-penalty` up to `2.0`.

-   **Context Management**: llama.cpp adopts the "rotating" context management by default.
    The `-c` controls the maximum context length (default 4096, 0 means loaded from model), and `-n` controls the maximum generation length each time (default -1 means infinite until ending, -2 means until context full).
    When the context is full but the generation doesn't end, the first `--keep` tokens (default 0, -1 means all) from the initial prompt is kept, and the first half of the rest is discarded.
    Then, the model continues to generate based on the new context tokens.
    You can set `--no-context-shift` to prevent this rotating behavior and the generation will stop once `-c` is reached.
    
    llama.cpp supports YaRN, which can be enabled by `-c 131072 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`.
-   **Chat**: `--jinja` indicates using the chat template embedded in the GGUF which is preferred and `--color` indicates coloring the texts so that user input and model output can be better differentiated.
    If there is a chat template, like in Qwen3 models, llama-cli will enter chat mode automatically.
    To stop generation or exit press "Ctrl+C".
    You can use `-sys` to add a system prompt.


### llama-server

[llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) is a simple HTTP server, including a set of LLM REST APIs and a simple web front end to interact with LLMs using llama.cpp.

The core command is similar to that of llama-cli.
In addition, it supports thinking content parsing and tool call parsing.

```shell
./llama-server -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --reasoning-format deepseek -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift
```

By default, the server will listen at `http://localhost:8080` which can be changed by passing `--host` and `--port`.
The web front end can be assessed from a browser at `http://localhost:8080/`.
The OpenAI compatible API is at `http://localhost:8080/v1/`.


## What's More

If you still find it difficult to use llama.cpp, don't worry, just check out other llama.cpp-based applications.
For example, Qwen3 has already been officially part of Ollama and LM Studio, which are platforms for your to search and run local LLMs. 

Have fun!
