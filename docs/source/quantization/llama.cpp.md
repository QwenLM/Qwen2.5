# llama.cpp

Quantization is a major topic for local inference of LLMs, as it reduces the memory footprint.
Undoubtably, llama.cpp natively supports LLM quantization and of course, with flexibility as always.

At high-level, all quantization supported by llama.cpp is weight quantization: 
Model parameters are quantized into lower bits, and in inference, they are dequantized and used in computation.

In addition, you can mix different quantization data types in a single quantized model, e.g., you can quantize the embedding weights using a quantization data type and other weights using a different one.
With an adequate mixture of quantization types, much lower quantization error can be attained with just a slight increase of bit-per-weight.
The example program `llama-quantize` supports many quantization presets, such as Q4_K_M and Q8_0.

If you find the quantization errors still more than expected, you can bring your own scales, e.g., as computed by AWQ, or use calibration data to compute an importance matrix using `llama-imatrix`, which can then be used during quantization to enhance the quality of the quantized models.

In this document, we demonstrate the common way to quantize your model and evaluate the performance of the quantized model.
We will assume you have the example programs from llama.cpp at your hand.
If you don't, check our guide [here](../run_locally/llama.cpp.html#getting-the-program){.external}.

## Getting the GGUF

Now, suppose you would like to quantize `Qwen3-8B`. 
You need to first make a GGUF file as shown below:
```bash
python convert-hf-to-gguf.py Qwen/Qwen3-8B --outfile Qwen3-8B-F16.gguf
```

Since Qwen3 are trained using the bfloat16 precision, the following should keep most information on supported machines:
```bash
python convert-hf-to-gguf.py Qwen/Qwen3-8B --outtype bf16 --outfile Qwen3-8B-BF16.gguf
```

Sometimes, it may be better to use fp32 as the start point for quantization.
In that case, use
```bash
python convert-hf-to-gguf.py Qwen/Qwen3-8B --outtype f32 --outfile Qwen3-8B-F32.gguf
```

## Quantizing the GGUF without Calibration

For the simplest way, you can directly quantize the model to lower-bits based on your requirements. 
An example of quantizing the model to 8 bits is shown below:
```bash
./llama-quantize Qwen3-8B-F16.gguf Qwen3-8B-Q8_0.gguf Q8_0
```

`Q8_0` is a code for a quantization preset.
You can find all the presets in [the source code of `llama-quantize`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp).
Look for the variable `QUANT_OPTIONS`.
Common ones used for 8B models include `Q8_0`, `Q5_K_M`, and `Q4_K_M`. 
The letter case doesn't matter, so `q8_0` or `q4_K_m` are perfectly fine.

Now you can use the GGUF file of the quantized model with applications based on llama.cpp.
Very simple indeed.

However, the accuracy of the quantized model could be lower than expected occasionally, especially for lower-bit quantization.
The program may even prevent you from doing that. 

There are several ways to improve quality of quantized models.
A common way is to use a calibration dataset in the target domain to identify the weights that really matter and quantize the model in a way that those weights have lower quantization errors, as introduced in the next two methods.


## Quantizing the GGUF with AWQ Scale

:::{attention}
To be updated for Qwen3.
:::

To improve the quality of your quantized models, one possible solution is to apply the AWQ scale, following [this script](https://github.com/casper-hansen/AutoAWQ/blob/main/docs/examples.md#gguf-export).
First, when you run `model.quantize()` with `autoawq`, remember to add `export_compatible=True` as shown below:
```python
...
model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True
)
model.save_pretrained(quant_path)
...
```

The above code will not actually quantize the weights.
Instead, it adjusts weights based on a dataset so that they are "easier" to quantize.[^AWQ]

Then, when you run `convert-hf-to-gguf.py`, remember to replace the model path with the path to the new model:
```bash
python convert-hf-to-gguf.py <quant_path> --outfile qwen2.5-7b-instruct-f16-awq.gguf
```

Finally, you can quantize the model as in the last example:
```bash
./llama-quantize qwen2.5-7b-instruct-f16-awq.gguf qwen2.5-7b-instruct-q8_0.gguf Q8_0
```

In this way, it should be possible to achieve similar quality with lower bit-per-weight.

[^AWQ]: If you are interested in what this means, refer to [the AWQ paper](https://arxiv.org/abs/2306.00978).
        Basically, important weights (called salient weights in the paper) are identified based on activations across data examples.
        The weights are scaled accordingly such that the salient weights are protected even after quantization.

## Quantizing the GGUF with Importance Matrix

Another possible solution is to use the "important matrix"[^imatrix], following [this](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix).

First, you need to compute the importance matrix data of the weights of a model (`-m`) using a calibration dataset (`-f`):
```bash
./llama-imatrix -m Qwen3-8B-F16.gguf -f calibration-text.txt --chunk 512 -o Qwen3-8B-imatrix.dat -ngl 80
```

The text is cut in chunks of length `--chunk` for computation.
Preferably, the text should be representative of the target domain.
The final results will be saved in a file named `Qwen3-8B-imatrix.dat` (`-o`), which can then be used:
```bash
./llama-quantize --imatrix Qwen3-8B-imatrix.dat \
    Qwen3-8B-F16.gguf Qwen3-8B-Q4_K_M.gguf Q4_K_M
```

For lower-bit quantization mixtures for 1-bit or 2-bit, if you do not provide `--imatrix`, a helpful warning will be printed by `llama-quantize`.

[^imatrix]: Here, the importance matrix keeps record of how weights affect the output: the weight should be important is a slight change in its value causes huge difference in the results, akin to the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm.

## Perplexity Evaluation

`llama.cpp` provides an example program for us to calculate the perplexity, which evaluate how unlikely the given text is to the model.
It should be mostly used for comparisons: the lower the perplexity, the better the model remembers the given text.

To do this, you need to prepare a dataset, say "wiki test"[^wiki]. 
You can download the dataset with:
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research -O wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
```

Then you can run the test with the following command:
```bash
./llama-perplexity -m Qwen3-8B-Q8_0.gguf -f wiki.test.raw -ngl 80
```
Wait for some time and you will get the perplexity of the model.
There are some numbers of different kinds of quantization mixture [here](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md).
It might be helpful to look at the difference and grab a sense of how that kind of quantization might perform.

[^wiki]: It is not a good evaluation dataset for instruct models though, but it is very common and easily accessible.
         You probably want to use a dataset similar to your target domain.

## Finally

In this guide, we demonstrate how to conduct quantization and evaluate the perplexity with llama.cpp.
For more information, please visit the [llama.cpp GitHub repo](https://github.com/ggml-org/llama.cpp).

We usually quantize the fp16 model to 4, 5, 6, and 8-bit models with different quantization mixtures, but sometimes a particular mixture just does not work, so we don't provide those in our Hugging Face Hub.
However, others in the community may have success, so if you haven't found what you need in our repos, look around.

Enjoy your freshly quantized models!
