# Speed Benchmark

We report the speed performance of bfloat16 models and quantized models (including FP8, GPTQ, AWQ) of the Qwen3 series. 
Specifically, we report the inference speed (tokens/s) as well as memory footprint (GB) under different context lengths.

## Environments

### Hugging Face Transformers

- **Hardware**: 
  - NVIDIA H20 96GB
- **Software for Non-AutoAWQ**:
  - PyTorch 2.6.0
  - Flash Attention 2.7.4
  - Transformers 4.51.3
  - GPTQModel 2.2.0+cu128torch2.6
- **Software for AutoAWQ**:
  - PyTorch 2.6.0+cu124
  - Transformers 4.51.3
  - AutoAWQ 0.2.9
  - AutoAWQ_kernels 0.0.9


### SGLang
- **Hardware**: 
  - NVIDIA H20 96GB
- **Software**:
  - PyTorch 2.6.0+cu124
  - Transformers 4.51.3
  - SGLang 0.4.6.post1
  - SGL-kernel 0.1.0
  - vLLM 0.7.2 (Required by SGLang for AWQ quantization)

## Notes

- **Inference Speed (tokens/s)** is calculated as:  
  
  ```{math}
  \text{Speed} = \frac{\text{tokens}_{\text{prompt}} + \text{tokens}_{\text{generation}}}{\text{time}}
  ```

- We use a **batch size of 1** and the **minimum number of GPUs** possible for evaluation.

- We test the **speed and memory usage** when generating **2048 tokens**, with input lengths of
  `1`, `6144`, `14336`, `30720`, `63488`, and `129024` tokens.

- **For SGLang**:
  - **Memory usage** is not reported because SGLang pre-allocates all GPU memory.  
    By default, we set `mem_fraction_static=0.85`.
  - We configure `context_length=140000` and enable `enable_mixed_chunk=True`.
  - For **AWQ quantization**, we use the **awq_marlin** backend.
  - We set `skip_tokenizer_init=True` and perform generation using `input_ids` instead of raw text prompts.

- **FP8 Performance in Transformers**: The inference speed of Transformers in FP8 mode is currently not optimal and requires further optimization.

- **GPTQ-INT4 Performance in SGLang**: The performance of GPTQ-INT4 in SGLang also needs improvement, and we are actively working with the team to enhance it.

## Results

### Qwen3-0.6B (SGLang)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-0.6B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>414.17</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>458.03</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>344.92</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>1426.46</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1572.95</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>1234.29</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>2478.02</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>2689.08</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>2198.82</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>3577.42</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>3819.86</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>3342.06</td><td></td>
        </tr>
    </tbody>
</table>

### Qwen3-0.6B (Transformers)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory(MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-0.6B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>58.57</td><td>1394</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>24.60</td><td>1217</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>26.56</td><td>986</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>154.82</td><td>2066</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>73.96</td><td>1943</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>93.84</td><td>1658</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>168.48</td><td>2963</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>104.99</td><td>2839</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>219.61</td><td>2554</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>175.93</td><td>4755</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>132.78</td><td>4632</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>345.71</td><td>4347</td>
        </tr>
    </tbody>
</table>


### Qwen3-1.7B (SGLang)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-1.7B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>227.80</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>333.90</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>257.40</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>838.28</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1198.20</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>945.91</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>1525.71</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>2095.61</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>1707.63</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>2439.03</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>3165.32</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>2706.16</td><td></td>
        </tr>
    </tbody>
</table>



### Qwen3-1.7B (Transformers)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory(MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-1.7B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>59.83</td><td>3412</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>23.83</td><td>2726</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>28.06</td><td>2229</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>238.53</td><td>4213</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>90.87</td><td>3462</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>110.82</td><td>2901</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>352.59</td><td>5109</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>153.37</td><td>4359</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>222.78</td><td>3798</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>418.13</td><td>6902</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>235.61</td><td>6151</td>
        </tr>
        <tr>
            <td>GPTQ-Int8</td><td>1</td><td>386.85</td><td>5590</td>
        </tr>
    </tbody>
</table>

### Qwen3-4B (SGLang)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-4B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>133.13</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>200.61</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>199.71</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>466.19</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>662.26</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>640.07</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>789.25</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1066.23</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1006.23</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>1165.75</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1467.71</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1358.84</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>1</td><td>1423.98</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1660.67</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1513.97</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>1</td><td>1371.04</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1497.27</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1375.71</td><td></td>
        </tr>
    </tbody>
</table>

### Qwen3-4B (Transformers)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory(MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="15">Qwen3-4B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>45.94</td><td>7973</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>17.33</td><td>5281</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>51.57</td><td>2915</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>159.95</td><td>8860</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>60.55</td><td>6144</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>183.04</td><td>3881</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>195.31</td><td>10012</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>96.81</td><td>7297</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>265.22</td><td>5151</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>217.97</td><td>12317</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>138.84</td><td>9611</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>481.69</td><td>7742</td>
        </tr>
    </tbody>
</table>

### Qwen3-8B (SGLang)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-8B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>81.73</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>150.25</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>144.11</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>296.25</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>516.64</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>477.89</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>524.70</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>859.92</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>770.44</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>832.67</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1242.24</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1075.91</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>1</td><td>1112.78</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1476.46</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1254.91</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>1</td><td>1173.32</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1393.21</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>1198.06</td><td></td>
        </tr>
    </tbody>
</table>


### Qwen3-8B (Transformers)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory(MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-8B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>45.32</td><td>15947</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>15.46</td><td>9323</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>51.33</td><td>6177</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>146.12</td><td>16811</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>55.07</td><td>10187</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>163.23</td><td>7113</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>183.29</td><td>17963</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>89.64</td><td>11340</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>242.97</td><td>8409</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>208.98</td><td>20267</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>130.93</td><td>13644</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>438.62</td><td>11001</td>
        </tr>
    </tbody>
</table>


### Qwen3-14B (SGLang)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-14B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>47.10</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>97.11</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>96.49</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>174.85</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>342.95</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>321.62</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>317.56</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>587.33</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>525.74</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>525.80</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>880.72</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>744.74</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>1</td><td>742.36</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1089.04</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>884.06</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>1</td><td>826.15</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1049.64</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>857.56</td><td></td>
        </tr>
    </tbody>
</table>


### Qwen3-14B (Transformers)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="15">Qwen3-14B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>40.66</td><td>28402</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>13.02</td><td>16012</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>44.67</td><td>9962</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>108.52</td><td>29495</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>44.86</td><td>16972</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>128.08</td><td>11020</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>136.36</td><td>30775</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>71.96</td><td>18253</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>220.62</td><td>12438</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>155.38</td><td>33336</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>102.63</td><td>20813</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>363.25</td><td>15323</td>
        </tr>
    </tbody>
</table>


### Qwen3-32B (SGLang)
    
<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-32B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>20.72</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>46.17</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>47.67</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>77.82</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>165.71</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>159.99</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>143.08</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>287.60</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>260.44</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>240.75</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>436.59</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>366.84</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>1</td><td>342.96</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>532.18</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>425.23</td><td></td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>2</td><td>711.40</td><td>TP=2</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>491.45</td><td></td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>395.96</td><td></td>
        </tr>
    </tbody>
</table>



### Qwen3-32B (Transformers)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="15">Qwen3-32B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>26.24</td><td>62751</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>7.37</td><td>33379</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>41.8</td><td>19109</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>51.41</td><td>64583</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>23.57</td><td>34915</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>68.71</td><td>20795</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>62.41</td><td>66632</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>36.30</td><td>36963</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>107.02</td><td>23105</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>69.16</td><td>70728</td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>49.44</td><td>41060</td>
        </tr>
        <tr>
            <td>AWQ-INT4</td><td>1</td><td>188.11</td><td>27718</td>
        </tr>
    </tbody>
</table>

### Qwen3-30B-A3B (SGLang)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-30B-A3B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>137.18</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>155.55</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>31.29</td><td>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>490.10</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>551.34</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>120.13</td><td>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>849.62</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>945.13</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>227.27</td><td>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>1283.94</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1405.91</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>404.45</td><td>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>1</td><td>1538.79</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1647.89</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>617.09</td><td>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>1</td><td>1385.65</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1442.14</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>1</td><td>704.82</td><td>GPTQ-Marlin</td>
        </tr>
    </tbody>
</table>

### Qwen3-30B-A3B (Transformers)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>GPU Memory (MB)</th>
            <th>Notes</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="12">Qwen3-30B-A3B</td>
            <td rowspan="3">1</td><td>BF16</td><td>1</td><td>1.89</td><td>58462</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>0.44</td><td>30296</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>-</td><td>-</td><td>-</td><td>MoE Kernel Unsupported</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>1</td><td>7.45</td><td>59037</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>1.77</td><td>30872</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>-</td><td>-</td><td>-</td><td>MoE Kernel Unsupported</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>1</td><td>14.47</td><td>59806</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>3.5</td><td>31641</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>-</td><td>-</td><td>-</td><td>MoE Kernel Unsupported</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>1</td><td>27.03</td><td>61342</td><td></td>
        </tr>
        <tr>
            <td>FP8</td><td>1</td><td>6.86</td><td>33177</td><td></td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>-</td><td>-</td><td>-</td><td>MoE Kernel Unsupported</td>
        </tr>
    </tbody>
</table>

### Qwen3-235B-A22B (SGLang)

<table border="1" cellpadding="8" style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th>Model</th>
            <th>Input Length</th>
            <th>Quantization</th>
            <th>GPU Num</th>
            <th>Speed (tokens/s)</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="18">Qwen3-235B-A22B</td>
            <td rowspan="3">1</td><td>BF16</td><td>8</td><td>74.50</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>71.65</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>14.69</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">6144</td><td>BF16</td><td>8</td><td>289.03</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>275.16</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>56.97</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">14336</td><td>BF16</td><td>8</td><td>546.73</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>514.23</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>109.13</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">30720</td><td>BF16</td><td>8</td><td>979.41</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>887.90</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>198.99</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">63488</td><td>BF16</td><td>8</td><td>1493.91</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>1269.34</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>422.77</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
        <tr>
            <td rowspan="3">129042</td><td>BF16</td><td>8</td><td>1639.54</td><td>TP=8</td>
        </tr>
        <tr>
            <td>FP8</td><td>4</td><td>1319.66</td><td>TP=4</td>
        </tr>
        <tr>
            <td>GPTQ-INT4</td><td>4</td><td>552.28</td><td>TP=4<br>GPTQ-Marlin</td>
        </tr>
    </tbody>
</table>