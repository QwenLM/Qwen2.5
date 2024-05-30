import os
import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM

# parse arguments
parser = argparse.ArgumentParser(description="parse ChatGLM model weight and save seperately in nd array")
parser.add_argument("-tp", "--tensor_parallel_size", type=int, help="tensor parallel size")
parser.add_argument("-i", "--model_path", type=str, help="the model file")
parser.add_argument("-o", "--weight_out_path", type=str, help="output seperated weight path")
parser.add_argument("-cfg", "--json_cfg", type=str, default=None, help="model json file path")
parser.add_argument("-se", "--split_embedding", type=bool, default=False, help="whether split embedding layer or not")
parser.add_argument("-sl", "--split_lmout", type=bool, default=True, help="whether split lmout layer or not")

args = parser.parse_args()
tp_size = args.tensor_parallel_size

if args.json_cfg is None:
  # default
  intermediate_size = 13696
  hidden_size = 5120
  rope_theta = 1000000.0
  head_dim = 128
  vocab_size = 152064
else:
  with open(args.json_cfg, 'r') as fcc_file:
    json_dict = json.load(fcc_file)

  intermediate_size = json_dict['intermediate_size']
  hidden_size = json_dict['hidden_size']
  rope_theta = json_dict['rope_theta']
  head_dim = hidden_size / json_dict['num_attention_heads']

q = intermediate_size // tp_size # 4h per card
p = int(np.ceil(q / 256) * 256) # padded q
q2 = q * 2
p2 = p * 2

def remove_useless_bytes(input_file, output_file):
  """
  Removes the first two bytes (00 00) from every four bytes in a binary file.

  Args:
    input_file: Path to the input binary file.
    output_file: Path to the output binary file.
  """
  with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
    data = infile.read(4)  # Read 4 bytes at a time
    while data:
      outfile.write(data[2:])  # Write only the last two bytes
      data = infile.read(4)

def weight_split(weight, rank_num, split_axis, base_name, model_out_path, split_flag=True):
  shape = weight.shape
  dims = len(shape)
  dim_size = shape[split_axis]
  assert split_axis <= dims, f"dim must less than {dims}"
  assert dim_size % rank_num == 0, f"rank num must less than {dim_size}"
  if split_flag:
    split_weights = np.split(weight, rank_num, split_axis)
  else:
    split_weights = [weight] * rank_num
  for i in range(rank_num):
    output_file = f"{model_out_path}/rank_{i}/{base_name}"
    split_weights[i].tofile(output_file)
    # output_file2 = output_file + "2"
    # # 保存切分后的数据到二进制文件
    # split_weights[i].tofile(output_file2)
    # remove_useless_bytes(output_file2, output_file)

def dump_json_and_weight(tp_size, model_path, model_out_path):

  model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
  
  # model = model.float() # numpy does not support bf16. Export in fp32 then truncate to bf16
  model = model.half()

  if os.path.exists(model_out_path):
    print(f"model_out_path: {model_out_path} already exist!!!")
    exit()
  else:
    os.makedirs(model_out_path)
    
  os.makedirs(f'{model_out_path}/total')
  for tp_rank in range(tp_size):
    os.makedirs(f'{model_out_path}/rank_{tp_rank}')
  
  weight_info = []
  index = 0
  flag = (tp_size != 1)
  flag_se = args.split_embedding
  flag_sl = args.split_lmout

  dtype_map = {torch.float32:0, torch.float16:1, torch.int8:2, torch.bfloat16:3}

  inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

  for layer, param in list(model.state_dict().items()) + [("inv_freq", inv_freq)]:
    print("process weight layer: {}, param: {} {}".format(layer, param.shape, param.dtype))
    # continue
    index += 1
    dtype = dtype_map[param.dtype]
    rank = len(param.shape)
    shape = []
    for i in range(rank):
      shape.append(param.shape[i])
    tensor_numpy = param.numpy()
    tensor_numpy.tofile(f'{model_out_path}/total/{layer}.bin')

    def process_weight(tensor, name, split_axis, split_flag=True) :
      weight_split(tensor, tp_size, split_axis, name + ".bin", model_out_path, split_flag)
      info_dict = {"name": name, "dtype": dtype, "rank": rank, "shape": tensor.shape}
      weight_info.append(info_dict)

    if "embed_tokens.weight" in layer:
      process_weight(tensor_numpy, layer, 1, flag_se) #切5120

    elif "inv_freq" in layer:
      process_weight(tensor_numpy, layer, 0, False)
    
    elif "self_attn" in layer:

      if "o_proj.weight" in layer:
        process_weight(tensor_numpy, layer, 1, flag) #行切分

        bias = np.zeros([hidden_size], dtype=np.float16) #qwen的o_proj没有bias，补零
        process_weight(bias, layer.replace('weight', 'bias'), 0, False)

      else: #qkv weight and bias
        process_weight(tensor_numpy, layer, 0, flag) #列切分

    elif "mlp.gate" in layer:
      gate = param
      continue
    elif "mlp.up" in layer:
      # each card should have 1/tp_size of gate and up
      gate_numpy = gate.numpy()
      weight = np.zeros([p2*tp_size, hidden_size], dtype=np.float16)
      for i in range(tp_size):
        # print(gates[i].shape)
        # print(ups[i].shape)
        print(p, q, i*p2, i*p2+q, i*p2+p, i*p2+p+q, i*q, i*q+q)
        weight[i*p2:i*p2+q, :] = gate_numpy[i*q:i*q+q, :]
        weight[i*p2+p:i*p2+p+q, :] = tensor_numpy[i*q:i*q+q, :]
      layer = layer.replace("up_proj", "gateup_proj")
      process_weight(weight, layer, 0, flag)

      bias = np.zeros([p*2*tp_size], dtype=np.float16)
      process_weight(bias, layer.replace('weight', 'bias'), 0, flag)

    elif "mlp.down_proj" in layer:
      weight = np.zeros([hidden_size, p*tp_size], dtype=np.float16)
      for i in range(tp_size):
        weight[:, i*p:i*p+q] = tensor_numpy[:, i*q:(i+1)*q]
      process_weight(weight, layer, 1, flag)
    
      bias = np.zeros([hidden_size], dtype=np.float16)
      process_weight(bias, layer.replace('weight', 'bias'), 0, flag)
    
    elif "lm_head.weight" in layer:
      vocab_each = vocab_size // tp_size
      vocab_each_pad = int(np.ceil(vocab_each/256)*256)
      print("vocab_each", vocab_each, vocab_each_pad)
      if vocab_each_pad != vocab_each:
        tensor_pad = np.zeros([vocab_each_pad * tp_size, hidden_size], dtype=np.float16)
        tensor_pad[:vocab_size, :] = tensor_numpy[:, :]
        tensor_numpy = tensor_pad
        print(vocab_each_pad * tp_size)
      process_weight(tensor_numpy, layer, 0, flag_sl)

    elif "norm" in layer:
      process_weight(tensor_numpy, layer, 0, False)
    
    else:
      print("TODO:", layer)
      assert False

  model_info = {"weight_info":weight_info}
  json_path = os.path.join(model_out_path, "weight_info.json")
  with open(json_path, "w") as f:
    json.dump(model_info, f, indent=2)

def main(tp_size, model_path, weight_out_path):
  dump_json_and_weight(tp_size, model_path, weight_out_path)

if __name__ == '__main__':
  model_path = args.model_path
  model_out_path = args.weight_out_path
  main(tp_size, model_path, model_out_path)
