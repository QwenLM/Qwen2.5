import timeit
import ctypes
import argparse
import os
import sys
import json
import configparser
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from threading import Thread

from ctypes import *

class QwenTP2(object):
    def __init__(
        self,
        token_config_path,
        batch,
        max_length,
        max_tokens,
        ini_path,
        weight_path,
        rank_num,
        lib_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
                token_config_path, trust_remote_code=True
            )
        self.batch = batch
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.ini_path = ini_path
        self.weight_path = weight_path
        self.eos_id = self.tokenizer.eos_token_id
        self.lib = ctypes.CDLL(lib_path)
       
        assert(rank_num in [2, 4])
        self.rank_num = rank_num

    def model_init(self):
        self.lib.qwen_init(c_int(self.rank_num))

    def model_finish(self):
        self.lib.qwen_finish()

    def model_run(self):
        self.lib.qwen_work(self.ini_path.encode('utf-8'), self.weight_path.encode('utf-8'),
                      c_int(self.batch), c_int(self.max_tokens), c_int(self.max_length),
                      c_int(self.rank_num), c_int(self.eos_id))

    def token_encode_qwen(self, prompt):
        # start = timeit.default_timer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.tokenizer([text], return_tensors="np")
        res = tokens["input_ids"]
        res = res.tolist()
        # end=timeit.default_timer()
        # print('Running encode time: %s Seconds'%(end-start))
        return res

    def token_decode_qwen(self, tokens):
        response = self.tokenizer.decode(tokens)
        return response

    def push_input(self, prompt, index, max_generate_length, is_close=False):
        input_ids = self.token_encode_qwen(prompt)[0]
        length = len(input_ids)
        if length <= 0:
            raise ValueError(f"seq_len must be at least 1, got {length}.")
        if length > self.max_length:
            raise ValueError(f"model only support max seq_len {self.max_length}, got {length}.")
        input = (c_int * length)(*input_ids)
        temp = 0.0
        top_p = 1.0
        top_k = -1
        seed = 1
        alpha_presence = 0.0
        alpha_frequency = 0.0
        repetition_penalty = 1.0
        status = self.lib.qwen_push_input(input, c_int(length), c_int(index),
                                     c_int(max_generate_length), c_bool(is_close),c_float(temp), c_float(top_p), c_int(top_k), c_uint32(seed),
                c_float(alpha_presence), c_float(alpha_frequency),
                c_float(repetition_penalty), c_bool(False))
        return status

    def push_input_for_benchmark(self, seq_length, index, max_generate_length, is_close=False):
        input_ids = [i+3 for i in range(seq_length)]
        length = len(input_ids)
        if length <= 0:
            raise ValueError(f"seq_len must be at least 1, got {length}.")
        if length > self.max_length:
            raise ValueError(f"model only support max seq_len {self.max_length}, got {length}.")
        input = (c_int * length)(*input_ids)
        temp = 0.0
        top_p = 1.0
        top_k = -1
        seed = 1
        alpha_presence = 0.0
        alpha_frequency = 0.0
        repetition_penalty = 1.0
        status = self.lib.qwen_push_input(input, c_int(length), c_int(index),
                                     c_int(max_generate_length), c_bool(is_close),c_float(temp), c_float(top_p), c_int(top_k), c_uint32(seed),
                c_float(alpha_presence), c_float(alpha_frequency),
                c_float(repetition_penalty), c_bool(True))
        return status

    def pop_output(self):
        output_num_per_pop = 1
        outs = c_int * output_num_per_pop
        indexes = c_int * output_num_per_pop
        first_times = c_double * output_num_per_pop
        actual_times = c_double * output_num_per_pop
        inner_times = c_double * output_num_per_pop
        is_ends = c_bool * output_num_per_pop
        outs_obj = outs()
        indexes_obj = indexes()
        first_times_obj = first_times()
        actual_times_obj = actual_times()
        inner_times_obj = inner_times()
        is_ends_obj = is_ends()
        status = self.lib.qwen_pop_output(outs_obj, indexes_obj, first_times_obj, actual_times_obj, inner_times_obj, is_ends_obj)
        outputs = [outs_obj[i] for i in range(output_num_per_pop)]
        indexes = [indexes_obj[i] for i in range(output_num_per_pop)]
        is_ends = [is_ends_obj[i] for i in range(output_num_per_pop)]
        if is_ends[0]:
            first_times = [first_times_obj[i] for i in range(output_num_per_pop)]
            actual_times = [actual_times_obj[i] for i in range(output_num_per_pop)]
            inner_times = [inner_times_obj[i] for i in range(output_num_per_pop)]
        return outputs, indexes, first_times, actual_times, inner_times, is_ends, status


def main(token_config_path, ini_path, weight_path, lib_path, batch, max_length, rank_num, text_path=None):

    max_tokens = max_length
    rank_num = rank_num

    model = QwenTP2(token_config_path, batch, max_length, max_tokens,
                    ini_path, weight_path, rank_num, lib_path)
    model.model_init()
    def work():
        model.model_run()
    work_thread = Thread(target=work, name="work")
    work_thread.start()
    def push_input():
        if text_path is None:
            for i in range(7):
                assert(model.push_input("你好", i, max_length, False) == 0)
            assert(model.push_input("你好", i, max_length, True) == 1)
        else:
            with open(text_path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    assert(model.push_input(line, i, max_length, False) == 0)
            assert(model.push_input("你好", i, max_length, True) == 1)

    push_thread = Thread(target=push_input, name="push")
    push_thread.start()

    def pop_output():
        out_tokens = {}
        time_tokens = {}
        step0_time = []
        stepn_time = []
        while(True):
            outputs, indexes, first_times, actual_times, inner_times, is_ends, status = model.pop_output()
            if (status == 0):
                if not out_tokens.get(indexes[0]):
                    out_tokens[indexes[0]] = outputs
                else:
                    out_tokens[indexes[0]] += outputs
                if is_ends[0] :
                    print("index: {} output: {}".format(indexes[0], model.token_decode_qwen(out_tokens[indexes[0]])))
            if (status == 1):
                break

    pop_thread = Thread(target=pop_output, name="pop")
    pop_thread.start()

    work_thread.join()
    push_thread.join()
    pop_thread.join()
    model.model_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen1.5 TP2 model")
    parser.add_argument(
        "-a",
        "--token_config_path",
        type=str,
        default="Qwen/Qwen1.5-14B-Chat",
        help="the token config path"
    )
    parser.add_argument(
        "-t", "--test_txt", type=str, default=None,
        help="test txt for model running, default none for throughput test"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int,
        default=32, help="inference batch size"
    )
    parser.add_argument(
        "-m", "--max_seq_length", type=int,
        default=2048, help="max_seq_length"
    )
    parser.add_argument(
        "-l", "--lib_path", default="/usr/lib/libtopstransformer.so", help="the path of libtopstransformer.so"
    )
    parser.add_argument(
        "-i", "--ini_path", default=None, help="the path of .ini"
    )
    parser.add_argument(
        "-w", "--weights_path", default=None, help="the path of weights"
    )

    parser.add_argument(
        "-s", "--seq_length", type=int, default=1024, help="input seq_length for benchmark"
    )

    parser.add_argument(
        "-g", "--max_generate_length", type=int, default=1056, help="total seq_length for benchmark"
    )

    parser.add_argument(
        "-r", "--repeat_time", type=int, default=10, help="total seq_length for benchmark"
    )

    parser.add_argument(
        "-tp", "--tensor_parallel_size", type=int, default=2, help="tensor parallel size"
    )

    args = parser.parse_args()
    token_config_path = args.token_config_path
    ini_path = args.ini_path
    weight_path = args.weights_path
    lib_path = args.lib_path
    batch = args.batch_size
    max_length = args.max_seq_length
    text_path = args.test_txt
    rank_num = args.tensor_parallel_size
   
    main(token_config_path, ini_path, weight_path, lib_path, batch, max_length, rank_num, text_path)
