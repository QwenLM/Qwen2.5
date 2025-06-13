# Qwen2.5-Instruct Chat Demo
import argparse
import os
import platform
import shutil
from copy import deepcopy
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = "Qwen/Qwen2.5-7B-Instruct"
_WELCOME_MSG = """
Welcome to Qwen2.5-Instruct model. Type text to start chat, type :h for help.
Note: Follow the original license, avoid harmful content generation.
"""
_HELP_MSG = """
Commands:
    :help / :h            Show help
    :exit / :q            Exit
    :clear / :cl          Clear screen
    :clear-history / :clh Clear chat history
    :history / :his       Show chat history
    :seed                 Show current seed
    :seed <N>             Set random seed
    :conf                 Show current config
    :conf <key>=<value>   Modify config
    :reset-conf           Reset config
"""
_ALL_COMMAND_NAMES = ["help", "h", "exit", "q", "clear", "cl", "clear-history", "clh", "history", "his", "seed", "conf", "reset-conf"]

class QwenChatDemo:
    def __init__(self, checkpoint_path=DEFAULT_CKPT_PATH, seed=1234, cpu_only=False):
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.cpu_only = cpu_only
        self.history = []
        self.model, self.tokenizer = self._load_model_tokenizer()
        self.orig_gen_config = deepcopy(self.model.generation_config)
        set_seed(self.seed)

    def _load_model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, resume_download=True)
        device_map = "cpu" if self.cpu_only else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, torch_dtype="auto", device_map=device_map, resume_download=True
        ).eval()
        model.generation_config.max_new_tokens = 2048
        return model, tokenizer

    def _gc(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_screen(self):
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def _print_history(self):
        terminal_width = shutil.get_terminal_size()[0]
        print(f"History ({len(self.history)})".center(terminal_width, "="))
        for index, (query, response) in enumerate(self.history):
            print(f"User[{index}]: {query}")
            print(f"Qwen[{index}]: {response}")
        print("=" * terminal_width)

    def _get_input(self):
        while True:
            try:
                message = input("User> ").strip()
            except (UnicodeDecodeError, KeyboardInterrupt):
                print("[ERROR] Invalid input or interrupted.")
                continue
            if message:
                return message
            print("[ERROR] Query is empty")

    def _chat_stream(self, query):
        conversation = [{"role": "user", "content": q} for q, _ in self.history]
        conversation.extend([{"role": "assistant", "content": r} for _, r in self.history])
        conversation.append({"role": "user", "content": query})
        input_text = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs={"streamer": streamer, **inputs})
        thread.start()
        return streamer

    def _handle_command(self, command, command_words):
        if command in ["exit", "q"]:
            return False
        elif command in ["clear", "cl"]:
            self._clear_screen()
            print(_WELCOME_MSG)
            self._gc()
        elif command in ["clear-history", "clh"]:
            self.history.clear()
            self._gc()
        elif command in ["help", "h"]:
            print(_HELP_MSG)
        elif command in ["history", "his"]:
            self._print_history()
        elif command in ["seed"]:
            if len(command_words) == 1:
                print(f"[INFO] Current random seed: {self.seed}")
            else:
                try:
                    self.seed = int(command_words[1])
                    print(f"[INFO] Random seed changed to {self.seed}")
                except ValueError:
                    print(f"[WARNING] Invalid seed: {command_words[1]}")
        elif command in ["conf"]:
            if len(command_words) == 1:
                print(self.model.generation_config)
            else:
                for key_value_str in command_words[1:]:
                    key, value = key_value_str.split("=", 1)
                    try:
                        setattr(self.model.generation_config, key, eval(value))
                    except Exception as e:
                        print(e)
        elif command in ["reset-conf"]:
            self.model.generation_config = deepcopy(self.orig_gen_config)
            print("[INFO] Reset generation config.")
        return True

    def run(self):
        self._clear_screen()
        print(_WELCOME_MSG)
        while True:
            query = self._get_input()
            if query.startswith(":"):
                command_words = query[1:].split()
                if not self._handle_command(command_words[0], command_words):
                    break
            else:
                self._clear_screen()
                print(f"\nUser: {query}\nQwen: ", end="")
                response = ""
                try:
                    for new_text in self._chat_stream(query):
                        print(new_text, end="", flush=True)
                        response += new_text
                except KeyboardInterrupt:
                    print("[WARNING] Generation interrupted")
                    continue
                self.history.append((query, response))

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Instruct command-line interactive chat demo.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH, help="Checkpoint path")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run with CPU only")
    args = parser.parse_args()

    chat_demo = QwenChatDemo(args.checkpoint_path, args.seed, args.cpu_only)
    chat_demo.run()

if __name__ == "__main__":
    main()
