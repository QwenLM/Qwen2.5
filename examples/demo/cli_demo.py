# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo"""

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

_WELCOME_MSG = """\
Welcome to use Qwen2.5-Instruct model. Type text to start chat, type :h to show command help.
(欢迎使用 Qwen2.5-Instruct 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen2.5.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, 
including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen2.5的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
"""

_HELP_MSG = """\
Commands:
    :help / :h              Show this help message              
    :exit / :quit / :q      Exit the demo                       
    :clear / :cl            Clear screen                        
    :clear-history / :clh   Clear history                       
    :history / :his         Show history                        
    :seed                   Show current random seed            
    :seed <N>               Set random seed to <N>              
    :conf                   Show current generation config      
    :conf <key>=<value>     Change generation config            
    :reset-conf             Reset generation config             
"""

_ALL_COMMAND_NAMES = [
    "help",
    "h",
    "exit",
    "quit",
    "q",
    "clear",
    "cl",
    "clear-history",
    "clh",
    "history",
    "his",
    "seed",
    "conf",
    "reset-conf",
]


def _setup_readline():
    """
    Sets up autocompletion for command names if readline is available.

    This function is a minor convenience for interactive usage
    and does not affect core functionality.
    """
    try:
        import readline
    except ImportError:
        return

    _matches = []

    def _completer(text, state):
        nonlocal _matches
        if state == 0:
            _matches = [
                cmd_name for cmd_name in _ALL_COMMAND_NAMES if cmd_name.startswith(text)
            ]
        return _matches[state] if 0 <= state < len(_matches) else None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")


def _load_model_tokenizer(args):
    """
    Loads the model and tokenizer.

    Optimization tips:
        - If you have the model locally, you can remove `resume_download=True`.
        - If you have enough GPU memory, you can force device_map to 'auto' or a custom mapping.
        - For CPU-only inference, device_map is set to 'cpu'.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        # 'auto' tries to shard the model automatically across GPUs (if multiple GPUs are available).
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    )
    model.eval()

    # Setting a default max_new_tokens for chat usage.
    model.generation_config.max_new_tokens = 2048

    return model, tokenizer


def _gc():
    """
    A helper function to clear Python garbage and (if GPU is used) empty CUDA cache.
    Called after large memory releases (like clearing the history).
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    """
    Cross-platform clear-screen command.
    """
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    """
    Utility to print all conversation history.
    """
    terminal_width = shutil.get_terminal_size((80, 20))[0]  # fallback if size not found
    title = f"History ({len(history)})"
    print(title.center(terminal_width, "="))
    for index, (query, response) in enumerate(history):
        print(f"User[{index}]: {query}")
        print(f"Qwen[{index}]: {response}")
    print("=" * terminal_width)


def _get_input() -> str:
    """
    Reads a non-empty user input from stdin, with some basic error handling.
    """
    while True:
        try:
            message = input("User> ").strip()
        except UnicodeDecodeError:
            print("[ERROR] Encoding error in input")
            continue
        except KeyboardInterrupt:
            print("\n[INFO] Exiting (Keyboard Interrupt).")
            exit(1)
        if message:
            return message
        print("[ERROR] Query is empty, please try again.")


def _generate_stream(model, tokenizer, query, history):
    """
    Creates a streaming generation for the user query given the conversation history.
    
    Yields new text tokens as they are generated by the model.
    """
    # Rebuild conversation from history
    conversation = []
    for q, r in history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": r})
    conversation.append({"role": "user", "content": query})

    # Prepare the model input
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Set up a streamer to yield text incrementally
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }

    # Launch a background thread for generation so we can yield tokens progressively
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Instruct command-line interactive chat demo (optimized)."
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default: %(default)r",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1234, help="Random seed (default: 1234)"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    args = parser.parse_args()

    # Model & Tokenizer
    model, tokenizer = _load_model_tokenizer(args)
    # Keep a copy of the original config for reset
    orig_gen_config = deepcopy(model.generation_config)

    # Setup environment
    _setup_readline()
    _clear_screen()
    print(_WELCOME_MSG)

    # Track conversation history and seed
    history = []
    current_seed = args.seed

    while True:
        # 1) Get user input
        query = _get_input()

        # 2) Check for commands
        if query.startswith(":"):
            command_line = query[1:].strip().split()
            if not command_line:
                continue  # user typed just ":", ignore

            cmd = command_line[0]
            args_rest = command_line[1:]

            if cmd in ["exit", "quit", "q"]:
                print("[INFO] Exiting demo.")
                break

            elif cmd in ["clear", "cl"]:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()

            elif cmd in ["clear-history", "clh"]:
                print(f"[INFO] Clearing all {len(history)} conversation history.")
                history.clear()
                _gc()

            elif cmd in ["help", "h"]:
                print(_HELP_MSG)

            elif cmd in ["history", "his"]:
                _print_history(history)

            elif cmd in ["seed"]:
                if len(args_rest) == 0:
                    print(f"[INFO] Current random seed: {current_seed}")
                else:
                    try:
                        new_seed = int(args_rest[0])
                        current_seed = new_seed
                        print(f"[INFO] Random seed changed to: {new_seed}")
                    except ValueError:
                        print(f"[ERROR] {args_rest[0]!r} is not a valid integer.")

            elif cmd in ["conf"]:
                if not args_rest:
                    # Show current config
                    print(model.generation_config)
                else:
                    for key_value_str in args_rest:
                        if "=" not in key_value_str:
                            print("[WARNING] Format should be :conf <key>=<value>")
                            continue
                        key, val_str = key_value_str.split("=", 1)
                        # Evaluate val_str carefully or parse manually
                        # For safety, you may limit to int/float or certain patterns
                        # Here we do a basic eval but caution is advised in production
                        try:
                            val = eval(val_str)
                        except Exception as e:
                            print(f"[WARNING] Could not parse {val_str!r}: {e}")
                            continue
                        setattr(model.generation_config, key, val)
                        print(f"[INFO] Set model.generation_config.{key} = {val}")

            elif cmd in ["reset-conf"]:
                print("[INFO] Resetting generation config to original.")
                model.generation_config = deepcopy(orig_gen_config)

            else:
                # If unknown command, treat as normal query
                pass

            # Since it was a recognized command, skip generation cycle.
            continue

        # 3) Run actual generation
        
         --------------------
        set_seed(current_seed)  # Setting seed for reproducibility

        # Clear screen just before generating the new response
        # so we have a "fresh" look for each conversation round.
        _clear_screen()
        print(f"\nUser: {query}")
        print(f"\nQwen: ", end="", flush=True)

        response_text = []
        try:
            for new_token in _generate_stream(model, tokenizer, query, history):
                response_text.append(new_token)
                print(new_token, end="", flush=True)
            print()  # add a newline
        except KeyboardInterrupt:
            print("\n[WARNING] Generation interrupted by user.")
            # Optional: you can skip adding incomplete response to history
            continue

        full_response = "".join(response_text)
        history.append((query, full_response))


if __name__ == "__main__":
    main()
