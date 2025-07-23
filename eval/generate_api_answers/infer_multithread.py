import json
import argparse
from tqdm import tqdm
import copy
import concurrent.futures
import threading
import os
import collections
import yaml

from utils_vllm import get_content

file_lock = threading.Lock()


def count_completed_samples(output_file):
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]
                    gen_count = len(item.get("gen", []))
                    prompt_counts[prompt] += gen_count
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def process_item(
    item,
    output_file,
    base_url,
    model_name,
    temperature,
    top_p,
    max_tokens,
    top_k,
    presence_penalty,
):
    result = copy.deepcopy(item)

    response = get_content(
        item["prompt"],
        base_url,
        model_name,
        temperature,
        top_p,
        max_tokens,
        top_k,
        presence_penalty,
    )

    if "gen" not in result:
        result["gen"] = []

    result["gen"].append(response)
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as g:
            g.write(json.dumps(result, ensure_ascii=False) + "\n")
            g.flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on model with prompts from a jsonl file, configurable via YAML."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    config_file_path = args.config
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(
            f"Error: Configuration file '{config_file_path}' not found. Please check the path."
        )
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{config_file_path}':\n{e}")
        return
    except Exception as e:
        print(f"An unknown error occurred while loading the configuration file: {e}")
        return

    input_file = config.get("input_file")
    output_file = config.get("output_file")

    if input_file is None:
        print(
            "Error: Required parameter 'input_file' is missing in the YAML configuration file."
        )
        return
    if output_file is None:
        print(
            "Error: Required parameter 'output_file' is missing in the YAML configuration file."
        )
        return

    n_samples = config.get("n_samples", 1)
    max_workers = config.get("max_workers", 128)
    base_url = config.get("base_url", "http://10.77.249.36:8030/v1")
    model_name = config.get("model_name", "Qwen/QwQ-32B")
    top_p = config.get("top_p", 0.7)
    temperature = config.get("temperature", 0.8)
    top_k = config.get("top_k", 20)
    max_tokens = config.get("max_tokens", 32768)
    presence_penalty = config.get("presence_penalty", 1.0)

    print("\n--- Configuration Information ---")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print(f"Number of Samples per Prompt: {n_samples}")
    print(f"Maximum Workers: {max_workers}")
    print(f"VLLM Server Base URL: {base_url}")
    print(f"VLLM Server Model Name: {model_name}")
    print(f"Top-p: {top_p}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print(f"Max Generation Tokens: {max_tokens}")
    print(f"Presence Penalty: {presence_penalty}")
    print("----------------\n")

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    if os.path.exists(output_file):
        completed_counts = count_completed_samples(output_file)
        total_completed = sum(completed_counts.values())
        print(f"Found {total_completed} completed samples from previous run")
    else:
        with open(output_file, "w", encoding="utf-8") as g:
            completed_counts = dict()

    expanded_data = []
    for item in data:
        prompt = item["prompt"]
        completed = completed_counts.get(prompt, 0)
        remaining = n_samples - completed
        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))

    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                process_item,
                item,
                output_file,
                base_url,
                model_name,
                temperature,
                top_p,
                max_tokens,
                top_k,
                presence_penalty,
            ): i
            for i, item in enumerate(expanded_data)
        }

        with tqdm(total=len(expanded_data), desc="Processing samples") as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    future.result()
                    completed_count += 1
                except Exception as exc:
                    print(f"Error processing sample {idx}: {exc}")
                pbar.update(1)

    print(f"Completed {completed_count}/{len(expanded_data)} samples")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
