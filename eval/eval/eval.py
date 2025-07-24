import json
import argparse
from tqdm import tqdm
import os
import yaml

ALL_TASKS = {}

from arc_agi_1 import compute_scores_arc_agi_1
ALL_TASKS['arc_agi_1'] = compute_scores_arc_agi_1

def get_after_think(text):
    parts = text.split("\n</think>\n\n", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text

def load_config(config_file_path):
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("Error: The configuration file is empty.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found. Please check the path.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{config_file_path}':\n{e}")
        return None
    except ValueError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An unknown error occurred while loading the configuration file: {e}")
        return None

def evaluate_task(config):
    eval_input_path = config.get("eval_input_path")
    details_path = config.get("details_path")
    task_name = config.get("task_name")

    if not all([eval_input_path, details_path, task_name]):
        print("Error: One or more required parameters ('eval_input_path', 'details_path', 'task_name') are missing in the YAML configuration file.")
        return

    if task_name not in ALL_TASKS:
        print(f"Error: Invalid value '{task_name}' for 'task_name'. It must be one of the following: {list(ALL_TASKS.keys())}")
        return

    print("\n--- Evaluation Configuration Information ---")
    print(f"Model Output File Path: {eval_input_path}")
    print(f"Results Details Path: {details_path}")
    print(f"Task Name: {task_name}")
    print("--------------------\n")

    os.makedirs(os.path.dirname(details_path), exist_ok=True)

    try:
        with open(eval_input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: The file '{eval_input_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file '{eval_input_path}'.")
        return

    for item in tqdm(data, desc=f"Processing {task_name}"):
        temp = get_after_think(item["gen"][0])
        item["gen"][0] = temp

    acc = ALL_TASKS[task_name](data, details_path)
    print(f"Task: {task_name}, Accuracy: {acc}")
    print("Evaluation complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs using a YAML configuration."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config:
        evaluate_task(config)

if __name__ == "__main__":
    main()
