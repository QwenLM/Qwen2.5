import json
import re
from collections import defaultdict
import numpy as np

def parse_model_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        json_match = re.findall(r"```(?:json|python)\s*(.*?)\s*```", output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match[-1])
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in the ```json``` block")
                return None
        else:
            array_match = re.findall(r"(\[\[(?:[\d,\[\]\s\n]*)\]\])", output, re.DOTALL)
            if array_match:
                try:
                    return json.loads(array_match[-1])
                except json.JSONDecodeError:
                    print("Error: Invalid JSON format in the last array-like structure")
                    return None
            else:
                print("Error: No valid JSON array found in the output")
                return None


def solution_score(predicted, ground_truth):
    if not predicted or not ground_truth:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0


def compute_scores_arc_agi_1(jobs, cache_path):
    taskid2score = defaultdict(list)
    for job in jobs:
        assert (
            len(job.get("gen", [])) == 1
        ), "Each job should contain exactly one generation output"
        answer = job.get("answer")
        pred_raw = job["gen"][0]
        parsed_pred = parse_model_output(pred_raw)
        if parsed_pred is not None:
            solu_score = solution_score(parsed_pred, answer)
        else:
            solu_score = 0.0
        job.update({"acc": solu_score})
        taskid2score[job["task_id"]].append(solu_score)
    save_cache(jobs, cache_path)
    assert len(taskid2score) == 400, 'The ARC-AGI-1 dataset should have 400 tasks'
    return sum(np.mean(x) for x in taskid2score.values()) / len(taskid2score) if jobs else 0.0


def save_cache(jobs, cache_path):
    with open(cache_path, "w", encoding="utf-8") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()
