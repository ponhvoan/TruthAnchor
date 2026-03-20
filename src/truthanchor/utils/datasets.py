from __future__ import annotations

import io
import json
import os
import pickle
import re
from collections.abc import Mapping, Sequence

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Value, load_dataset
from sklearn.utils import resample

from truthanchor.utils.resources import load_prompt_template


def binarize(batch):
    labs = [str(x).strip().upper() for x in batch["label"]]
    y = [0 if l.startswith("REFUTES") else 1 for l in labs]
    return {"label": y}


def format_prompt(dataset_name, dataset):
    all_prompts = []
    all_gt = []
    if dataset_name == "trivia":
        for row in dataset:
            query = row["question"] + ' Be concise, and output only the final answer. If you do not know the answer, simply say "Unknown".\n'
            all_prompts.append(query)
            all_gt.append(row["answer"]["value"])
    elif dataset_name == "sciq":
        for row in dataset:
            query = row["question"] + ' Be concise, and output only the final answer. If you do not know the answer, simply say "Unknown".\n'
            all_prompts.append(query)
            all_gt.append(row["correct_answer"])
    elif dataset_name == "popqa":
        for row in dataset:
            query = row["question"] + ' Be concise, and output only the final answer. If you do not know the answer, simply say "Unknown".\n'
            all_prompts.append(query)
            all_gt.append(json.loads(row["possible_answers"]))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return all_prompts, all_gt


def prepare_dataset(dataset_name):
    if dataset_name == "trivia":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(1000))
    elif dataset_name == "sciq":
        dataset = load_dataset("allenai/sciq", split="test")
    elif dataset_name == "popqa":
        dataset = load_dataset("akariasai/PopQA", split="test")
        dataset = dataset.shuffle(seed=42).select(range(1000))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset, format_prompt


def parse_answer(answer_txt, dataset_name):
    if dataset_name in ["true_false", "halueval", "fever"]:
        lower = answer_txt.lower()
        if "true" in lower:
            return 1
        if "false" in lower:
            return 0
        return None
    if dataset_name in ["mmlu", "medmcqa"]:
        mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
        match = re.search(r"(?i)Answer\s*:\s*([A-D])", answer_txt)
        return mapping[match.group(1).lower()] if match else None
    if dataset_name == "commonsenseqa":
        mapping = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        match = re.search(r"(?i)Answer\s*:\s*([A-E])", answer_txt)
        return mapping[match.group(1).lower()] if match else None
    if dataset_name == "gsm":
        answer_prefix = {
            "en": "Answer",
            "bn": "উত্তর",
            "de": "Antwort",
            "es": "Respuesta",
            "fr": "Réponse",
            "ja": "答え",
            "ru": "Ответ",
            "sw": "Jibu",
            "te": "సమాధానం",
            "th": "คำตอบ",
            "zh": "答案",
        }
        normalized = answer_txt
        for prefix in answer_prefix.values():
            if prefix in normalized:
                normalized = normalized.split(prefix)[-1].strip()
                break
        else:
            return None
        match = re.search(r"\d+\.?\d*", normalized.replace(",", ""))
        return float(match.group()) if match else None
    if dataset_name == "math":
        pattern = re.compile(r"\\boxed{")
        for match in pattern.finditer(answer_txt):
            start = match.end()
            depth = 1
            idx = start
            while idx < len(answer_txt) and depth > 0:
                if answer_txt[idx] == "{":
                    depth += 1
                elif answer_txt[idx] == "}":
                    depth -= 1
                idx += 1
            if depth == 0:
                return answer_txt[start : idx - 1].replace(" ", "")
        return None
    return answer_txt


def append_answer(labels, gen_ans, gt, dataset_name):
    if dataset_name in ["trivia", "sciq"]:
        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(predictions=[gen_ans], references=[gt])["rougeL"]
        labels.append(1 if rouge_score > 0.7 else 0)
    elif dataset_name == "popqa":
        match_found = any(re.search(rf"\b{re.escape(ref)}\b", gen_ans) for ref in gt)
        labels.append(1 if match_found else 0)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return labels


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def balance_classes(X, y):
    X_class_0 = X[y == 0]
    y_class_0 = y[y == 0]
    X_class_1 = X[y == 1]
    y_class_1 = y[y == 1]
    X_class_0_downsampled, y_class_0_downsampled = resample(
        X_class_0,
        y_class_0,
        replace=False,
        n_samples=len(y_class_1),
        random_state=42,
    )
    return np.vstack((X_class_0_downsampled, X_class_1)), np.hstack((y_class_0_downsampled, y_class_1))


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, Sequence):
        return type(obj)(to_cpu(x) for x in obj)
    return obj


def save_results_to_csv(
    in_dir,
    dataset_name,
    method,
    auroc_orig,
    auroc_clf,
    ece_orig,
    ece_clf,
    prr_orig,
    prr_clf,
    train_size=None,
    corrupt_ratio=None,
    overwrite_same_method=True,
):
    os.makedirs(in_dir, exist_ok=True)
    if (train_size is None) and (corrupt_ratio is None):
        filename = f"{dataset_name}_results.csv"
    elif (train_size is not None) and (corrupt_ratio is None):
        filename = f"{dataset_name}_results_{train_size}.csv"
    elif (train_size is None) and (corrupt_ratio is not None):
        filename = f"{dataset_name}_results_{corrupt_ratio}.csv"
    else:
        filename = f"{dataset_name}_results_{train_size}{corrupt_ratio}.csv"

    csv_path = os.path.join(in_dir, filename)
    row = {
        "auroc_orig": auroc_orig * 100,
        "auroc_clf": auroc_clf * 100,
        "auroc_gain": (auroc_clf - auroc_orig) * 100,
        "ece_orig": ece_orig * 100,
        "ece_clf": ece_clf * 100,
        "ece_gain": (ece_clf - ece_orig) * 100,
        "prr_orig": prr_orig * 100,
        "prr_clf": prr_clf * 100,
        "prr_gain": (prr_clf - prr_orig) * 100,
    }
    new_df = pd.DataFrame([row], index=[method])
    new_df.index.name = "method"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        if overwrite_same_method:
            df.loc[method] = row
        else:
            df = pd.concat([df, new_df], axis=0)
    else:
        df = new_df
    df = df.round(4)
    df.to_csv(csv_path)
    return df
