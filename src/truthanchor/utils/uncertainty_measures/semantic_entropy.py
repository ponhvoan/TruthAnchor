"""Semantic entropy estimator."""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        return None


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        prediction = torch.argmax(F.softmax(outputs.logits, dim=1)).cpu().item()
        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)
        return prediction


def get_semantic_ids(strings_list, model=None, strict_entailment=False, example=None):
    model = model or EntailmentDeberta()

    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        if strict_entailment:
            return implication_1 == 2 and implication_2 == 2
        implications = [implication_1, implication_2]
        return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] != -1:
            continue
        semantic_set_ids[i] = next_id
        for j in range(i + 1, len(strings_list)):
            if are_equivalent(string1, strings_list[j]):
                semantic_set_ids[j] = next_id
        next_id += 1
    return semantic_set_ids


def neglog_by_id(semantic_ids, log_likelihoods, agg="sum_normalized"):
    del agg
    unique_ids = sorted(list(set(semantic_ids)))
    neg_log_likelihood_per_semantic_id = []
    likelihoods = np.exp(log_likelihoods)
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_likelihoods = [likelihoods[i] for i in id_indices]
        id_likelihoods_sum = float(sum(id_likelihoods)) / float(sum(likelihoods) + 1e-9)
        neg_log_likelihood_per_semantic_id.append(-np.log(id_likelihoods_sum + 1e-9))
    return neg_log_likelihood_per_semantic_id, len(unique_ids)
