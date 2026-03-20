from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from truthanchor.utils.datasets import append_answer, parse_answer
from truthanchor.utils.io import load_jsonl, load_npz_dict
from truthanchor.utils.paths import resolve_artifact_dir
from truthanchor.utils.resources import load_prompt_template
from truthanchor.utils.uncertainty_measures.internal_score import InternalScore
from truthanchor.utils.uncertainty_measures.p_true import PTrueScore
from truthanchor.utils.uncertainty_measures.semantic_entropy import EntailmentDeberta, get_semantic_ids, neglog_by_id
from truthanchor.utils.uncertainty_measures.verb_score import VerbScore


SCORE_KEY_MAP = {
    "semantic_entropies": "semantic_entropy",
    "eigenscores": "eigenscore",
    "maxprobs": "maxprob",
    "ppls": "ppl",
    "entropies": "entropy",
    "tempscales": "tempscale",
    "energies": "energy",
    "likelihoods": "likelihood",
    "coe_c": "coe_c",
    "coe_r": "coe_r",
    "circ_var": "circ_var",
    "p_true": "p_true",
    "verb": "verb",
}


def compute_uncertainty_scores(model_name, dataset_name, output_root="outputs"):
    np.random.seed(0)
    in_dir = resolve_artifact_dir(output_root, dataset_name, model_name)
    responses = load_jsonl(in_dir / f"responses_{dataset_name}.jsonl")
    internal_dict = load_npz_dict(in_dir / "generation_results.npz")
    p_true_scorer = PTrueScore(model_name)
    verb_scorer = VerbScore(model_name)
    p_true_prompt_template = load_prompt_template("p_true")
    verb_prompt_template = load_prompt_template("verb")
    uncertainty_scores = {key: [] for key in SCORE_KEY_MAP}
    labels = []
    
    entail_model = EntailmentDeberta()
    for generation, log_liks, embeddings, maxprobs, ppls, entropies, tempscales, energies, coe_c, coe_r, circ_var in tqdm(
        zip(
            responses,
            internal_dict["likelihoods"],
            internal_dict["embeddings"],
            internal_dict["maxprobs"],
            internal_dict["ppls"],
            internal_dict["entropies"],
            internal_dict["tempscales"],
            internal_dict["energies"],
            internal_dict["coe_c"],
            internal_dict["coe_r"],
            internal_dict["circ_var"],
        ),
        total=len(responses),
        desc="Computing uncertainty scores",
    ):
        question = generation["prompt"]
        greedy_ans = generation["greedy_response"]
        samples = generation["samples"]
        ref_ans = generation["ref_ans"]
        qa_pairs = [question + sample for sample in samples]
        semantic_ids = get_semantic_ids(qa_pairs, model=entail_model)
        log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
        neg_log_likelihood_per_semantic_id, num_unique_ids = neglog_by_id(semantic_ids, log_liks_agg, agg="sum_normalized")
        uncertainty_scores["semantic_entropies"].append(sum(neg_log_likelihood_per_semantic_id) / num_unique_ids)
        uncertainty_scores["eigenscores"].append(InternalScore(embeddings[1:]).eigenscore())
        uncertainty_scores["maxprobs"].append(np.mean(maxprobs))
        uncertainty_scores["ppls"].append(np.mean(ppls))
        uncertainty_scores["entropies"].append(np.mean(entropies))
        uncertainty_scores["tempscales"].append(np.mean(tempscales))
        uncertainty_scores["energies"].append(np.mean(energies))
        uncertainty_scores["likelihoods"].append(np.mean(log_liks))
        uncertainty_scores["coe_c"].append(np.mean(coe_c))
        uncertainty_scores["coe_r"].append(np.mean(coe_r))
        uncertainty_scores["circ_var"].append(np.mean(circ_var))
        statement = f"{question} {greedy_ans}"
        uncertainty_scores["p_true"].append(p_true_scorer.compute_p_true(p_true_prompt_template.format(query=statement)))
        uncertainty_scores["verb"].append(verb_scorer.compute_verb(verb_prompt_template.format(question=question, answer=greedy_ans), temperature=0.2, max_tries=10))
        labels = append_answer(labels, parse_answer(greedy_ans, dataset_name), ref_ans, dataset_name)
    save_kwargs = {saved_key: np.asarray(uncertainty_scores[dict_key]) for dict_key, saved_key in SCORE_KEY_MAP.items()}
    save_kwargs["labels"] = np.asarray(labels)
    np.savez_compressed(in_dir / "uncertainty_scores.npz", **save_kwargs)
    print("Saved uncertainty scores successfully.")
    return in_dir / "uncertainty_scores.npz"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trivia")
    parser.add_argument("--output_root", type=str, default="outputs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    compute_uncertainty_scores(args.model, args.dataset_name, output_root=args.output_root)
