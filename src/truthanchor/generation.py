from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from truthanchor.utils.datasets import parse_answer, prepare_dataset
from truthanchor.utils.io import save_jsonl
from truthanchor.utils.modeling import load_causal_lm
from truthanchor.utils.paths import artifact_dir
from truthanchor.utils.uncertainty_measures.internal_score import CoEScore, VarianceScore


class Inference:
    def __init__(
        self,
        model,
        tokenizer,
        dataset_name,
        prompts,
        ref_ans,
        out_dir: str | Path | None = None,
        max_tokens=128,
        num_samples=20,
        batch_size=1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.prompts = prompts
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.ref_ans = ref_ans
        self.max_tokens = max_tokens
        self.out_dir = Path(out_dir) if out_dir else None

    def extract_internal(self, full_hidden_states, logits, sequences, input_len, internal_scores=False):
        stacked_logits = torch.stack(logits, dim=1)
        log_probs = F.log_softmax(stacked_logits, dim=-1)
        probs = log_probs.exp()
        gen_seq_len = stacked_logits.shape[1]
        gen_tokens = sequences[:, input_len : input_len + gen_seq_len]
        eos_token_id = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            gen_mask = torch.ones_like(gen_tokens, dtype=torch.bool)
        else:
            if isinstance(eos_token_id, int):
                is_eos = gen_tokens.eq(eos_token_id)
            else:
                eos_ids = torch.tensor(eos_token_id, device=gen_tokens.device)
                is_eos = (gen_tokens.unsqueeze(-1) == eos_ids).any(dim=-1)
            eos_rank = is_eos.cumsum(dim=1)
            gen_mask = (eos_rank == 0) | ((eos_rank == 1) & is_eos)
        valid_token_counts = gen_mask.sum(dim=1).clamp_min(1)
        hidden_states = torch.cat(
            [h[-1] if i > 0 else h[-1][:, -1, :].unsqueeze(1) for i, h in enumerate(full_hidden_states)],
            dim=1,
        )
        masked_embeddings = hidden_states * gen_mask.unsqueeze(-1)
        mean_embedding = masked_embeddings.sum(dim=1) / valid_token_counts.unsqueeze(-1)
        token_log_probs = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(~gen_mask, 0.0)
        seq_log_prob = token_log_probs.sum(dim=1)
        avg_token_log_prob = seq_log_prob / valid_token_counts
        ppl = torch.exp(-avg_token_log_prob)
        avg_maxprob = probs.max(dim=-1).values.masked_fill(~gen_mask, 0.0).sum(dim=1) / valid_token_counts
        avg_entropy = (-(probs * log_probs).sum(dim=-1).masked_fill(~gen_mask, 0.0).sum(dim=1) / valid_token_counts)
        scaled_probs = F.log_softmax(stacked_logits / 0.7, dim=-1).exp()
        avg_tempscale = scaled_probs.max(dim=-1).values.masked_fill(~gen_mask, 0.0).sum(dim=1) / valid_token_counts
        avg_energy = (-0.7 * torch.logsumexp(stacked_logits / 0.7, dim=-1)).masked_fill(~gen_mask, 0.0).sum(dim=1) / valid_token_counts
        output_scores = {
            "likelihoods": seq_log_prob,
            "maxprobs": avg_maxprob,
            "ppls": ppl,
            "entropies": avg_entropy,
            "tempscales": avg_tempscale,
            "energies": avg_energy,
        }
        if not internal_scores:
            return mean_embedding, output_scores
        coe_scorer = CoEScore(full_hidden_states)
        var_scorer = VarianceScore(full_hidden_states)
        return mean_embedding, output_scores, {
            "coe_c": coe_scorer.compute_CoE_C(),
            "coe_r": coe_scorer.compute_CoE_R(),
            "circ_var": var_scorer.circ_variance(),
        }

    def prepare_input(self, prompts):
        batch_messages = [[{"role": "user", "content": f"You are a helpful assistant, providing accurate and concise information without overthinking. {prompt}"}] for prompt in prompts]
        return self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True,
        ).to(self.model.device)

    def generate(self, model_input, greedy=True, num_samples=20):
        with torch.inference_mode():
            if greedy:
                return self.model.generate(
                    **model_input,
                    max_new_tokens=self.max_tokens,
                    max_length=None,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_logits=True,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                )
            return self.model.generate(
                **model_input,
                max_new_tokens=self.max_tokens,
                max_length=None,
                do_sample=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_logits=True,
                top_p=0.9,
                top_k=50,
                temperature=1.0,
                num_return_sequences=num_samples,
            )

    def data_inference(self):
        tensors = {
            "embeddings": [],
            "likelihoods": [],
            "maxprobs": [],
            "ppls": [],
            "entropies": [],
            "tempscales": [],
            "energies": [],
            "coe_c": [],
            "coe_r": [],
            "circ_var": [],
        }
        generations = []
        for idx in tqdm(range(0, len(self.prompts), self.batch_size), total=(len(self.prompts) + self.batch_size - 1) // self.batch_size, desc="Generating responses"):
            batch_prompts = self.prompts[idx : idx + self.batch_size]
            batch_refs = self.ref_ans[idx : idx + self.batch_size]
            batch_size_actual = len(batch_prompts)
            model_input = self.prepare_input(batch_prompts)
            greedy_output = self.generate(model_input, greedy=True)
            greedy_ids = greedy_output.sequences[:, model_input.input_ids.shape[1] :]
            answer_txts = self.tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
            samples_output = self.generate(model_input, greedy=False, num_samples=self.num_samples)
            samples_ids = samples_output.sequences[:, model_input.input_ids.shape[1] :]
            samples_txts = self.tokenizer.batch_decode(samples_ids, skip_special_tokens=True)
            greedy_embedding, greedy_output_scores, greedy_internal_scores = self.extract_internal(
                greedy_output.hidden_states,
                greedy_output.logits,
                greedy_output.sequences,
                model_input["input_ids"].shape[1],
                internal_scores=True,
            )
            samples_embedding, samples_output_scores = self.extract_internal(
                samples_output.hidden_states,
                samples_output.logits,
                samples_output.sequences,
                model_input["input_ids"].shape[1],
                internal_scores=False,
            )
            greedy_embedding = greedy_embedding.view(batch_size_actual, 1, greedy_embedding.shape[-1])
            samples_embedding = samples_embedding.view(batch_size_actual, self.num_samples, greedy_embedding.shape[-1])
            batch_metrics = {
                "embeddings": torch.cat([greedy_embedding, samples_embedding], dim=1),
                "likelihoods": torch.cat([greedy_output_scores["likelihoods"].view(batch_size_actual, 1), samples_output_scores["likelihoods"].view(batch_size_actual, self.num_samples)], dim=1),
                "maxprobs": torch.cat([greedy_output_scores["maxprobs"].view(batch_size_actual, 1), samples_output_scores["maxprobs"].view(batch_size_actual, self.num_samples)], dim=1),
                "ppls": torch.cat([greedy_output_scores["ppls"].view(batch_size_actual, 1), samples_output_scores["ppls"].view(batch_size_actual, self.num_samples)], dim=1),
                "entropies": torch.cat([greedy_output_scores["entropies"].view(batch_size_actual, 1), samples_output_scores["entropies"].view(batch_size_actual, self.num_samples)], dim=1),
                "tempscales": torch.cat([greedy_output_scores["tempscales"].view(batch_size_actual, 1), samples_output_scores["tempscales"].view(batch_size_actual, self.num_samples)], dim=1),
                "energies": torch.cat([greedy_output_scores["energies"].view(batch_size_actual, 1), samples_output_scores["energies"].view(batch_size_actual, self.num_samples)], dim=1),
                "coe_c": torch.as_tensor(greedy_internal_scores["coe_c"]).reshape(batch_size_actual, 1),
                "coe_r": torch.as_tensor(greedy_internal_scores["coe_r"]).reshape(batch_size_actual, 1),
                "circ_var": torch.as_tensor(greedy_internal_scores["circ_var"]).reshape(batch_size_actual, 1),
            }
            for i, answer in enumerate(answer_txts):
                parsed = parse_answer(answer, self.dataset_name)
                if parsed is None:
                    continue
                sample_start = self.num_samples * i
                sample_end = sample_start + self.num_samples
                generations.append(
                    {
                        "prompt": batch_prompts[i],
                        "ref_ans": batch_refs[i],
                        "greedy_response": answer_txts[i],
                        "greedy_id": greedy_ids[i].detach().cpu().numpy().tolist(),
                        "samples": samples_txts[sample_start:sample_end],
                        "samples_ids": samples_ids[sample_start:sample_end].detach().cpu().numpy().tolist(),
                    }
                )
                for key, value in batch_metrics.items():
                    tensors[key].append(value[i].detach().cpu())
            if idx == 0 and answer_txts:
                print(f"Prompt: {batch_prompts[0]}\nAnswer: {answer_txts[0]}")
            torch.cuda.empty_cache()

        if not generations:
            raise RuntimeError("No valid generations were produced; all parsed answers were filtered out.")

        arrays = {}
        for key, values in tensors.items():
            stacked = torch.stack(values).float().numpy() if key not in {"coe_c", "coe_r", "circ_var"} else np.stack([np.asarray(v, dtype=np.float32) for v in values])
            arrays[key] = stacked
        print(f"{len(generations)}/{len(self.prompts)} answered.")
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            save_jsonl(self.out_dir / f"responses_{self.dataset_name}.jsonl", generations)
            np.savez_compressed(self.out_dir / "generation_results.npz", **arrays)
            print("Saved results successfully.")
        return generations


def generate_responses(model_name, dataset_name, max_new_tokens=128, data_portion=1.0, num_samples=5, save=True, output_root="outputs", batch_size=1):
    np.random.seed(0)
    dataset, formatter = prepare_dataset(dataset_name)
    prompts, ref_ans = formatter(dataset_name, dataset)
    data_len = int(data_portion * len(ref_ans))
    prompts = prompts[:data_len]
    ref_ans = ref_ans[:data_len]
    model, tokenizer = load_causal_lm(model_name)
    out_dir = artifact_dir(output_root, dataset_name, model_name) if save else None
    inference = Inference(model, tokenizer, dataset_name, prompts, ref_ans, out_dir=out_dir, max_tokens=max_new_tokens, num_samples=num_samples, batch_size=batch_size)
    return inference.data_inference()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="sciq")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--save", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generate_responses(
        model_name=args.model,
        dataset_name=args.dataset_name,
        max_new_tokens=args.max_new_tokens,
        data_portion=args.data_portion,
        num_samples=args.num_samples,
        save=args.save,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )
