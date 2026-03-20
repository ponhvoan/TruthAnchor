import torch
import torch.nn.functional as F
from truthanchor.utils.modeling import load_causal_lm


def candidate_first_token_ids(tokenizer, letter="A"):
    variants = (letter, f" {letter}", f"\n{letter}", f"({letter})", f" ({letter})", f"\n({letter})")
    ids = set()
    for variant in variants:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


class PTrueScore:
    def __init__(self, model_name):
        self.model, self.tokenizer = load_causal_lm(model_name)

    @torch.no_grad()
    def compute_p_true(self, prompt, temperature=1.0):
        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        next_logits = logits[0, -1]
        probs = F.softmax(next_logits / temperature, dim=-1)
        a_ids = torch.tensor(candidate_first_token_ids(self.tokenizer, "A"), device=device)
        return probs.index_select(0, a_ids).sum().item()
