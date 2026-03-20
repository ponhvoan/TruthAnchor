import re
import torch
from truthanchor.utils.modeling import load_causal_lm


class VerbScore:
    def __init__(self, model_name):
        self.model, self.tokenizer = load_causal_lm(model_name)

    @torch.no_grad()
    def compute_verb(self, prompt: str, max_new_tokens: int = 10, temperature: float = 0.2, max_tries: int = 10):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for _ in range(max_tries):
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=True,
                temperature=temperature,
            )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            match = re.search(r"(\d+\.\d+|\d+)", output)
            if not match:
                continue
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value
        return 0.0
