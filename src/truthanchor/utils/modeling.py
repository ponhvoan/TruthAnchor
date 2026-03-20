from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name: str, use_fast: bool | None = None):
    tokenizer_kwargs = {"trust_remote_code": True}
    if use_fast is not None:
        tokenizer_kwargs["use_fast"] = use_fast
    elif "mistral" in model_name.lower():
        tokenizer_kwargs["use_fast"] = False
    else:
        tokenizer_kwargs["use_fast"] = True

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float16

    if "mistral" in model_name.lower():
        from transformers import Mistral3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = Mistral3ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model.eval(), tokenizer
