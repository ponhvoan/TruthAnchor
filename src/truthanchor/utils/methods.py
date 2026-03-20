METHOD_INFO = {
    "maxprobs": ("maxprob", False),
    "ppls": ("ppl", True),
    "entropies": ("entropy", True),
    "energies": ("energy", True),
    "p_true": ("p_true", False),
    "verb": ("verb", False),
    "coe_c": ("coe_c", False),
    "coe_r": ("coe_r", False),
    "eigenscores": ("eigenscore", True),
    "semantic_entropies": ("semantic_entropy", True),
}

METHODS_PLOT = {
    "maxprobs": "MSP",
    "ppls": "Perplexity",
    "entropies": "Entropy",
    "energies": "Energy",
    "p_true": "P(True)",
    "verb": "VC",
    "coe_c": "CoE-C",
    "coe_r": "CoE-R",
    "eigenscores": "Eigenscore",
    "semantic_entropies": "SE",
}
