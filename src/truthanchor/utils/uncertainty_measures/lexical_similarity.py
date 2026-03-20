import itertools
import re

import numpy as np
from nltk.translate.meteor_score import single_meteor_score


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text):
    if text is None:
        return []
    return _TOKEN_RE.findall(text.lower().strip())


def lexical_similarity(responses):
    responses = [r for r in responses if isinstance(r, str) and r.strip()]
    n = len(responses)
    if n == 0:
        return np.nan
    if n == 1:
        return 1.0
    tokenized = [simple_tokenize(r) for r in responses]
    sims = []
    for i, j in itertools.combinations(range(n), 2):
        s_ij = single_meteor_score(tokenized[i], tokenized[j])
        s_ji = single_meteor_score(tokenized[j], tokenized[i])
        sims.append(0.5 * (s_ij + s_ji))
    return float(np.mean(sims))
