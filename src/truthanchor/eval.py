from __future__ import annotations

import numpy as np
from truthanchor.utils.metrics import eval_metrics


def evaluate_saved_mappers(mapper, X_test, y_test, corrector=None, texts_test=None, cue_w=None, cue_scaler=None):
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    mapped_scores = mapper.predict_proba(X_test)
    auroc_mapped, ece_mapped, prr_mapped = eval_metrics(y_test, mapped_scores)
    auroc_orig, ece_orig, prr_orig = eval_metrics(y_test, X_test.reshape(-1))
    report = {
        "labels": y_test,
        "raw_scores": X_test.reshape(-1),
        "mapped_scores": mapped_scores,
        "auroc_orig": auroc_orig,
        "ece_orig": ece_orig,
        "prr_orig": prr_orig,
        "auroc_mapped": auroc_mapped,
        "ece_mapped": ece_mapped,
        "prr_mapped": prr_mapped,
    }
    if corrector is not None:
        if texts_test is None or cue_w is None or cue_scaler is None:
            raise ValueError("texts_test, cue_w, and cue_scaler are required for CUE evaluation.")
        U_norm_test = cue_scaler.transform(X_test).reshape(-1)
        cue_probs = corrector.predict_proba(list(texts_test))
        cue_scores = cue_w * U_norm_test + (1 - cue_w) * cue_probs
        auroc_cue, ece_cue, prr_cue = eval_metrics(y_test, cue_scores)
        report.update(
            {
                "cue_scores": cue_scores,
                "auroc_cue": auroc_cue,
                "ece_cue": ece_cue,
                "prr_cue": prr_cue,
            }
        )
    return report
