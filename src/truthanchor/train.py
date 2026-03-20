from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from truthanchor.utils.mapper import MLPScoreMapper
from truthanchor.utils.metrics import eval_metrics
from truthanchor.utils.uncertainty_measures.cue import CUECorrector, find_optimal_w


def train_mappers(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    rank_weight=1.0,
    use_cue=False,
    texts_train=None,
    texts_val=None,
    cue_epochs=3,
):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    texts_train = None if texts_train is None else np.asarray(texts_train)
    texts_val = None if texts_val is None else np.asarray(texts_val)

    if X_val is None or y_val is None:
        if use_cue and texts_train is not None:
            X_fit, X_val, y_fit, y_val, texts_fit, texts_val = train_test_split(
                X_train,
                y_train,
                texts_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train,
            )
        else:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train,
            )
            texts_fit = texts_train
    else:
        X_fit = X_train
        y_fit = y_train
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        texts_fit = texts_train

    mapper = MLPScoreMapper(
        hidden_dim=32,
        num_layers=3,
        dropout=0.1,
        lr=0.01,
        epochs=1000,
        batch_size=128,
        rank_weight=rank_weight,
        patience=100,
        val_size=0.2,
        num_bins_ece=10,
        normalize=True,
        seed=42,
        verbose=False,
    )
    mapper.fit_with_validation(X_fit, y_fit, X_val, y_val)

    mapped_scores = mapper.predict_proba(X_val)
    auroc_mapped, ece_mapped, prr_mapped = eval_metrics(y_val, mapped_scores)
    auroc_orig, ece_orig, prr_orig = eval_metrics(y_val, np.asarray(X_val).reshape(-1))

    report = {
        "labels": np.asarray(y_val),
        "raw_scores": np.asarray(X_val).reshape(-1),
        "mapped_scores": mapped_scores,
        "auroc_orig": auroc_orig,
        "ece_orig": ece_orig,
        "prr_orig": prr_orig,
        "auroc_mapped": auroc_mapped,
        "ece_mapped": ece_mapped,
        "prr_mapped": prr_mapped,
    }
    corrector = None

    if use_cue:
        if texts_train is None or texts_val is None:
            raise ValueError("texts_train and texts_val are required when use_cue=True.")
        corrector = CUECorrector(epochs=cue_epochs)
        corrector.fit(list(texts_fit), y_fit)
        cue_val = corrector.predict_proba(list(texts_val))
        scaler = MinMaxScaler()
        scaler.fit(X_fit)
        U_norm_val = scaler.transform(X_val).reshape(-1)
        cue_w = find_optimal_w(U_norm_val, cue_val, y_val)
        cue_scores = cue_w * U_norm_val + (1 - cue_w) * cue_val
        auroc_cue, ece_cue, prr_cue = eval_metrics(y_val, cue_scores)
        report.update(
            {
                "cue_scores": cue_scores,
                "cue_w": cue_w,
                "cue_scaler": scaler,
                "auroc_cue": auroc_cue,
                "ece_cue": ece_cue,
                "prr_cue": prr_cue,
            }
        )

    return mapper, corrector, report
