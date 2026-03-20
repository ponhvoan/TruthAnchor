import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(y_true, scores):
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def compute_ece(confidences, labels, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)
    for i in range(num_bins):
        bin_start = bin_boundaries[i]
        bin_end = bin_boundaries[i + 1]
        if i == num_bins - 1:
            bin_mask = (confidences >= bin_start) & (confidences <= bin_end)
        else:
            bin_mask = (confidences >= bin_start) & (confidences < bin_end)
        if not np.any(bin_mask):
            continue
        bin_confidences = confidences[bin_mask]
        bin_labels = labels[bin_mask]
        bin_accuracy = np.mean(bin_labels)
        bin_avg_confidence = np.mean(bin_confidences)
        ece += (np.sum(bin_mask) / total_samples) * abs(bin_accuracy - bin_avg_confidence)
    return ece


def compute_prr(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    m = labels.sum()
    if m == 0 or m == n:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    r_vals = np.arange(n + 1) / n
    cum_errors = np.concatenate(([0], np.cumsum(sorted_labels)))
    error_curve = (m - cum_errors) / n
    base_error = m / n
    random_curve = base_error * (1 - r_vals)
    oracle_curve = np.maximum(base_error - r_vals, 0.0)
    ar_uns = np.trapezoid(random_curve - error_curve, r_vals)
    ar_orc = np.trapezoid(random_curve - oracle_curve, r_vals)
    return ar_uns / ar_orc


def eval_metrics(labels, raw_scores, num_bins=10):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(raw_scores, dtype=float)
    auroc = safe_auroc(labels, scores)
    if not np.all((scores >= 0) & (scores <= 1)):
        if scores.size == 0:
            confidences = scores
        else:
            lo = np.min(scores)
            hi = np.max(scores)
            confidences = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    else:
        confidences = scores
    ece = compute_ece(confidences, labels, num_bins=num_bins)
    prr = compute_prr(labels, scores)
    return auroc, ece, prr
