"""
BMEN-499 AlphaFold -- K-Fold Cross-Validation Test: Level 4 (STRICT)
----------------------------------------------------------------------
Strictness Level : 4 / 5  --  STRICT
K                : 5
Threshold        : disorder_content >= 0.70  (high-confidence IDR only)
Min Region Length: 20 aa
Calibration      : Isotonic-style rank calibration (no sklearn required)
Metrics          : Accuracy, Precision, Recall, F1, MCC, AUROC (approx),
                   Brier score, fold variance, positive rate enforcement
Split Strategy   : Stratified + protein length quartile balancing
Extra Checks     : Fold class-imbalance warning if pos_rate < 0.15

Purpose:
    Strict mode. Only proteins with very high disorder content (>= 0.70)
    AND at least one long validated region (>= 20 aa) are labeled positive.
    This removes ambiguous borderline cases entirely. Isotonic-style
    rank calibration re-orders predictions to be monotone relative to
    ground truth frequency within bins.

Output: kfold_L4_strict_output.txt
"""

import json
import random
import math
import os
from pathlib import Path

SEED            = 42
K               = 5
THRESHOLD       = 0.70
MIN_REGION      = 20
LEVEL_LABEL     = "L4 STRICT"
IMBALANCE_WARN  = 0.15   # warn if positive rate below this

# ── helpers ──────────────────────────────────────────────────────────

def load_disprot(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    return raw


def get_disorder_score(p: dict) -> float:
    v = p.get("disorder_content_pure") or p.get("disorder_content_obs")
    return float(v) if v is not None else 0.0


def get_max_region_length(p: dict) -> int:
    lengths = []
    for r in p.get("regions", []):
        if isinstance(r, dict):
            lengths.append(r.get("end", 0) - r.get("start", 0) + 1)
    return max(lengths) if lengths else 0


def seq_length(p: dict) -> int:
    return len(p.get("sequence", ""))


def label(p: dict, threshold: float, min_len: int) -> int:
    score      = get_disorder_score(p)
    max_region = get_max_region_length(p)
    return 1 if (score >= threshold and max_region >= min_len) else 0


def stratified_folds(proteins, k, seed):
    """Stratify by both disorder score AND protein length to balance folds."""
    random.seed(seed)

    # Bin on (disorder_quartile, length_quartile) pair
    scores  = [get_disorder_score(p)    for p in proteins]
    lengths = [seq_length(p)            for p in proteins]

    sorted_score_idx  = sorted(range(len(scores)),  key=lambda i: scores[i])
    sorted_length_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])

    score_rank  = [0] * len(proteins)
    length_rank = [0] * len(proteins)
    for rank, i in enumerate(sorted_score_idx):
        score_rank[i]  = rank
    for rank, i in enumerate(sorted_length_idx):
        length_rank[i] = rank

    n = len(proteins)
    bins = {}
    for i in range(n):
        sq = min(int(score_rank[i]  / n * 4), 3)
        lq = min(int(length_rank[i] / n * 4), 3)
        key = (sq, lq)
        bins.setdefault(key, []).append(i)

    folds = [[] for _ in range(k)]
    for key, group in bins.items():
        random.shuffle(group)
        for pos, idx in enumerate(group):
            folds[pos % k].append(idx)
    return folds


def rank_calibrate(scores_with_labels, test_scores):
    """
    Isotonic-style rank calibration.
    For each test score, find its rank-based calibrated probability
    relative to training distribution.
    """
    if not scores_with_labels:
        return test_scores

    # Build monotone calibration bins (10 bins)
    sorted_train = sorted(scores_with_labels, key=lambda x: x[0])
    n_bins = 10
    bin_size = max(1, len(sorted_train) // n_bins)
    bins_list = []
    for b in range(n_bins):
        start = b * bin_size
        end   = start + bin_size if b < n_bins - 1 else len(sorted_train)
        chunk = sorted_train[start:end]
        if chunk:
            mid_score  = sum(x[0] for x in chunk) / len(chunk)
            pos_frac   = sum(x[1] for x in chunk) / len(chunk)
            bins_list.append((mid_score, pos_frac))

    # For each test score, interpolate calibrated probability
    calibrated = []
    for s in test_scores:
        if s <= bins_list[0][0]:
            calibrated.append(bins_list[0][1])
        elif s >= bins_list[-1][0]:
            calibrated.append(bins_list[-1][1])
        else:
            for j in range(len(bins_list) - 1):
                lo, lo_p = bins_list[j]
                hi, hi_p = bins_list[j + 1]
                if lo <= s <= hi:
                    t = (s - lo) / (hi - lo) if hi != lo else 0
                    calibrated.append(lo_p + t * (hi_p - lo_p))
                    break
            else:
                calibrated.append(bins_list[-1][1])
    return calibrated


def approx_auroc(scores, labels):
    """Wilcoxon-Mann-Whitney AUROC approximation."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    count = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))


def brier_score(probs, labels):
    if not probs:
        return 1.0
    return sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(probs)


def evaluate_fold(proteins, train_idx, test_idx, threshold, min_len):
    train_pairs  = [(get_disorder_score(proteins[i]),
                     label(proteins[i], threshold, min_len)) for i in train_idx]
    test_scores  = [get_disorder_score(proteins[i]) for i in test_idx]
    test_labels  = [label(proteins[i], threshold, min_len) for i in test_idx]

    cal_probs    = rank_calibrate(train_pairs, test_scores)
    auroc        = approx_auroc(cal_probs, test_labels)
    brier        = brier_score(cal_probs, test_labels)

    # Classify using 0.5 on calibrated probability
    pred_labels  = [1 if cp >= 0.5 else 0 for cp in cal_probs]

    tp = fp = tn = fn = 0
    for true_lbl, pred_lbl in zip(test_labels, pred_labels):
        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    denom_mcc = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc       = (tp * tn - fp * fn) / denom_mcc
    pos_rate  = sum(test_labels) / total if total else 0

    imbalance_warn = pos_rate < IMBALANCE_WARN

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "auroc": round(auroc, 4), "brier": round(brier, 4),
        "pos_rate": round(pos_rate, 4),
        "imbalance_warn": imbalance_warn,
        "n_test": total,
    }


def mean_std(values):
    n  = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in values) / n)
    return mu, sd


def run_kfold(disprot_path):
    proteins = load_disprot(disprot_path)
    print(f"[INFO] Loaded {len(proteins):,} proteins")

    folds   = stratified_folds(proteins, K, SEED)
    results = []

    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        warn = "  [!] IMBALANCE WARNING" if res["imbalance_warn"] else ""
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"mcc={res['mcc']:.4f}  auroc={res['auroc']:.4f}  "
              f"pos_rate={res['pos_rate']:.3f}{warn}")

    return results, proteins


def write_output(results, proteins, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Fold Validation: {LEVEL_LABEL}")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Dataset: {len(proteins):,} DisProt proteins  |  Seed: {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 4 -- STRICT")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.70  (high confidence IDRs only)")
    lines.append("  Min region length  : 20 aa (longer validated regions only)")
    lines.append("  Calibration        : Isotonic-style rank calibration")
    lines.append("  Split strategy     : Stratified by disorder + length quartile")
    lines.append("  Extra metrics      : AUROC (approx), Brier score")
    lines.append("  Imbalance warning  : triggered if pos_rate < 0.15")
    lines.append("")

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'AUROC':>8} {'Brier':>7} {'PosRate':>8} {'Warn':>6}")
    lines.append("  " + "-" * 74)
    for i, r in enumerate(results, 1):
        warn = "  [!]" if r["imbalance_warn"] else ""
        lines.append(
            f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
            f"{r['auroc']:>8.4f} {r['brier']:>7.4f} {r['pos_rate']:>8.4f}{warn}"
        )

    accs   = [r["accuracy"]  for r in results]
    precs  = [r["precision"] for r in results]
    recs   = [r["recall"]    for r in results]
    f1s    = [r["f1"]        for r in results]
    mccs   = [r["mcc"]       for r in results]
    aurocs = [r["auroc"]     for r in results]
    briers = [r["brier"]     for r in results]

    mu_acc,   sd_acc   = mean_std(accs)
    mu_prec,  sd_prec  = mean_std(precs)
    mu_rec,   sd_rec   = mean_std(recs)
    mu_f1,    sd_f1    = mean_std(f1s)
    mu_mcc,   sd_mcc   = mean_std(mccs)
    mu_auroc, sd_auroc = mean_std(aurocs)
    mu_brier, sd_brier = mean_std(briers)

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mu_acc:.4f}  (+/- {sd_acc:.4f})")
    lines.append(f"  Mean Precision : {mu_prec:.4f}  (+/- {sd_prec:.4f})")
    lines.append(f"  Mean Recall    : {mu_rec:.4f}  (+/- {sd_rec:.4f})")
    lines.append(f"  Mean F1        : {mu_f1:.4f}  (+/- {sd_f1:.4f})")
    lines.append(f"  Mean MCC       : {mu_mcc:.4f}  (+/- {sd_mcc:.4f})")
    lines.append(f"  Mean AUROC     : {mu_auroc:.4f}  (+/- {sd_auroc:.4f})")
    lines.append(f"  Mean Brier     : {mu_brier:.4f}  (+/- {sd_brier:.4f})")
    lines.append("")
    imbalance_folds = sum(1 for r in results if r["imbalance_warn"])
    if imbalance_folds:
        lines.append(f"  [!] IMBALANCE WARNING: {imbalance_folds} fold(s) had pos_rate < {IMBALANCE_WARN}")
        lines.append("      Consider SMOTE or class-weighting for production use.")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L4 Strict filters to only the most confidently disordered")
    lines.append("  proteins. Expect lower recall (many true IDPs miss the 0.70")
    lines.append("  cutoff) but very high precision. Brier score < 0.10 indicates")
    lines.append("  well-calibrated predictions. AUROC > 0.85 indicates good rank")
    lines.append("  ordering even if absolute threshold is conservative.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    disprot_path = (
        r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina"
        r"\Desktop\MIHETU\AI_Insitute_Work\BMEN 499"
        r"\BMEN-499_AlphaFold\Data\Baseline\DisProt_ProteinData.json"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "kfold_L4_strict_output.txt")

    results, proteins = run_kfold(disprot_path)
    write_output(results, proteins, out_path)