"""
BMEN-499 AlphaFold -- K-Fold Cross-Validation Test: Level 1 (LENIENT)
-----------------------------------------------------------------------
Strictness Level : 1 / 5  --  LENIENT
K                : 5
Threshold        : disorder_content >= 0.10  (very permissive)
Min Region Length: 1 aa
Calibration      : None
Metrics          : Accuracy, simple hit-rate per fold
Split Strategy   : Random shuffle, no stratification

Purpose:
    Absolute baseline. Nearly everything that has ANY disorder content
    passes. Used to establish ceiling-level "easy" performance so that
    tighter thresholds can be compared against it.

Output: kfold_L1_lenient_output.txt
"""

import json
import random
import math
import os
from pathlib import Path

SEED        = 42
K           = 5
THRESHOLD   = 0.10   # very lenient disorder cutoff
MIN_REGION  = 1      # accept any region length
LEVEL_LABEL = "L1 LENIENT"

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


def get_region_count(p: dict, min_len: int) -> int:
    count = 0
    for r in p.get("regions", []):
        if isinstance(r, dict):
            length = r.get("end", 0) - r.get("start", 0) + 1
            if length >= min_len:
                count += 1
    return count


def label(p: dict, threshold: float, min_len: int) -> int:
    """1 = disordered, 0 = ordered."""
    score = get_disorder_score(p)
    regions = get_region_count(p, min_len)
    return 1 if (score >= threshold or regions > 0) else 0


def split_k_folds(data: list, k: int, seed: int) -> list:
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    fold_size = len(indices) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else len(indices)
        folds.append(indices[start:end])
    return folds


def evaluate_fold(proteins, train_idx, test_idx, threshold, min_len):
    # "Model" = mean disorder score of training set as the decision boundary
    train_scores = [get_disorder_score(proteins[i]) for i in train_idx]
    mean_train   = sum(train_scores) / len(train_scores) if train_scores else threshold

    # For L1 lenient: use the simpler of threshold vs mean_train (whichever is lower)
    effective_threshold = min(threshold, mean_train * 1.2)

    tp = fp = tn = fn = 0
    for i in test_idx:
        p        = proteins[i]
        true_lbl = label(p, threshold, min_len)
        score    = get_disorder_score(p)
        pred_lbl = 1 if score >= effective_threshold else 0

        if true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else: fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "effective_threshold": round(effective_threshold, 4),
        "n_test": total,
    }


def mean_std(values):
    n    = len(values)
    mu   = sum(values) / n
    var  = sum((x - mu) ** 2 for x in values) / n
    return mu, math.sqrt(var)


def run_kfold(disprot_path: str):
    proteins = load_disprot(disprot_path)
    print(f"[INFO] Loaded {len(proteins):,} proteins")

    folds   = split_k_folds(proteins, K, SEED)
    results = []

    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"thr={res['effective_threshold']:.4f}  n={res['n_test']}")

    return results, proteins


def write_output(results, proteins, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Fold Validation: {LEVEL_LABEL}")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Dataset: {len(proteins):,} DisProt proteins")
    lines.append(f"  Seed: {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 1 -- LENIENT")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.10  (very permissive)")
    lines.append("  Min region length  : 1 aa  (any region counts)")
    lines.append("  Calibration        : None")
    lines.append("  Split strategy     : Random shuffle, no stratification")
    lines.append("  Decision rule      : score >= min(0.10, 1.2 * mean_train)")
    lines.append("")

    accs  = [r["accuracy"]  for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'Thr':>8} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'N':>7}")
    lines.append("  " + "-" * 66)
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} "
            f"{r['effective_threshold']:>8.4f} "
            f"{r['tp']:>6} {r['fp']:>6} {r['tn']:>6} {r['fn']:>6} {r['n_test']:>7}"
        )

    mu_acc,  sd_acc  = mean_std(accs)
    mu_prec, sd_prec = mean_std(precs)
    mu_rec,  sd_rec  = mean_std(recs)
    mu_f1,   sd_f1   = mean_std(f1s)

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mu_acc:.4f}  (+/- {sd_acc:.4f})")
    lines.append(f"  Mean Precision : {mu_prec:.4f}  (+/- {sd_prec:.4f})")
    lines.append(f"  Mean Recall    : {mu_rec:.4f}  (+/- {sd_rec:.4f})")
    lines.append(f"  Mean F1        : {mu_f1:.4f}  (+/- {sd_f1:.4f})")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L1 Lenient is the most permissive configuration.")
    lines.append("  A low threshold (0.10) means nearly all proteins with any")
    lines.append("  measured disorder are labelled positive. Recall is expected")
    lines.append("  to be very high but precision may be lower due to many FPs.")
    lines.append("  Use this as the upper-bound recall reference.")
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
    out_path   = os.path.join(script_dir, "kfold_L1_lenient_output.txt")

    results, proteins = run_kfold(disprot_path)
    write_output(results, proteins, out_path)