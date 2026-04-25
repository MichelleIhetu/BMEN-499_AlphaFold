"""
BMEN-499 AlphaFold -- K-Fold Cross-Validation Test: Level 2 (MILD)
--------------------------------------------------------------------
Strictness Level : 2 / 5  --  MILD
K                : 5
Threshold        : disorder_content >= 0.30  (standard liberal cutoff)
Min Region Length: 5 aa
Calibration      : Mean-shift normalization on training fold
Metrics          : Accuracy, Precision, Recall, F1, fold variance
Split Strategy   : Random shuffle, no stratification

Purpose:
    Mild strictness. The 0.30 threshold is a commonly used liberal
    boundary for borderline disordered proteins. Requires regions to
    be at least 5 aa to filter trivially short annotations. Light
    calibration via training-fold mean shift applied.

Output: kfold_L2_mild_output.txt
"""

import json
import random
import math
import os
from pathlib import Path

SEED        = 42
K           = 5
THRESHOLD   = 0.30
MIN_REGION  = 5
LEVEL_LABEL = "L2 MILD"

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
    score   = get_disorder_score(p)
    regions = get_region_count(p, min_len)
    return 1 if (score >= threshold and regions > 0) else 0


def split_k_folds(data, k, seed):
    random.seed(seed)
    idx = list(range(len(data)))
    random.shuffle(idx)
    fold_size = len(idx) // k
    folds = []
    for i in range(k):
        s = i * fold_size
        e = s + fold_size if i < k - 1 else len(idx)
        folds.append(idx[s:e])
    return folds


def calibrate_threshold(train_proteins, base_threshold):
    """Mean-shift: adjust threshold by offset of training mean from 0.30."""
    scores     = [get_disorder_score(p) for p in train_proteins]
    train_mean = sum(scores) / len(scores) if scores else base_threshold
    offset     = train_mean - 0.30
    return max(0.05, base_threshold + offset * 0.5)


def evaluate_fold(proteins, train_idx, test_idx, threshold, min_len):
    train_proteins = [proteins[i] for i in train_idx]
    cal_threshold  = calibrate_threshold(train_proteins, threshold)

    tp = fp = tn = fn = 0
    for i in test_idx:
        p        = proteins[i]
        true_lbl = label(p, threshold, min_len)
        score    = get_disorder_score(p)
        pred_lbl = 1 if score >= cal_threshold else 0

        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Fold disorder stats
    test_scores  = [get_disorder_score(proteins[i]) for i in test_idx]
    mean_score   = sum(test_scores) / len(test_scores) if test_scores else 0.0
    pos_count    = sum(1 for p in [proteins[i] for i in test_idx] if label(p, threshold, min_len) == 1)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "cal_threshold": round(cal_threshold, 4),
        "mean_test_score": round(mean_score, 4),
        "pos_rate": round(pos_count / total, 4) if total else 0,
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

    folds   = split_k_folds(proteins, K, SEED)
    results = []

    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"cal_thr={res['cal_threshold']:.4f}  pos_rate={res['pos_rate']:.3f}")

    return results, proteins


def write_output(results, proteins, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Fold Validation: {LEVEL_LABEL}")
    lines.append(f"  K={K}  |  Base Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Dataset: {len(proteins):,} DisProt proteins  |  Seed: {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 2 -- MILD")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.30  (liberal but standard cutoff)")
    lines.append("  Min region length  : 5 aa  (filters trivially short regions)")
    lines.append("  Calibration        : Mean-shift on training fold")
    lines.append("  Split strategy     : Random shuffle, no stratification")
    lines.append("  Decision rule      : score >= calibrated_threshold")
    lines.append("")

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'CalThr':>8} {'PosRate':>8} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    lines.append("  " + "-" * 68)
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} "
            f"{r['cal_threshold']:>8.4f} {r['pos_rate']:>8.4f} "
            f"{r['tp']:>6} {r['fp']:>6} {r['tn']:>6} {r['fn']:>6}"
        )

    accs  = [r["accuracy"]  for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]

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

    # Calibration drift across folds
    thresholds = [r["cal_threshold"] for r in results]
    mu_thr, sd_thr = mean_std(thresholds)
    lines.append("CALIBRATION ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Base threshold       : {THRESHOLD:.4f}")
    lines.append(f"  Mean cal threshold   : {mu_thr:.4f}  (+/- {sd_thr:.4f})")
    lines.append(f"  Max cal drift        : {max(abs(t - THRESHOLD) for t in thresholds):.4f}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L2 Mild applies the commonly accepted 0.30 disorder boundary.")
    lines.append("  Mean-shift calibration adjusts for training fold composition,")
    lines.append("  which helps when disorder prevalence varies across splits.")
    lines.append("  Compare F1 here vs L1 to see the cost of tightening the cutoff.")
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
    out_path   = os.path.join(script_dir, "kfold_L2_mild_output.txt")

    results, proteins = run_kfold(disprot_path)
    write_output(results, proteins, out_path)