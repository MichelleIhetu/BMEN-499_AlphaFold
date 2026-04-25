"""
BMEN-499 AlphaFold -- K-Fold Cross-Validation Test: Level 3 (MODERATE)
------------------------------------------------------------------------
Strictness Level : 3 / 5  --  MODERATE
K                : 5
Threshold        : disorder_content >= 0.50  (canonical IDR cutoff)
Min Region Length: 10 aa
Calibration      : Z-score normalization per fold
Metrics          : Accuracy, Precision, Recall, F1, MCC, fold variance
Split Strategy   : Stratified by disorder score quartile

Purpose:
    The canonical 0.50 threshold. This is the standard cutoff used in
    most IDR prediction benchmarks. Stratified splitting ensures each
    fold has balanced disorder score distributions. Z-score calibration
    normalises prediction scores per fold.

Output: kfold_L3_moderate_output.txt
"""

import json
import random
import math
import os
from pathlib import Path

SEED        = 42
K           = 5
THRESHOLD   = 0.50
MIN_REGION  = 10
LEVEL_LABEL = "L3 MODERATE"

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


def get_valid_regions(p: dict, min_len: int) -> list:
    out = []
    for r in p.get("regions", []):
        if isinstance(r, dict):
            length = r.get("end", 0) - r.get("start", 0) + 1
            if length >= min_len:
                out.append(r)
    return out


def label(p: dict, threshold: float, min_len: int) -> int:
    score   = get_disorder_score(p)
    regions = get_valid_regions(p, min_len)
    return 1 if (score >= threshold and len(regions) > 0) else 0


def stratified_k_folds(proteins, k, seed):
    """Stratify by disorder score quartile so each fold has balanced distribution."""
    random.seed(seed)
    scores  = [(i, get_disorder_score(p)) for i, p in enumerate(proteins)]
    sorted_ = sorted(scores, key=lambda x: x[1])

    # Assign quartile labels
    n = len(sorted_)
    quartile_groups = [[] for _ in range(4)]
    for rank, (idx, _) in enumerate(sorted_):
        q = min(int(rank / n * 4), 3)
        quartile_groups[q].append(idx)

    # Shuffle within each quartile then interleave into folds
    folds = [[] for _ in range(k)]
    for group in quartile_groups:
        random.shuffle(group)
        for i, idx in enumerate(group):
            folds[i % k].append(idx)

    return folds


def z_score_calibrate(score, mean, std):
    if std < 1e-9:
        return score
    return (score - mean) / std


def evaluate_fold(proteins, train_idx, test_idx, threshold, min_len):
    train_scores = [get_disorder_score(proteins[i]) for i in train_idx]
    mu    = sum(train_scores) / len(train_scores) if train_scores else 0.5
    var   = sum((x - mu) ** 2 for x in train_scores) / len(train_scores) if train_scores else 1.0
    sigma = math.sqrt(var) if var > 0 else 1.0

    # Calibrated threshold in z-score space
    cal_thr_z = z_score_calibrate(threshold, mu, sigma)

    tp = fp = tn = fn = 0
    for i in test_idx:
        p        = proteins[i]
        true_lbl = label(p, threshold, min_len)
        score    = get_disorder_score(p)
        score_z  = z_score_calibrate(score, mu, sigma)
        pred_lbl = 1 if score_z >= cal_thr_z else 0

        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Matthews Correlation Coefficient
    denom_mcc = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = (tp * tn - fp * fn) / denom_mcc

    pos_rate = sum(1 for i in test_idx if label(proteins[i], threshold, min_len) == 1) / total if total else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "train_mu": round(mu, 4), "train_sigma": round(sigma, 4),
        "pos_rate": round(pos_rate, 4),
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

    folds   = stratified_k_folds(proteins, K, SEED)
    results = []

    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"mcc={res['mcc']:.4f}  pos_rate={res['pos_rate']:.3f}")

    return results, proteins


def write_output(results, proteins, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Fold Validation: {LEVEL_LABEL}")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Dataset: {len(proteins):,} DisProt proteins  |  Seed: {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 3 -- MODERATE")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.50  (canonical IDR benchmark cutoff)")
    lines.append("  Min region length  : 10 aa (standard minimum IDR length)")
    lines.append("  Calibration        : Z-score normalization per training fold")
    lines.append("  Split strategy     : Stratified by disorder score quartile")
    lines.append("  Extra metric       : MCC (Matthews Correlation Coefficient)")
    lines.append("")

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'mu':>7} {'sigma':>7} {'PosRate':>8}")
    lines.append("  " + "-" * 68)
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
            f"{r['train_mu']:>7.4f} {r['train_sigma']:>7.4f} {r['pos_rate']:>8.4f}"
        )

    accs  = [r["accuracy"]  for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]
    mccs  = [r["mcc"]       for r in results]

    mu_acc,  sd_acc  = mean_std(accs)
    mu_prec, sd_prec = mean_std(precs)
    mu_rec,  sd_rec  = mean_std(recs)
    mu_f1,   sd_f1   = mean_std(f1s)
    mu_mcc,  sd_mcc  = mean_std(mccs)

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mu_acc:.4f}  (+/- {sd_acc:.4f})")
    lines.append(f"  Mean Precision : {mu_prec:.4f}  (+/- {sd_prec:.4f})")
    lines.append(f"  Mean Recall    : {mu_rec:.4f}  (+/- {sd_rec:.4f})")
    lines.append(f"  Mean F1        : {mu_f1:.4f}  (+/- {sd_f1:.4f})")
    lines.append(f"  Mean MCC       : {mu_mcc:.4f}  (+/- {sd_mcc:.4f})")
    lines.append("")
    lines.append("CONFUSION MATRIX TOTALS (across all folds)")
    lines.append("-" * 70)
    all_tp = sum(r["tp"] for r in results)
    all_fp = sum(r["fp"] for r in results)
    all_tn = sum(r["tn"] for r in results)
    all_fn = sum(r["fn"] for r in results)
    lines.append(f"  TP={all_tp:>7,}  FP={all_fp:>7,}  TN={all_tn:>7,}  FN={all_fn:>7,}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L3 Moderate uses the canonical 0.50 threshold adopted by")
    lines.append("  most IDR benchmarks (e.g., CAID). Stratified splitting ensures")
    lines.append("  each fold has a realistic disorder score distribution.")
    lines.append("  MCC is the most balanced metric for imbalanced datasets.")
    lines.append("  Use this as the primary reference fold configuration.")
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
    out_path   = os.path.join(script_dir, "kfold_L3_moderate_output.txt")

    results, proteins = run_kfold(disprot_path)
    write_output(results, proteins, out_path)