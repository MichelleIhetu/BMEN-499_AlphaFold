"""
BMEN-499 AlphaFold -- K-Fold Cross-Validation Test: Level 5 (SEVERE)
----------------------------------------------------------------------
Strictness Level : 5 / 5  --  SEVERE
K                : 5
Threshold        : disorder_content >= 0.85  (only fully/nearly fully disordered)
Min Region Length: 30 aa
Calibration      : Platt-style sigmoid calibration (no sklearn)
Metrics          : Accuracy, Precision, Recall, F1, MCC, AUROC (approx),
                   Brier score, Expected Calibration Error (ECE),
                   McNemar-style fold agreement test, fold variance
Split Strategy   : Stratified by disorder quartile + Pfam presence flag
Extra Checks     : Fold agreement (McNemar chi-squared), ECE < 0.05 pass/fail,
                   Statistical significance report (p-value approximation)

Purpose:
    Maximum strictness. Only fully or near-fully disordered proteins
    with long validated IDRs (>= 30 aa) qualify as positive. Platt
    sigmoid calibration ensures predicted probabilities are meaningful.
    McNemar test checks whether fold performance differences are
    statistically significant. ECE (Expected Calibration Error) measures
    how well predicted probabilities match empirical positive rates.

Output: kfold_L5_severe_output.txt
"""

import json
import random
import math
import os
from pathlib import Path

SEED           = 42
K              = 5
THRESHOLD      = 0.85
MIN_REGION     = 30
LEVEL_LABEL    = "L5 SEVERE"
IMBALANCE_WARN = 0.10
ECE_PASS       = 0.05
N_ECE_BINS     = 10

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


def has_pfam(p: dict) -> bool:
    return len(p.get("features", {}).get("pfam", [])) > 0


def get_max_region_length(p: dict) -> int:
    lengths = [r.get("end", 0) - r.get("start", 0) + 1
               for r in p.get("regions", []) if isinstance(r, dict)]
    return max(lengths) if lengths else 0


def seq_length(p: dict) -> int:
    return len(p.get("sequence", ""))


def label(p: dict, threshold: float, min_len: int) -> int:
    score      = get_disorder_score(p)
    max_region = get_max_region_length(p)
    return 1 if (score >= threshold and max_region >= min_len) else 0


def stratified_folds_with_pfam(proteins, k, seed):
    """Stratify by disorder quartile AND Pfam flag."""
    random.seed(seed)
    scores = [get_disorder_score(p) for p in proteins]
    sorted_idx = sorted(range(len(proteins)), key=lambda i: scores[i])

    score_rank = [0] * len(proteins)
    for rank, i in enumerate(sorted_idx):
        score_rank[i] = rank

    n = len(proteins)
    bins = {}
    for i in range(n):
        sq  = min(int(score_rank[i] / n * 4), 3)
        pfm = 1 if has_pfam(proteins[i]) else 0
        key = (sq, pfm)
        bins.setdefault(key, []).append(i)

    folds = [[] for _ in range(k)]
    for key, group in bins.items():
        random.shuffle(group)
        for pos, idx in enumerate(group):
            folds[pos % k].append(idx)
    return folds


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def platt_calibrate(train_pairs, test_scores):
    """
    Platt scaling: fit a + b * score to training (label, score) pairs.
    Uses a simple gradient descent on logistic loss.
    Returns calibrated probabilities for test_scores.
    """
    if not train_pairs or len(train_pairs) < 2:
        return [sigmoid(s * 2 - 1) for s in test_scores]

    a, b = 0.0, 1.0
    lr   = 0.01
    for _ in range(200):
        da = db = 0.0
        for s, lbl in train_pairs:
            p     = sigmoid(a + b * s)
            err   = p - lbl
            da   += err
            db   += err * s
        a -= lr * da / len(train_pairs)
        b -= lr * db / len(train_pairs)

    return [sigmoid(a + b * s) for s in test_scores]


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: mean |confidence - accuracy| weighted by bin size."""
    bins = [[] for _ in range(n_bins)]
    for prob, lbl in zip(probs, labels):
        b = min(int(prob * n_bins), n_bins - 1)
        bins[b].append((prob, lbl))
    ece = 0.0
    n   = len(probs)
    for bin_items in bins:
        if bin_items:
            mean_conf = sum(x[0] for x in bin_items) / len(bin_items)
            mean_acc  = sum(x[1] for x in bin_items) / len(bin_items)
            ece      += len(bin_items) / n * abs(mean_conf - mean_acc)
    return ece


def approx_auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    count = sum(1 for p in pos for n in neg if p > n) + \
            0.5 * sum(1 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))


def brier_score(probs, labels):
    if not probs:
        return 1.0
    return sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(probs)


def mcnemar_p_approx(b, c):
    """
    McNemar chi-squared approximation.
    b = cases where model1 right, model2 wrong
    c = cases where model1 wrong, model2 right
    Returns approximate p-value.
    """
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Approximate p-value from chi-squared with 1 df
    # Using approximation: p ≈ exp(-chi2/2) for chi2 < 6
    if chi2 < 0.001:
        return 1.0
    elif chi2 > 10.0:
        return 0.001
    else:
        return math.exp(-chi2 / 2)


def mean_std(values):
    n  = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in values) / n)
    return mu, sd


def evaluate_fold(proteins, train_idx, test_idx, threshold, min_len):
    train_pairs = [(get_disorder_score(proteins[i]),
                    label(proteins[i], threshold, min_len)) for i in train_idx]
    test_scores = [get_disorder_score(proteins[i]) for i in test_idx]
    test_labels = [label(proteins[i], threshold, min_len) for i in test_idx]

    cal_probs   = platt_calibrate(train_pairs, test_scores)
    auroc       = approx_auroc(cal_probs, test_labels)
    brier       = brier_score(cal_probs, test_labels)
    ece         = expected_calibration_error(cal_probs, test_labels, N_ECE_BINS)

    pred_labels = [1 if cp >= 0.5 else 0 for cp in cal_probs]

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

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "pred_labels": pred_labels,
        "true_labels": test_labels,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "auroc": round(auroc, 4), "brier": round(brier, 4),
        "ece": round(ece, 4),
        "ece_pass": ece < ECE_PASS,
        "pos_rate": round(pos_rate, 4),
        "imbalance_warn": pos_rate < IMBALANCE_WARN,
        "n_test": total,
    }


def run_kfold(disprot_path):
    proteins = load_disprot(disprot_path)
    print(f"[INFO] Loaded {len(proteins):,} proteins")

    folds   = stratified_folds_with_pfam(proteins, K, SEED)
    results = []

    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        ece_tag   = "PASS" if res["ece_pass"] else "FAIL"
        warn_tag  = "  [!] IMBALANCE" if res["imbalance_warn"] else ""
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"mcc={res['mcc']:.4f}  auroc={res['auroc']:.4f}  "
              f"ece={res['ece']:.4f}[{ece_tag}]  pos={res['pos_rate']:.3f}{warn_tag}")

    return results, proteins


def write_output(results, proteins, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Fold Validation: {LEVEL_LABEL}")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Dataset: {len(proteins):,} DisProt proteins  |  Seed: {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 5 -- SEVERE")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.85  (fully/near-fully disordered only)")
    lines.append("  Min region length  : 30 aa (only long experimentally verified IDRs)")
    lines.append("  Calibration        : Platt sigmoid scaling")
    lines.append("  Split strategy     : Stratified by disorder quartile + Pfam flag")
    lines.append("  Extra metrics      : AUROC, Brier, ECE, McNemar fold agreement")
    lines.append("  ECE pass threshold : < 0.05")
    lines.append("  Imbalance warning  : pos_rate < 0.10")
    lines.append("")

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'AUROC':>7} {'Brier':>7} {'ECE':>7} {'ECE?':>5} {'PosRate':>8}")
    lines.append("  " + "-" * 78)
    for i, r in enumerate(results, 1):
        ece_tag = "PASS" if r["ece_pass"] else "FAIL"
        lines.append(
            f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
            f"{r['auroc']:>7.4f} {r['brier']:>7.4f} {r['ece']:>7.4f} "
            f"{ece_tag:>5} {r['pos_rate']:>8.4f}"
        )

    # McNemar pairwise fold comparison
    lines.append("")
    lines.append("MCNEMAR PAIRWISE FOLD AGREEMENT")
    lines.append("-" * 70)
    lines.append("  Tests whether pairs of folds have statistically different error")
    lines.append("  patterns. p < 0.05 = significant difference between folds.")
    lines.append("")
    lines.append(f"  {'Fold Pair':<12} {'b':>6} {'c':>6} {'chi2_approx':>12} {'p_approx':>10} {'sig':>5}")
    lines.append("  " + "-" * 52)
    for i in range(K):
        for j in range(i + 1, K):
            pred_i = results[i]["pred_labels"]
            pred_j = results[j]["pred_labels"]
            true_i = results[i]["true_labels"]
            true_j = results[j]["true_labels"]
            # Use min length (folds differ in size)
            min_n  = min(len(pred_i), len(pred_j))
            b = sum(1 for k in range(min_n) if pred_i[k] != true_i[k] and pred_j[k] == true_j[k])
            c = sum(1 for k in range(min_n) if pred_i[k] == true_i[k] and pred_j[k] != true_j[k])
            p = mcnemar_p_approx(b, c)
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
            sig  = "*" if p < 0.05 else "ns"
            lines.append(
                f"  Fold {i+1} vs {j+1}   {b:>6} {c:>6} {chi2:>12.4f} {p:>10.4f} {sig:>5}"
            )

    accs   = [r["accuracy"]  for r in results]
    precs  = [r["precision"] for r in results]
    recs   = [r["recall"]    for r in results]
    f1s    = [r["f1"]        for r in results]
    mccs   = [r["mcc"]       for r in results]
    aurocs = [r["auroc"]     for r in results]
    briers = [r["brier"]     for r in results]
    eces   = [r["ece"]       for r in results]

    mu_acc,   sd_acc   = mean_std(accs)
    mu_prec,  sd_prec  = mean_std(precs)
    mu_rec,   sd_rec   = mean_std(recs)
    mu_f1,    sd_f1    = mean_std(f1s)
    mu_mcc,   sd_mcc   = mean_std(mccs)
    mu_auroc, sd_auroc = mean_std(aurocs)
    mu_brier, sd_brier = mean_std(briers)
    mu_ece,   sd_ece   = mean_std(eces)

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
    lines.append(f"  Mean ECE       : {mu_ece:.4f}  (+/- {sd_ece:.4f})")

    ece_passes = sum(1 for r in results if r["ece_pass"])
    lines.append(f"  ECE PASS folds : {ece_passes}/{K}")
    imbalance_folds = sum(1 for r in results if r["imbalance_warn"])
    if imbalance_folds:
        lines.append(f"  [!] IMBALANCE WARNING: {imbalance_folds} fold(s) had pos_rate < {IMBALANCE_WARN}")

    lines.append("")
    lines.append("OVERALL PASS/FAIL VERDICT")
    lines.append("-" * 70)
    pass_conditions = [
        ("Mean F1 >= 0.50",        mu_f1    >= 0.50),
        ("Mean MCC >= 0.30",       mu_mcc   >= 0.30),
        ("Mean AUROC >= 0.70",     mu_auroc >= 0.70),
        ("Mean Brier <= 0.20",     mu_brier <= 0.20),
        ("Mean ECE <= 0.05",       mu_ece   <= 0.05),
        ("All folds completed",    True),
    ]
    all_pass = all(p for _, p in pass_conditions)
    for cond, passed in pass_conditions:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}]  {cond}")
    lines.append("")
    lines.append(f"  OVERALL: {'PASS -- model meets severe strictness criteria' if all_pass else 'FAIL -- one or more criteria not met'}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L5 Severe is the most demanding configuration. Positive labels")
    lines.append("  are reserved for proteins with disorder content >= 0.85 AND a")
    lines.append("  region >= 30 aa. This means very few proteins pass -- expect")
    lines.append("  low recall and high precision. The McNemar test checks whether")
    lines.append("  fold variability is random or systematic. ECE < 0.05 means the")
    lines.append("  predicted probabilities are well-matched to actual positive rates.")
    lines.append("  This level is appropriate for high-stakes annotation tasks where")
    lines.append("  false positives are more costly than false negatives.")
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
    out_path   = os.path.join(script_dir, "kfold_L5_severe_output.txt")

    results, proteins = run_kfold(disprot_path)
    write_output(results, proteins, out_path)