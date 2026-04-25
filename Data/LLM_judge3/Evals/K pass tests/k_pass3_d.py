"""
BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test Level 4 (STRICT)
----------------------------------------------------------------
File    : k_pass3_d.py
Output  : k_pass3_d_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

Strictness Level : 4 / 5  --  STRICT
K                : 5
Threshold        : disorder_content >= 0.70  (high-confidence IDR only)
Min Region Length: 20 aa
Calibration      : Isotonic-style rank calibration
Split Strategy   : Stratified by disorder + protein length quartile
Metrics          : Accuracy, Precision, Recall, F1, MCC, AUROC, Brier
Extra Checks     : Imbalance warning if pos_rate < 0.15
"""

import json
import re
import os
import math
import random
from pathlib import Path
from collections import Counter

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH    = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
DISPROT_PATH = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\Baseline\DisProt_ProteinData.json"
OUT_PATH     = os.path.join(SCRIPT_DIR, "k_pass3_d_output.txt")

SEED            = 42
K               = 5
THRESHOLD       = 0.70
MIN_REGION      = 20
LEVEL_LABEL     = "L4 STRICT"
IMBALANCE_WARN  = 0.15

def mean(lst):   return sum(lst) / len(lst) if lst else 0.0
def std(lst):
    mu = mean(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / len(lst)) if lst else 0.0

def load_disprot(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    return raw

def load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"={6,}", text)
    preds  = []
    q_pat  = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat  = re.compile(r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)", re.DOTALL)
    for block in blocks:
        q_m = q_pat.search(block)
        a_m = a_pat.search(block)
        if q_m and a_m:
            preds.append({
                "q_num":    int(q_m.group(1)),
                "question": q_m.group(2).strip(),
                "answer":   re.sub(r"\s+", " ", a_m.group(1)).strip(),
            })
    preds.sort(key=lambda x: x["q_num"])
    return preds

def get_disorder_score(p):
    v = p.get("disorder_content_pure") or p.get("disorder_content_obs")
    return float(v) if v is not None else 0.0

def get_max_region_len(p):
    lengths = [r.get("end", 0) - r.get("start", 0) + 1
               for r in p.get("regions", []) if isinstance(r, dict)]
    return max(lengths) if lengths else 0

def seq_length(p):
    return len(p.get("sequence", ""))

def protein_label(p, threshold, min_len):
    return 1 if (get_disorder_score(p) >= threshold and get_max_region_len(p) >= min_len) else 0

def answer_quality_score(answer):
    a = answer.lower()
    signals = [
        "plddt", "disprot", "disorder", "threshold", "pfam",
        "proline", "glycine", "idr", "idp", "alphafold",
        "experimentally", "validated", "sequence", "residue",
        "sliding window", "calibrat", "intrinsically disordered",
        "gray zone", "pyrrolidine", "conformational", "molecular recognition",
        "morf", "conditional", "hub protein",
    ]
    hits = sum(1 for s in signals if s in a)
    return min(1.0, hits / len(signals) * 2)

def stratified_folds(proteins, k, seed):
    random.seed(seed)
    scores  = [get_disorder_score(p) for p in proteins]
    lengths = [seq_length(p) for p in proteins]
    n = len(proteins)
    sorted_score_idx  = sorted(range(n), key=lambda i: scores[i])
    sorted_length_idx = sorted(range(n), key=lambda i: lengths[i])
    score_rank  = [0] * n
    length_rank = [0] * n
    for rank, i in enumerate(sorted_score_idx):  score_rank[i]  = rank
    for rank, i in enumerate(sorted_length_idx): length_rank[i] = rank
    bins = {}
    for i in range(n):
        sq  = min(int(score_rank[i]  / n * 4), 3)
        lq  = min(int(length_rank[i] / n * 4), 3)
        bins.setdefault((sq, lq), []).append(i)
    folds = [[] for _ in range(k)]
    for group in bins.values():
        random.shuffle(group)
        for pos, idx in enumerate(group):
            folds[pos % k].append(idx)
    return folds

def rank_calibrate(train_pairs, test_scores):
    if not train_pairs:
        return test_scores
    sorted_train = sorted(train_pairs, key=lambda x: x[0])
    n_bins    = 10
    bin_size  = max(1, len(sorted_train) // n_bins)
    bins_list = []
    for b in range(n_bins):
        chunk = sorted_train[b * bin_size: b * bin_size + bin_size if b < n_bins - 1 else len(sorted_train)]
        if chunk:
            mid  = sum(x[0] for x in chunk) / len(chunk)
            frac = sum(x[1] for x in chunk) / len(chunk)
            bins_list.append((mid, frac))
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

def evaluate_fold(proteins, predictions, train_idx, test_idx, threshold, min_len):
    n_preds     = len(predictions)
    train_pairs = []
    for i in train_idx:
        score = get_disorder_score(proteins[i])
        lbl   = protein_label(proteins[i], threshold, min_len)
        pred_idx = i % n_preds
        aq    = answer_quality_score(predictions[pred_idx]["answer"])
        combined = score * 0.55 + aq * 0.45
        train_pairs.append((combined, lbl))

    test_scores = []
    test_labels = []
    for i in test_idx:
        score    = get_disorder_score(proteins[i])
        pred_idx = i % n_preds
        aq       = answer_quality_score(predictions[pred_idx]["answer"])
        combined = score * 0.55 + aq * 0.45
        test_scores.append(combined)
        test_labels.append(protein_label(proteins[i], threshold, min_len))

    cal_probs   = rank_calibrate(train_pairs, test_scores)
    pred_labels = [1 if cp >= 0.5 else 0 for cp in cal_probs]
    auroc       = approx_auroc(cal_probs, test_labels)
    brier       = brier_score(cal_probs, test_labels)

    tp = fp = tn = fn = 0
    for true_lbl, pred_lbl in zip(test_labels, pred_labels):
        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total    = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall   = tp / (tp + fn) if (tp + fn) else 0
    f1       = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    denom_mcc = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc      = (tp * tn - fp * fn) / denom_mcc
    pos_rate = sum(test_labels) / total if total else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "auroc": round(auroc, 4), "brier": round(brier, 4),
        "pos_rate": round(pos_rate, 4),
        "imbalance_warn": pos_rate < IMBALANCE_WARN,
        "n_test": total,
    }

def run_kfold(proteins, predictions):
    folds   = stratified_folds(proteins, K, SEED)
    results = []
    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, predictions, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        warn = "  [!]" if res["imbalance_warn"] else ""
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"mcc={res['mcc']:.4f}  auroc={res['auroc']:.4f}  pos={res['pos_rate']:.3f}{warn}")
    return results

def write_output(results, proteins, predictions, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test {LEVEL_LABEL}")
    lines.append(f"  Script   : k_pass3_d.py")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Proteins : {len(proteins):,}  |  Predictions : {len(predictions)}")
    lines.append(f"  Seed     : {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 4 -- STRICT")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.70  (high confidence IDRs only)")
    lines.append("  Min region length  : 20 aa")
    lines.append("  Calibration        : Isotonic-style rank calibration")
    lines.append("  Split strategy     : Stratified by disorder + length quartile")
    lines.append("  Extra metrics      : AUROC, Brier score")
    lines.append("  Imbalance warning  : pos_rate < 0.15")
    lines.append("  Prediction signal  : 55% disorder score + 45% LLM3 answer quality")
    lines.append("")
    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'AUROC':>7} {'Brier':>7} {'Pos':>6} {'Warn':>5}")
    lines.append("  " + "-" * 72)
    for i, r in enumerate(results, 1):
        warn = " [!]" if r["imbalance_warn"] else "    "
        lines.append(f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                     f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
                     f"{r['auroc']:>7.4f} {r['brier']:>7.4f} {r['pos_rate']:>6.3f}{warn}")

    accs   = [r["accuracy"]  for r in results]
    f1s    = [r["f1"]        for r in results]
    mccs   = [r["mcc"]       for r in results]
    aurocs = [r["auroc"]     for r in results]
    briers = [r["brier"]     for r in results]
    imb    = sum(1 for r in results if r["imbalance_warn"])

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mean(accs):.4f}  (+/- {std(accs):.4f})")
    lines.append(f"  Mean F1        : {mean(f1s):.4f}  (+/- {std(f1s):.4f})")
    lines.append(f"  Mean MCC       : {mean(mccs):.4f}  (+/- {std(mccs):.4f})")
    lines.append(f"  Mean AUROC     : {mean(aurocs):.4f}  (+/- {std(aurocs):.4f})")
    lines.append(f"  Mean Brier     : {mean(briers):.4f}  (+/- {std(briers):.4f})")
    if imb:
        lines.append(f"  [!] IMBALANCE WARNING: {imb} fold(s) had pos_rate < {IMBALANCE_WARN}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L4 Strict filters to only high-confidence IDPs (>=0.70).")
    lines.append("  Expect high precision but lower recall. AUROC > 0.85 indicates")
    lines.append("  good rank ordering even if the absolute threshold is conservative.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    print(f"[INFO] Loading DisProt from:\n       {DISPROT_PATH}\n")
    proteins    = load_disprot(DISPROT_PATH)
    print(f"[INFO] Loaded {len(proteins):,} proteins")
    print(f"[INFO] Loading LLM3 predictions from:\n       {LLM3_PATH}\n")
    predictions = load_predictions(LLM3_PATH)
    print(f"[INFO] Loaded {len(predictions)} predictions")
    print(f"\n[INFO] Running {K}-fold cross-validation ({LEVEL_LABEL})...\n")
    results = run_kfold(proteins, predictions)
    write_output(results, proteins, predictions, OUT_PATH)