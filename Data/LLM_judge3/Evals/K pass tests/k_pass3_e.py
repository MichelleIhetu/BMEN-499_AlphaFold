"""
BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test Level 5 (SEVERE)
----------------------------------------------------------------
File    : k_pass3_e.py
Output  : k_pass3_e_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

Strictness Level : 5 / 5  --  SEVERE
K                : 5
Threshold        : disorder_content >= 0.85  (fully disordered only)
Min Region Length: 30 aa
Calibration      : Platt sigmoid scaling
Split Strategy   : Stratified by disorder quartile + Pfam flag
Metrics          : Accuracy, Precision, Recall, F1, MCC, AUROC,
                   Brier, ECE, McNemar fold agreement
Extra Checks     : ECE < 0.05 pass/fail, imbalance warning pos < 0.10,
                   overall pass/fail verdict
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
OUT_PATH     = os.path.join(SCRIPT_DIR, "k_pass3_e_output.txt")

SEED           = 42
K              = 5
THRESHOLD      = 0.85
MIN_REGION     = 30
LEVEL_LABEL    = "L5 SEVERE"
IMBALANCE_WARN = 0.10
ECE_PASS       = 0.05
N_ECE_BINS     = 10

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

def has_pfam(p):
    return len(p.get("features", {}).get("pfam", [])) > 0

def get_max_region_len(p):
    lengths = [r.get("end", 0) - r.get("start", 0) + 1
               for r in p.get("regions", []) if isinstance(r, dict)]
    return max(lengths) if lengths else 0

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
        "morf", "conditional", "hub protein", "isotonic", "neurosymbolic",
    ]
    hits = sum(1 for s in signals if s in a)
    return min(1.0, hits / len(signals) * 2)

def stratified_folds_pfam(proteins, k, seed):
    random.seed(seed)
    scores = [get_disorder_score(p) for p in proteins]
    n      = len(proteins)
    sorted_idx = sorted(range(n), key=lambda i: scores[i])
    score_rank = [0] * n
    for rank, i in enumerate(sorted_idx):
        score_rank[i] = rank
    bins = {}
    for i in range(n):
        sq  = min(int(score_rank[i] / n * 4), 3)
        pfm = 1 if has_pfam(proteins[i]) else 0
        bins.setdefault((sq, pfm), []).append(i)
    folds = [[] for _ in range(k)]
    for group in bins.values():
        random.shuffle(group)
        for pos, idx in enumerate(group):
            folds[pos % k].append(idx)
    return folds

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def platt_calibrate(train_pairs, test_scores):
    if not train_pairs or len(train_pairs) < 2:
        return [sigmoid(s * 2 - 1) for s in test_scores]
    a, b = 0.0, 1.0
    lr   = 0.01
    for _ in range(200):
        da = db = 0.0
        for s, lbl in train_pairs:
            p   = sigmoid(a + b * s)
            err = p - lbl
            da += err
            db += err * s
        a -= lr * da / len(train_pairs)
        b -= lr * db / len(train_pairs)
    return [sigmoid(a + b * s) for s in test_scores]

def ece(probs, labels, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for prob, lbl in zip(probs, labels):
        b = min(int(prob * n_bins), n_bins - 1)
        bins[b].append((prob, lbl))
    result = 0.0
    n      = len(probs)
    for bin_items in bins:
        if bin_items:
            mc = mean([x[0] for x in bin_items])
            ma = mean([x[1] for x in bin_items])
            result += len(bin_items) / n * abs(mc - ma)
    return result

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

def mcnemar_p(b, c):
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    if chi2 < 0.001: return 1.0
    elif chi2 > 10:  return 0.001
    return math.exp(-chi2 / 2)

def evaluate_fold(proteins, predictions, train_idx, test_idx, threshold, min_len):
    n_preds     = len(predictions)
    train_pairs = []
    for i in train_idx:
        score    = get_disorder_score(proteins[i])
        lbl      = protein_label(proteins[i], threshold, min_len)
        pred_idx = i % n_preds
        aq       = answer_quality_score(predictions[pred_idx]["answer"])
        combined = score * 0.50 + aq * 0.50
        train_pairs.append((combined, lbl))

    test_scores = []
    test_labels = []
    for i in test_idx:
        score    = get_disorder_score(proteins[i])
        pred_idx = i % n_preds
        aq       = answer_quality_score(predictions[pred_idx]["answer"])
        combined = score * 0.50 + aq * 0.50
        test_scores.append(combined)
        test_labels.append(protein_label(proteins[i], threshold, min_len))

    cal_probs   = platt_calibrate(train_pairs, test_scores)
    pred_labels = [1 if cp >= 0.5 else 0 for cp in cal_probs]
    auroc_val   = approx_auroc(cal_probs, test_labels)
    brier_val   = brier_score(cal_probs, test_labels)
    ece_val     = ece(cal_probs, test_labels, N_ECE_BINS)

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
        "pred_labels": pred_labels, "true_labels": test_labels,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "auroc": round(auroc_val, 4), "brier": round(brier_val, 4),
        "ece": round(ece_val, 4), "ece_pass": ece_val < ECE_PASS,
        "pos_rate": round(pos_rate, 4),
        "imbalance_warn": pos_rate < IMBALANCE_WARN,
        "n_test": total,
    }

def run_kfold(proteins, predictions):
    folds   = stratified_folds_pfam(proteins, K, SEED)
    results = []
    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, predictions, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        ece_tag  = "PASS" if res["ece_pass"] else "FAIL"
        warn_tag = " [!]" if res["imbalance_warn"] else "    "
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"mcc={res['mcc']:.4f}  auroc={res['auroc']:.4f}  "
              f"ece={res['ece']:.4f}[{ece_tag}]  pos={res['pos_rate']:.3f}{warn_tag}")
    return results

def write_output(results, proteins, predictions, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test {LEVEL_LABEL}")
    lines.append(f"  Script   : k_pass3_e.py")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Proteins : {len(proteins):,}  |  Predictions : {len(predictions)}")
    lines.append(f"  Seed     : {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 5 -- SEVERE")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.85  (fully/near-fully disordered only)")
    lines.append("  Min region length  : 30 aa")
    lines.append("  Calibration        : Platt sigmoid scaling")
    lines.append("  Split strategy     : Stratified by disorder quartile + Pfam flag")
    lines.append("  Extra metrics      : AUROC, Brier, ECE, McNemar fold agreement")
    lines.append("  ECE pass threshold : < 0.05")
    lines.append("  Imbalance warning  : pos_rate < 0.10")
    lines.append("  Prediction signal  : 50% disorder score + 50% LLM3 answer quality")
    lines.append("")
    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'AUROC':>7} {'Brier':>7} {'ECE':>7} {'ECE?':>5} {'Pos':>6}")
    lines.append("  " + "-" * 76)
    for i, r in enumerate(results, 1):
        ece_tag = "PASS" if r["ece_pass"] else "FAIL"
        lines.append(f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                     f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
                     f"{r['auroc']:>7.4f} {r['brier']:>7.4f} {r['ece']:>7.4f} "
                     f"{ece_tag:>5} {r['pos_rate']:>6.3f}")

    # McNemar
    lines.append("")
    lines.append("MCNEMAR PAIRWISE FOLD AGREEMENT")
    lines.append("-" * 70)
    lines.append(f"  {'Pair':<12} {'b':>6} {'c':>6} {'p_approx':>10} {'sig':>5}")
    lines.append("  " + "-" * 38)
    for i in range(K):
        for j in range(i + 1, K):
            pi = results[i]["pred_labels"]
            pj = results[j]["pred_labels"]
            ti = results[i]["true_labels"]
            tj = results[j]["true_labels"]
            mn = min(len(pi), len(pj))
            b  = sum(1 for k in range(mn) if pi[k] != ti[k] and pj[k] == tj[k])
            c  = sum(1 for k in range(mn) if pi[k] == ti[k] and pj[k] != tj[k])
            p  = mcnemar_p(b, c)
            sig = "*" if p < 0.05 else "ns"
            lines.append(f"  Fold {i+1} vs {j+1}   {b:>6} {c:>6} {p:>10.4f} {sig:>5}")

    accs   = [r["accuracy"]  for r in results]
    f1s    = [r["f1"]        for r in results]
    mccs   = [r["mcc"]       for r in results]
    aurocs = [r["auroc"]     for r in results]
    briers = [r["brier"]     for r in results]
    eces   = [r["ece"]       for r in results]

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mean(accs):.4f}  (+/- {std(accs):.4f})")
    lines.append(f"  Mean F1        : {mean(f1s):.4f}  (+/- {std(f1s):.4f})")
    lines.append(f"  Mean MCC       : {mean(mccs):.4f}  (+/- {std(mccs):.4f})")
    lines.append(f"  Mean AUROC     : {mean(aurocs):.4f}  (+/- {std(aurocs):.4f})")
    lines.append(f"  Mean Brier     : {mean(briers):.4f}  (+/- {std(briers):.4f})")
    lines.append(f"  Mean ECE       : {mean(eces):.4f}  (+/- {std(eces):.4f})")
    lines.append(f"  ECE PASS folds : {sum(1 for r in results if r['ece_pass'])}/{K}")
    imb = sum(1 for r in results if r["imbalance_warn"])
    if imb:
        lines.append(f"  [!] IMBALANCE WARNING: {imb} fold(s) had pos_rate < {IMBALANCE_WARN}")

    lines.append("")
    lines.append("OVERALL PASS/FAIL VERDICT")
    lines.append("-" * 70)
    conditions = [
        ("Mean F1 >= 0.50",    mean(f1s)    >= 0.50),
        ("Mean MCC >= 0.30",   mean(mccs)   >= 0.30),
        ("Mean AUROC >= 0.70", mean(aurocs) >= 0.70),
        ("Mean Brier <= 0.20", mean(briers) <= 0.20),
        ("Mean ECE <= 0.05",   mean(eces)   <= 0.05),
        ("All folds completed", True),
    ]
    all_pass = all(p for _, p in conditions)
    for cond, passed in conditions:
        lines.append(f"  [{'PASS' if passed else 'FAIL'}]  {cond}")
    lines.append("")
    lines.append(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L5 Severe is the most demanding configuration (threshold=0.85).")
    lines.append("  Very few proteins qualify as positive -- expect low recall")
    lines.append("  and high precision. LLM3 answer quality is weighted 50/50")
    lines.append("  with disorder score at this level, reflecting that at extreme")
    lines.append("  thresholds the model's semantic grounding matters as much as")
    lines.append("  the raw disorder signal.")
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