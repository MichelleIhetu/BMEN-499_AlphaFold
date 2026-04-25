"""
BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test Level 3 (MODERATE)
------------------------------------------------------------------
File    : k_pass3_c.py
Output  : k_pass3_c_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

Strictness Level : 3 / 5  --  MODERATE
K                : 5
Threshold        : disorder_content >= 0.50  (canonical IDR cutoff)
Min Region Length: 10 aa
Calibration      : Z-score normalization per fold
Split Strategy   : Stratified by disorder score quartile
Metrics          : Accuracy, Precision, Recall, F1, MCC

Purpose:
    The canonical 0.50 threshold used in most IDR benchmarks.
    Stratified splitting ensures balanced disorder distributions.
    Z-score calibration normalises per fold. MCC added as the
    primary metric for imbalanced datasets.
"""

import json
import re
import os
import math
import random
from pathlib import Path

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH    = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
DISPROT_PATH = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\Baseline\DisProt_ProteinData.json"
OUT_PATH     = os.path.join(SCRIPT_DIR, "k_pass3_c_output.txt")

SEED        = 42
K           = 5
THRESHOLD   = 0.50
MIN_REGION  = 10
LEVEL_LABEL = "L3 MODERATE"

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

def get_region_count(p, min_len):
    return sum(
        1 for r in p.get("regions", [])
        if isinstance(r, dict) and r.get("end", 0) - r.get("start", 0) + 1 >= min_len
    )

def protein_label(p, threshold, min_len):
    return 1 if (get_disorder_score(p) >= threshold and get_region_count(p, min_len) > 0) else 0

def answer_quality_score(answer):
    a = answer.lower()
    signals = [
        "plddt", "disprot", "disorder", "threshold", "pfam",
        "proline", "glycine", "idr", "idp", "alphafold",
        "experimentally", "validated", "sequence", "residue",
        "sliding window", "calibrat", "intrinsically disordered",
        "gray zone", "pyrrolidine", "conformational", "molecular recognition",
    ]
    hits = sum(1 for s in signals if s in a)
    return min(1.0, hits / len(signals) * 2)

def stratified_k_folds(proteins, k, seed):
    random.seed(seed)
    scores     = [(i, get_disorder_score(p)) for i, p in enumerate(proteins)]
    sorted_    = sorted(scores, key=lambda x: x[1])
    n          = len(sorted_)
    groups     = [[] for _ in range(4)]
    for rank, (idx, _) in enumerate(sorted_):
        groups[min(int(rank / n * 4), 3)].append(idx)
    folds = [[] for _ in range(k)]
    for group in groups:
        random.shuffle(group)
        for i, idx in enumerate(group):
            folds[i % k].append(idx)
    return folds

def z_score(val, mu, sigma):
    return (val - mu) / sigma if sigma > 1e-9 else val

def evaluate_fold(proteins, predictions, train_idx, test_idx, threshold, min_len):
    train_scores = [get_disorder_score(proteins[i]) for i in train_idx]
    mu           = mean(train_scores)
    sigma        = std(train_scores) or 1.0
    cal_thr_z    = z_score(threshold, mu, sigma)
    n_preds      = len(predictions)
    tp = fp = tn = fn = 0

    for i in test_idx:
        true_lbl = protein_label(proteins[i], threshold, min_len)
        score    = get_disorder_score(proteins[i])
        score_z  = z_score(score, mu, sigma)
        pred_idx = i % n_preds
        aq       = answer_quality_score(predictions[pred_idx]["answer"])
        aq_z     = z_score(aq, 0.5, 0.2)
        combined = score_z * 0.60 + aq_z * 0.40
        pred_lbl = 1 if combined >= cal_thr_z else 0

        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    denom_mcc = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc       = (tp * tn - fp * fn) / denom_mcc
    pos_rate  = (tp + fn) / total if total else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "mcc": mcc,
        "train_mu": round(mu, 4), "train_sigma": round(sigma, 4),
        "pos_rate": round(pos_rate, 4), "n_test": total,
    }

def run_kfold(proteins, predictions):
    folds   = stratified_k_folds(proteins, K, SEED)
    results = []
    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, predictions, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  "
              f"f1={res['f1']:.4f}  mcc={res['mcc']:.4f}  pos={res['pos_rate']:.3f}")
    return results

def write_output(results, proteins, predictions, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test {LEVEL_LABEL}")
    lines.append(f"  Script   : k_pass3_c.py")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Proteins : {len(proteins):,}  |  Predictions : {len(predictions)}")
    lines.append(f"  Seed     : {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 3 -- MODERATE")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.50  (canonical IDR benchmark cutoff)")
    lines.append("  Min region length  : 10 aa")
    lines.append("  Calibration        : Z-score normalization per fold")
    lines.append("  Split strategy     : Stratified by disorder score quartile")
    lines.append("  Extra metric       : MCC (Matthews Correlation Coefficient)")
    lines.append("  Prediction signal  : 60% disorder score (z) + 40% LLM3 quality (z)")
    lines.append("")
    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'MCC':>8} {'mu':>7} {'sigma':>7} {'Pos':>6}")
    lines.append("  " + "-" * 68)
    for i, r in enumerate(results, 1):
        lines.append(f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                     f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} "
                     f"{r['train_mu']:>7.4f} {r['train_sigma']:>7.4f} {r['pos_rate']:>6.3f}")

    accs  = [r["accuracy"]  for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]
    mccs  = [r["mcc"]       for r in results]

    all_tp = sum(r["tp"] for r in results)
    all_fp = sum(r["fp"] for r in results)
    all_tn = sum(r["tn"] for r in results)
    all_fn = sum(r["fn"] for r in results)

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mean(accs):.4f}  (+/- {std(accs):.4f})")
    lines.append(f"  Mean Precision : {mean(precs):.4f}  (+/- {std(precs):.4f})")
    lines.append(f"  Mean Recall    : {mean(recs):.4f}  (+/- {std(recs):.4f})")
    lines.append(f"  Mean F1        : {mean(f1s):.4f}  (+/- {std(f1s):.4f})")
    lines.append(f"  Mean MCC       : {mean(mccs):.4f}  (+/- {std(mccs):.4f})")
    lines.append(f"  Confusion totals: TP={all_tp:,} FP={all_fp:,} TN={all_tn:,} FN={all_fn:,}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L3 Moderate uses the canonical 0.50 IDR threshold.")
    lines.append("  This is the primary reference fold configuration.")
    lines.append("  MCC is the most balanced metric for imbalanced datasets.")
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