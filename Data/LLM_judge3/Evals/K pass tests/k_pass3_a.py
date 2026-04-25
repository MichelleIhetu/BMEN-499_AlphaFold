"""
BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test Level 1 (LENIENT)
-----------------------------------------------------------------
File    : k_pass3_a.py
Output  : k_pass3_a_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

Strictness Level : 1 / 5  --  LENIENT
K                : 5
Threshold        : disorder_content >= 0.10  (very permissive)
Min Region Length: 1 aa
Calibration      : None
Split Strategy   : Random shuffle, no stratification
Metrics          : Accuracy, Precision, Recall, F1, fold variance

Purpose:
    Absolute baseline. Nearly everything with ANY disorder content
    passes. Establishes ceiling-level recall so tighter thresholds
    can be compared against it. Applied to LLM Judge 3 predictions
    to measure how well BioMistral RAG answers align with lenient
    disorder classifications from the DisProt dataset.
"""

import json
import re
import os
import math
import random
from pathlib import Path
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH    = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
DISPROT_PATH = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\Baseline\DisProt_ProteinData.json"
OUT_PATH     = os.path.join(SCRIPT_DIR, "k_pass3_a_output.txt")

SEED         = 42
K            = 5
THRESHOLD    = 0.10
MIN_REGION   = 1
LEVEL_LABEL  = "L1 LENIENT"

# ── Math helpers ──────────────────────────────────────────────────────

def mean(lst):   return sum(lst) / len(lst) if lst else 0.0
def std(lst):
    mu = mean(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / len(lst)) if lst else 0.0

# ── Load DisProt ──────────────────────────────────────────────────────

def load_disprot(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    return raw

# ── Load LLM3 predictions ─────────────────────────────────────────────

def load_predictions(path):
    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
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

# ── Protein helpers ───────────────────────────────────────────────────

def get_disorder_score(p):
    v = p.get("disorder_content_pure") or p.get("disorder_content_obs")
    return float(v) if v is not None else 0.0

def get_region_count(p, min_len):
    return sum(
        1 for r in p.get("regions", [])
        if isinstance(r, dict) and r.get("end", 0) - r.get("start", 0) + 1 >= min_len
    )

def protein_label(p, threshold, min_len):
    return 1 if (get_disorder_score(p) >= threshold or get_region_count(p, min_len) > 0) else 0

# ── Answer quality scorer (LLM3-specific) ────────────────────────────

def answer_quality_score(answer):
    """
    Score an LLM3 answer on a 0-1 scale for use as prediction signal.
    Combines factual signal density and biomedical term coverage.
    """
    a      = answer.lower()
    signals = [
        "plddt", "disprot", "disorder", "threshold", "pfam",
        "proline", "glycine", "idr", "idp", "alphafold",
        "experimentally", "validated", "sequence", "residue",
        "sliding window", "calibrat", "intrinsically disordered",
    ]
    hits  = sum(1 for s in signals if s in a)
    score = min(1.0, hits / len(signals) * 2)
    return score

# ── K-fold split ──────────────────────────────────────────────────────

def split_k_folds(data, k, seed):
    random.seed(seed)
    idx  = list(range(len(data)))
    random.shuffle(idx)
    fold_size = len(idx) // k
    folds = []
    for i in range(k):
        s = i * fold_size
        e = s + fold_size if i < k - 1 else len(idx)
        folds.append(idx[s:e])
    return folds

# ── Evaluate fold ─────────────────────────────────────────────────────

def evaluate_fold(proteins, predictions, train_idx, test_idx, threshold, min_len):
    # Train: mean disorder score as baseline
    train_scores = [get_disorder_score(proteins[i]) for i in train_idx]
    train_mean   = mean(train_scores)
    eff_threshold = min(threshold, train_mean * 1.2)

    # Map predictions to test proteins by index
    n_preds = len(predictions)
    tp = fp = tn = fn = 0

    for i in test_idx:
        true_lbl = protein_label(proteins[i], threshold, min_len)
        score    = get_disorder_score(proteins[i])
        # LLM3 answer quality boosts prediction confidence
        pred_idx = i % n_preds
        aq       = answer_quality_score(predictions[pred_idx]["answer"])
        combined = score * 0.7 + aq * 0.3
        pred_lbl = 1 if combined >= eff_threshold else 0

        if   true_lbl == 1 and pred_lbl == 1: tp += 1
        elif true_lbl == 0 and pred_lbl == 1: fp += 1
        elif true_lbl == 0 and pred_lbl == 0: tn += 1
        else:                                  fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    pos_rate  = (tp + fn) / total if total else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "eff_threshold": round(eff_threshold, 4),
        "pos_rate": round(pos_rate, 4),
        "n_test": total,
    }

# ── Run K-fold ────────────────────────────────────────────────────────

def run_kfold(proteins, predictions):
    folds   = split_k_folds(proteins, K, SEED)
    results = []
    for fold_idx in range(K):
        test_idx  = folds[fold_idx]
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        res = evaluate_fold(proteins, predictions, train_idx, test_idx, THRESHOLD, MIN_REGION)
        results.append(res)
        print(f"  Fold {fold_idx+1}/{K}: acc={res['accuracy']:.4f}  "
              f"f1={res['f1']:.4f}  thr={res['eff_threshold']:.4f}  "
              f"pos={res['pos_rate']:.3f}")
    return results

# ── Write output ──────────────────────────────────────────────────────

def write_output(results, proteins, predictions, out_path):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- LLM Judge 3: K-Pass Test {LEVEL_LABEL}")
    lines.append(f"  Script   : k_pass3_a.py")
    lines.append(f"  K={K}  |  Threshold={THRESHOLD}  |  MinRegion={MIN_REGION} aa")
    lines.append(f"  Proteins : {len(proteins):,}  |  Predictions : {len(predictions)}")
    lines.append(f"  Seed     : {SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("STRICTNESS LEVEL 1 -- LENIENT")
    lines.append("-" * 70)
    lines.append("  Disorder threshold : 0.10  (very permissive)")
    lines.append("  Min region length  : 1 aa  (any region counts)")
    lines.append("  Calibration        : None")
    lines.append("  Split strategy     : Random shuffle, no stratification")
    lines.append("  Prediction signal  : 70% disorder score + 30% LLM3 answer quality")
    lines.append("")

    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
                 f"{'Thr':>8} {'Pos':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    lines.append("  " + "-" * 72)
    for i, r in enumerate(results, 1):
        lines.append(f"  {i:<6} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                     f"{r['recall']:>8.4f} {r['f1']:>8.4f} "
                     f"{r['eff_threshold']:>8.4f} {r['pos_rate']:>7.3f} "
                     f"{r['tp']:>6} {r['fp']:>6} {r['tn']:>6} {r['fn']:>6}")

    accs  = [r["accuracy"]  for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    f1s   = [r["f1"]        for r in results]

    lines.append("")
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean Accuracy  : {mean(accs):.4f}  (+/- {std(accs):.4f})")
    lines.append(f"  Mean Precision : {mean(precs):.4f}  (+/- {std(precs):.4f})")
    lines.append(f"  Mean Recall    : {mean(recs):.4f}  (+/- {std(recs):.4f})")
    lines.append(f"  Mean F1        : {mean(f1s):.4f}  (+/- {std(f1s):.4f})")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  L1 Lenient is the most permissive configuration (threshold=0.10).")
    lines.append("  Expected very high recall -- almost all proteins pass.")
    lines.append("  Use this as the upper-bound recall reference when comparing")
    lines.append("  across k_pass3_a through k_pass3_e.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] {out_path}")

# ── Entry point ───────────────────────────────────────────────────────

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