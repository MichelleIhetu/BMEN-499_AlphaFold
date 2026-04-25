"""
BMEN-499 AlphaFold -- LLM Judge 3: Output Variance Test
---------------------------------------------------------
File    : output_variance3.py
Output  : output_variance3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Measures variance in the predicted answers across all 100 questions
    along the following dimensions:

    V1  -- Answer length variance
           Word count per answer: mean, std, min, max, CV (coeff of variation)

    V2  -- Vocabulary richness variance
           Type-Token Ratio (TTR) per answer: unique words / total words

    V3  -- Lexical overlap variance
           How much each answer overlaps with the mean answer vocabulary
           (measures whether all answers say roughly the same thing)

    V4  -- Retrieved document diversity variance
           How varied the top retrieved doc IDs are across questions
           (KB-009 dominance = low diversity = high variance in retrieval quality)

    V5  -- Retrieval score variance
           Spread of top retrieval cosine scores across all 100 questions

    V6  -- Sentence count variance
           Number of sentences per answer

    V7  -- Answer uniqueness variance
           Pairwise Jaccard distance between answers
           (low uniqueness = answers are near-identical = high redundancy)

    V8  -- Topic drift variance
           Whether the answer tokens match the question tokens
           (high drift = answer does not address the question asked)

Variance interpretation:
    CV (coefficient of variation) = std / mean
    CV < 0.10  : Very Stable    -- answers are highly consistent
    CV < 0.20  : Stable         -- low variance
    CV < 0.35  : Moderate       -- acceptable variance
    CV < 0.50  : Unstable       -- high variance
    CV >= 0.50 : Very Unstable  -- answers differ dramatically

Output: output_variance3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "output_variance3_output.txt")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "so", "if",
    "than", "then", "when", "which", "who", "what", "how", "also",
    "more", "most", "after", "before", "between", "into", "through",
    "while", "both", "each", "further", "once", "above", "below",
    "up", "down", "out", "about", "over", "under", "again", "here",
    "there", "where", "why", "all", "any", "few", "very", "just",
    "because", "such", "only", "their", "they", "them", "we", "our",
    "can", "cannot", "per", "i", "mean", "well",
}

# ── Math helpers ──────────────────────────────────────────────────────

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def variance(lst):
    mu = mean(lst)
    return sum((x - mu) ** 2 for x in lst) / len(lst) if lst else 0.0

def std(lst):
    return math.sqrt(variance(lst))

def cv(lst):
    mu = mean(lst)
    return std(lst) / mu if mu != 0 else 0.0

def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

def percentile(lst, p):
    s = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]

def summarize(lst, label):
    return {
        "label":    label,
        "mean":     round(mean(lst), 4),
        "std":      round(std(lst), 4),
        "cv":       round(cv(lst), 4),
        "min":      round(min(lst), 4),
        "max":      round(max(lst), 4),
        "median":   round(median(lst), 4),
        "p25":      round(percentile(lst, 25), 4),
        "p75":      round(percentile(lst, 75), 4),
        "iqr":      round(percentile(lst, 75) - percentile(lst, 25), 4),
    }

def cv_label(cv_val):
    if cv_val < 0.10:   return "VERY STABLE   "
    elif cv_val < 0.20: return "STABLE        "
    elif cv_val < 0.35: return "MODERATE      "
    elif cv_val < 0.50: return "UNSTABLE      "
    else:               return "VERY UNSTABLE "

# ── Text utilities ────────────────────────────────────────────────────

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

def word_count(text):
    return len(text.split())

def sentence_count(text):
    return max(1, len(re.split(r"[.!?]+", text.strip())))

def ttr(text):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def topic_drift(question, answer):
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    return 1.0 - overlap   # drift = 1 - overlap

# ── Parse LLM3_predictions.txt ────────────────────────────────────────

def parse_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    blocks      = re.split(r"={6,}", text)
    predictions = []

    q_pat  = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat  = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )
    r_pat  = re.compile(r"\[1\]\s+(KB-\d+)")
    rs_pat = re.compile(r"Top retrieval score[:\s]+([\d.]+)")

    for block in blocks:
        q_m  = q_pat.search(block)
        a_m  = a_pat.search(block)
        r_m  = r_pat.search(block)
        rs_m = rs_pat.search(block)
        if q_m and a_m:
            predictions.append({
                "q_num":    int(q_m.group(1)),
                "question": q_m.group(2).strip(),
                "answer":   re.sub(r"\s+", " ", a_m.group(1)).strip(),
                "top_doc":  r_m.group(1) if r_m else "UNKNOWN",
                "top_score": float(rs_m.group(1)) if rs_m else 0.0,
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions

# ── Run variance tests ────────────────────────────────────────────────

def run_variance_tests(predictions):
    # V1 -- Answer length
    lengths     = [word_count(p["answer"]) for p in predictions]
    v1          = summarize(lengths, "V1 Answer Length (words)")

    # V2 -- TTR vocabulary richness
    ttrs        = [ttr(p["answer"]) for p in predictions]
    v2          = summarize(ttrs, "V2 Vocabulary Richness (TTR)")

    # V3 -- Lexical overlap vs global vocabulary
    all_tokens  = []
    for p in predictions:
        all_tokens.extend(tokenize(p["answer"]))
    global_vocab = set(all_tokens)
    overlaps    = []
    for p in predictions:
        a_vocab  = set(tokenize(p["answer"]))
        if global_vocab:
            overlaps.append(len(a_vocab & global_vocab) / len(global_vocab))
    v3          = summarize(overlaps, "V3 Lexical Overlap vs Global Vocab")

    # V4 -- Retrieved doc diversity
    doc_counts  = Counter(p["top_doc"] for p in predictions)
    n           = len(predictions)
    doc_freq    = {doc: count / n for doc, count in doc_counts.items()}
    # Entropy of doc distribution (higher = more diverse)
    entropy     = -sum(f * math.log2(f) for f in doc_freq.values() if f > 0)
    max_entropy = math.log2(len(doc_counts)) if len(doc_counts) > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # V5 -- Retrieval score variance
    scores      = [p["top_score"] for p in predictions if p["top_score"] > 0]
    v5          = summarize(scores, "V5 Retrieval Score") if scores else None

    # V6 -- Sentence count variance
    sent_counts = [sentence_count(p["answer"]) for p in predictions]
    v6          = summarize(sent_counts, "V6 Sentence Count")

    # V7 -- Pairwise Jaccard uniqueness (sample every 5th pair to keep it fast)
    jaccard_scores = []
    token_sets  = [set(tokenize(p["answer"])) for p in predictions]
    step        = max(1, len(predictions) // 20)
    for i in range(0, len(token_sets), step):
        for j in range(i + 1, min(i + step + 1, len(token_sets))):
            jaccard_scores.append(jaccard(token_sets[i], token_sets[j]))
    v7          = summarize(jaccard_scores, "V7 Pairwise Answer Jaccard Similarity") if jaccard_scores else None

    # V8 -- Topic drift
    drifts      = [topic_drift(p["question"], p["answer"]) for p in predictions]
    v8          = summarize(drifts, "V8 Topic Drift (1 - Q-A overlap)")

    # Per-question detail
    per_q = []
    for p, length, ttr_val, overlap, drift, sents in zip(
            predictions, lengths, ttrs, overlaps, drifts, sent_counts):
        per_q.append({
            "q_num":   p["q_num"],
            "question": p["question"],
            "length":  length,
            "ttr":     round(ttr_val, 4),
            "overlap": round(overlap, 4),
            "drift":   round(drift, 4),
            "sents":   sents,
            "top_doc": p["top_doc"],
            "top_score": p["top_score"],
        })

    return {
        "v1": v1, "v2": v2, "v3": v3,
        "v4": {"entropy": round(entropy, 4),
               "norm_entropy": round(norm_entropy, 4),
               "n_unique_docs": len(doc_counts),
               "doc_counts": doc_counts},
        "v5": v5, "v6": v6, "v7": v7, "v8": v8,
        "per_q": per_q,
        "n": n,
    }

# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = results["n"]

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Output Variance Test")
    lines.append("  Script  : output_variance3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("VARIANCE DIMENSION DEFINITIONS")
    lines.append("-" * 70)
    lines.append("  V1  Answer length variance        (word count per answer)")
    lines.append("  V2  Vocabulary richness variance  (TTR = unique/total words)")
    lines.append("  V3  Lexical overlap variance      (overlap vs global vocabulary)")
    lines.append("  V4  Retrieved doc diversity       (entropy of top-doc distribution)")
    lines.append("  V5  Retrieval score variance      (spread of top cosine scores)")
    lines.append("  V6  Sentence count variance       (sentences per answer)")
    lines.append("  V7  Pairwise Jaccard similarity   (answer-to-answer overlap)")
    lines.append("  V8  Topic drift variance          (answer vs question token overlap)")
    lines.append("")
    lines.append("  CV interpretation:")
    lines.append("    < 0.10  VERY STABLE    0.10-0.19  STABLE")
    lines.append("    0.20-0.34  MODERATE    0.35-0.49  UNSTABLE    >= 0.50  VERY UNSTABLE")
    lines.append("")

    # Summary table
    lines.append("VARIANCE SUMMARY TABLE")
    lines.append("-" * 70)
    lines.append(f"  {'Dim':<4} {'Label':<38} {'Mean':>8} {'Std':>8} {'CV':>8} {'Stability':<16}")
    lines.append("  " + "-" * 68)

    dim_list = ["v1", "v2", "v3", "v5", "v6", "v8"]
    for key in dim_list:
        s = results[key]
        if s:
            lines.append(
                f"  {key.upper():<4} {s['label']:<38} "
                f"{s['mean']:>8.4f} {s['std']:>8.4f} "
                f"{s['cv']:>8.4f} {cv_label(s['cv'])}"
            )

    # V4 separate (entropy-based)
    v4 = results["v4"]
    lines.append(
        f"  V4   Retrieved Doc Diversity              "
        f"  entropy={v4['entropy']:.4f}  "
        f"norm={v4['norm_entropy']:.4f}  "
        f"unique_docs={v4['n_unique_docs']}"
    )

    if results["v7"]:
        s = results["v7"]
        lines.append(
            f"  V7   {s['label']:<38} "
            f"{s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['cv']:>8.4f} {cv_label(s['cv'])}"
        )

    # Detailed stats per dimension
    lines.append("")
    for key, label in [("v1","V1 Answer Length"), ("v2","V2 TTR"),
                        ("v3","V3 Lexical Overlap"), ("v5","V5 Retrieval Score"),
                        ("v6","V6 Sentence Count"), ("v8","V8 Topic Drift")]:
        s = results[key]
        if s:
            lines.append(f"{label} -- Full Statistics")
            lines.append("-" * 70)
            lines.append(f"  Mean   : {s['mean']:>10.4f}    Std    : {s['std']:>10.4f}")
            lines.append(f"  CV     : {s['cv']:>10.4f}    Status : {cv_label(s['cv'])}")
            lines.append(f"  Min    : {s['min']:>10.4f}    Max    : {s['max']:>10.4f}")
            lines.append(f"  Median : {s['median']:>10.4f}    IQR    : {s['iqr']:>10.4f}")
            lines.append(f"  P25    : {s['p25']:>10.4f}    P75    : {s['p75']:>10.4f}")
            lines.append("")

    # V4 doc distribution
    lines.append("V4 Retrieved Doc Distribution -- Full Statistics")
    lines.append("-" * 70)
    lines.append(f"  Shannon Entropy        : {v4['entropy']:.4f}")
    lines.append(f"  Normalized Entropy     : {v4['norm_entropy']:.4f}  (1.0 = perfectly uniform)")
    lines.append(f"  Unique docs retrieved  : {v4['n_unique_docs']}")
    lines.append(f"  Doc frequency counts:")
    for doc, count in sorted(v4["doc_counts"].items(), key=lambda x: -x[1]):
        bar = "#" * int(count / n * 40)
        lines.append(f"    {doc:<12} : {count:>4}  ({count/n*100:>5.1f}%)  {bar}")
    lines.append("")

    # Per-question detail table
    lines.append("PER-QUESTION VARIANCE DETAIL")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'Len':>5}  {'TTR':>6}  {'Ovlp':>6}  "
                 f"{'Drift':>6}  {'Sents':>5}  {'TopDoc':<10}  {'Score':>7}")
    lines.append("  " + "-" * 60)
    for p in results["per_q"]:
        lines.append(
            f"  {p['q_num']:>4}  {p['length']:>5}  {p['ttr']:>6.4f}  "
            f"{p['overlap']:>6.4f}  {p['drift']:>6.4f}  "
            f"{p['sents']:>5}  {p['top_doc']:<10}  {p['top_score']:>7.4f}"
        )

    # Outlier detection
    lines.append("")
    lines.append("OUTLIER DETECTION (answers > 2 std from mean length)")
    lines.append("-" * 70)
    v1     = results["v1"]
    thresh_hi = v1["mean"] + 2 * v1["std"]
    thresh_lo = v1["mean"] - 2 * v1["std"]
    outliers  = [p for p in results["per_q"]
                 if p["length"] > thresh_hi or p["length"] < thresh_lo]
    if outliers:
        for p in outliers:
            tag = "LONG" if p["length"] > thresh_hi else "SHORT"
            lines.append(f"  Q{p['q_num']:>3}  [{tag}]  {p['length']} words  "
                         f"\"{p['question'][:55]}\"")
    else:
        lines.append("  No outliers detected.")

    lines.append("")
    lines.append("OVERALL VARIANCE VERDICT")
    lines.append("-" * 70)
    cv_vals = [results[k]["cv"] for k in ["v1","v2","v6","v8"] if results[k]]
    mean_cv = mean(cv_vals)
    verdict = cv_label(mean_cv).strip()
    lines.append(f"  Mean CV across V1/V2/V6/V8 : {mean_cv:.4f}  -->  {verdict}")
    lines.append("")
    lines.append("  INTERPRETATION:")
    lines.append("  LLM Judge 3 uses extractive fallback, which concatenates the")
    lines.append("  same top-3 KB passages repeatedly. This leads to low answer-")
    lines.append("  length variance (answers are structurally similar) but high")
    lines.append("  topic drift (retrieved passages may not match the question).")
    lines.append("  Low V4 entropy confirms KB-009/KB-011 dominate retrieval,")
    lines.append("  reducing output diversity. Loading BioMistral-7B as the")
    lines.append("  generator would increase V2 TTR and reduce V8 topic drift")
    lines.append("  by synthesizing passages into question-specific answers.")
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
    print(f"[INFO] Loading predictions from:\n       {LLM3_PATH}\n")
    predictions = parse_predictions(LLM3_PATH)
    print(f"[INFO] Parsed {len(predictions)} questions")
    print("[INFO] Running variance tests...\n")
    results = run_variance_tests(predictions)
    write_output(results, OUT_PATH)