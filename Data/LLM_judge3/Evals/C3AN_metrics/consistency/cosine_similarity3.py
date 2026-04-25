"""
BMEN-499 AlphaFold -- LLM Judge 3: Cosine Similarity Score
-----------------------------------------------------------
File    : cosine_similarity3.py
Output  : cosine_similarity3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Computes cosine similarity between:
        1. Question vector vs Answer vector       (Q-A relevance)
        2. Answer vector vs Retrieved passage     (A-P grounding)
        3. Question vector vs Retrieved passage   (Q-P retrieval quality)

    All vectors are computed using TF-IDF weighted bag-of-words
    (no external model required -- pure stdlib + math).

Similarity interpretation:
    0.80 - 1.00  : Very High -- answer closely mirrors question/passage
    0.60 - 0.79  : High      -- strong topical overlap
    0.40 - 0.59  : Moderate  -- partial overlap
    0.20 - 0.39  : Low       -- weak overlap
    0.00 - 0.19  : Very Low  -- little to no overlap

Output: cosine_similarity3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "cosine_similarity3_output.txt")

# ── Biomedical stopwords ───────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "so", "if",
    "than", "then", "when", "which", "who", "what", "how", "also",
    "more", "most", "after", "before", "between", "into", "through",
    "while", "both", "each", "than", "further", "once", "above",
    "below", "up", "down", "out", "about", "over", "under", "again",
    "here", "there", "where", "why", "all", "any", "few", "very",
    "just", "because", "such", "only", "their", "they", "them",
    "we", "our", "can", "cannot", "per", "i", "mean", "well",
}

# ── Text utilities ────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return tokens


def tf(tokens: list) -> dict:
    counts = Counter(tokens)
    total  = len(tokens) if tokens else 1
    return {w: c / total for w, c in counts.items()}


def idf(all_token_lists: list) -> dict:
    n     = len(all_token_lists)
    df    = Counter()
    for tokens in all_token_lists:
        for w in set(tokens):
            df[w] += 1
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}


def tfidf_vector(tokens: list, idf_scores: dict) -> dict:
    tf_scores = tf(tokens)
    return {w: tf_scores[w] * idf_scores.get(w, 1.0) for w in tf_scores}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot    = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity_label(score: float) -> str:
    if score >= 0.80:   return "VERY HIGH"
    elif score >= 0.60: return "HIGH     "
    elif score >= 0.40: return "MODERATE "
    elif score >= 0.20: return "LOW      "
    else:               return "VERY LOW "


# ── Parse LLM3_predictions.txt ────────────────────────────────────────

def parse_predictions(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    blocks      = re.split(r"={6,}", text)
    predictions = []

    q_pat = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )
    r_pat = re.compile(
        r"RETRIEVAL DETAILS[:\s]*\n(.*?)(?:\n\s*Top retrieved|$)",
        re.DOTALL
    )

    for block in blocks:
        q_m = q_pat.search(block)
        a_m = a_pat.search(block)
        r_m = r_pat.search(block)
        if q_m and a_m:
            q_num    = int(q_m.group(1))
            q_text   = q_m.group(2).strip()
            a_text   = re.sub(r"\s+", " ", a_m.group(1)).strip()
            r_text   = re.sub(r"\s+", " ", r_m.group(1)).strip() if r_m else ""
            predictions.append({
                "q_num":    q_num,
                "question": q_text,
                "answer":   a_text,
                "retrieval": r_text,
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions


# ── Compute similarities ──────────────────────────────────────────────

def compute_similarities(predictions: list) -> list:
    # Build IDF across all questions + answers + retrieval texts
    all_token_lists = []
    for p in predictions:
        all_token_lists.append(tokenize(p["question"]))
        all_token_lists.append(tokenize(p["answer"]))
        all_token_lists.append(tokenize(p["retrieval"]))

    idf_scores = idf(all_token_lists)

    results = []
    for p in predictions:
        q_tokens = tokenize(p["question"])
        a_tokens = tokenize(p["answer"])
        r_tokens = tokenize(p["retrieval"])

        q_vec = tfidf_vector(q_tokens, idf_scores)
        a_vec = tfidf_vector(a_tokens, idf_scores)
        r_vec = tfidf_vector(r_tokens, idf_scores)

        qa_sim = cosine_similarity(q_vec, a_vec)   # Q vs A
        ap_sim = cosine_similarity(a_vec, r_vec)   # A vs Retrieval
        qp_sim = cosine_similarity(q_vec, r_vec)   # Q vs Retrieval

        results.append({
            "q_num":    p["q_num"],
            "question": p["question"],
            "qa_sim":   round(qa_sim, 4),
            "ap_sim":   round(ap_sim, 4),
            "qp_sim":   round(qp_sim, 4),
            "mean_sim": round((qa_sim + ap_sim + qp_sim) / 3, 4),
        })

    return results


# ── Statistics helper ─────────────────────────────────────────────────

def stats(values: list) -> dict:
    n    = len(values)
    mu   = sum(values) / n
    sd   = math.sqrt(sum((x - mu) ** 2 for x in values) / n)
    mn   = min(values)
    mx   = max(values)
    vals = sorted(values)
    med  = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"mean": round(mu, 4), "std": round(sd, 4),
            "min": round(mn, 4), "max": round(mx, 4), "median": round(med, 4)}


# ── Write output ──────────────────────────────────────────────────────

def write_output(results: list, out_path: str):
    lines = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Cosine Similarity Scores")
    lines.append("  Script  : cosine_similarity3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {len(results)}")
    lines.append("  Method  : TF-IDF weighted cosine similarity (no external model)")
    lines.append("=" * 70)
    lines.append("")

    lines.append("SIMILARITY DIMENSIONS")
    lines.append("-" * 70)
    lines.append("  Q-A  : Question vs Answer        -- measures answer relevance")
    lines.append("  A-P  : Answer vs Retrieved Passage -- measures answer grounding")
    lines.append("  Q-P  : Question vs Retrieved Passage -- measures retrieval quality")
    lines.append("  MEAN : Average of Q-A, A-P, Q-P")
    lines.append("")
    lines.append("  Score ranges:")
    lines.append("    0.80-1.00  VERY HIGH   0.60-0.79  HIGH")
    lines.append("    0.40-0.59  MODERATE    0.20-0.39  LOW     0.00-0.19  VERY LOW")
    lines.append("")

    lines.append("PER-QUESTION COSINE SIMILARITY SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'Q-A':>8} {'Label':<12} {'A-P':>8} {'Label':<12} {'Q-P':>8} {'Label':<12} {'MEAN':>8}")
    lines.append("  " + "-" * 72)

    for r in results:
        lines.append(
            f"  {r['q_num']:>4}  "
            f"{r['qa_sim']:>8.4f} {similarity_label(r['qa_sim'])}  "
            f"{r['ap_sim']:>8.4f} {similarity_label(r['ap_sim'])}  "
            f"{r['qp_sim']:>8.4f} {similarity_label(r['qp_sim'])}  "
            f"{r['mean_sim']:>8.4f}"
        )

    # Aggregate stats
    qa_vals   = [r["qa_sim"]   for r in results]
    ap_vals   = [r["ap_sim"]   for r in results]
    qp_vals   = [r["qp_sim"]   for r in results]
    mean_vals = [r["mean_sim"] for r in results]

    qa_stats   = stats(qa_vals)
    ap_stats   = stats(ap_vals)
    qp_stats   = stats(qp_vals)
    mean_stats = stats(mean_vals)

    lines.append("")
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  {'Dimension':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    lines.append("  " + "-" * 52)
    for label, s in [("Q-A", qa_stats), ("A-P", ap_stats),
                      ("Q-P", qp_stats), ("MEAN", mean_stats)]:
        lines.append(
            f"  {label:<10} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f} {s['median']:>8.4f}"
        )

    # Distribution buckets
    lines.append("")
    lines.append("MEAN SIMILARITY DISTRIBUTION")
    lines.append("-" * 70)
    buckets = {
        "VERY HIGH (0.80-1.00)": sum(1 for v in mean_vals if v >= 0.80),
        "HIGH      (0.60-0.79)": sum(1 for v in mean_vals if 0.60 <= v < 0.80),
        "MODERATE  (0.40-0.59)": sum(1 for v in mean_vals if 0.40 <= v < 0.60),
        "LOW       (0.20-0.39)": sum(1 for v in mean_vals if 0.20 <= v < 0.40),
        "VERY LOW  (0.00-0.19)": sum(1 for v in mean_vals if v < 0.20),
    }
    n = len(results)
    for label, count in buckets.items():
        bar = "#" * int(count / n * 40)
        lines.append(f"  {label} : {count:>4} ({count/n*100:>5.1f}%)  {bar}")

    # Top and bottom 5
    sorted_by_mean = sorted(results, key=lambda r: r["mean_sim"], reverse=True)
    lines.append("")
    lines.append("TOP 5 MOST SIMILAR Q&A PAIRS (by mean score)")
    lines.append("-" * 70)
    for r in sorted_by_mean[:5]:
        lines.append(f"  Q{r['q_num']:>3}  mean={r['mean_sim']:.4f}  "
                     f"QA={r['qa_sim']:.4f}  AP={r['ap_sim']:.4f}  QP={r['qp_sim']:.4f}")
        lines.append(f"       \"{r['question'][:60]}\"")

    lines.append("")
    lines.append("BOTTOM 5 LEAST SIMILAR Q&A PAIRS (by mean score)")
    lines.append("-" * 70)
    for r in sorted_by_mean[-5:]:
        lines.append(f"  Q{r['q_num']:>3}  mean={r['mean_sim']:.4f}  "
                     f"QA={r['qa_sim']:.4f}  AP={r['ap_sim']:.4f}  QP={r['qp_sim']:.4f}")
        lines.append(f"       \"{r['question'][:60]}\"")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  Q-A similarity measures how well the answer addresses the question.")
    lines.append("  A-P similarity measures how grounded the answer is in the retrieved")
    lines.append("  passages (high = extractive, low = hallucinated or drifted).")
    lines.append("  Q-P similarity measures retrieval quality -- whether BiomedBERT")
    lines.append("  retrieved passages relevant to the question.")
    lines.append("")
    lines.append("  In extractive fallback mode (BioMistral not loaded), A-P should")
    lines.append("  be very high since answers ARE the passages. Lower Q-A scores")
    lines.append("  indicate the retrieved passages did not directly address the")
    lines.append("  specific question asked.")
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
    print("[INFO] Computing TF-IDF cosine similarities...\n")
    results = compute_similarities(predictions)
    write_output(results, OUT_PATH)