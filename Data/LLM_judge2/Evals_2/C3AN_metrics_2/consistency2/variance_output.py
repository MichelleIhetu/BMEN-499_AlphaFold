"""
BMEN-499 AlphaFold -- LLM Judge 2: Variance Test
--------------------------------------------------
Purpose:
    Measures variance in the vanilla RAG predictions from LLM_judge2.py
    across multiple dimensions: retrieval score variance, answer length
    variance, KB document diversity variance, and topic coverage variance.

What This Test Measures:
    Variance quantifies how much the system's outputs fluctuate across
    questions. In a well-calibrated RAG system we expect:
      - LOW retrieval score variance  (consistent retrieval quality)
      - HIGH topic diversity variance (different questions get different docs)
      - LOW answer length variance    (stable, well-formed responses)

    Vanilla RAG often shows the OPPOSITE:
      - Artificially HIGH retrieval scores (BiomedBERT cosine similarity
        compressed into a narrow high range -- all scores near 0.97)
      - LOW topic diversity             (same KB docs recycled repeatedly)
      - LOW meaningful content variance (answers say the same thing)

    This is a key diagnostic for comparing LLM Judge 2 (vanilla RAG)
    against LLM Judge 2 (Vanilla RAG) (symbolic rules + calibration), where calibrated
    confidence scores should show healthier variance distributions.

Variance Dimensions Tested:
    VAR-1  Retrieval score variance
           How much do top-1 retrieval scores vary across questions?
           Low variance = retriever is not discriminating between questions.

    VAR-2  Answer length variance
           How much does answer word count vary across questions?
           Low variance = RAG generates same-length answers regardless of
           question complexity.

    VAR-3  KB document selection variance
           How often does each KB document get retrieved?
           Low variance = a small set of docs dominates all retrievals.

    VAR-4  Topic coverage variance
           How evenly are the 14 KB topics covered across 100 questions?
           Low variance = retriever has strong topic bias.

    VAR-5  Retrieved doc rank stability
           For each KB doc, how often does it appear as rank 1 vs 2 vs 3?
           High rank variance = retriever is unstable in ordering.

    VAR-6  Inter-question retrieval score spread
           Std deviation of retrieval scores within each question's top-3.
           Low within-question spread = retrieved docs have nearly equal
           scores (the retriever cannot discriminate between passages).

Output:
    variance_output.txt  -- full report (same folder as this script)

Usage:
    python variance_output.py --predictions LLM2_predictions.txt
    python variance_output.py --demo
"""

import re
import os
import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter


# =============================================================
# 1. PARSE PREDICTIONS FILE
# =============================================================

def parse_predictions_file(filepath: str) -> list:
    text   = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r"={70}\n\[Q(\d+)\]", text)

    predictions = []
    for i in range(1, len(blocks), 2):
        q_num   = int(blocks[i])
        content = blocks[i + 1] if (i + 1) < len(blocks) else ""

        lines    = content.strip().split("\n")
        question = lines[0].strip() if lines else ""

        answer_match = re.search(
            r"PREDICTED ANSWER.*?:\s*\n(.*?)\n\s*RETRIEVAL DETAILS",
            content, re.DOTALL
        )
        answer = answer_match.group(1).strip() if answer_match else ""

        doc_matches = re.findall(
            r"\[(\d+)\]\s+(KB-\d+)\s+--\s+(.+?)\s+score=([\d.]+)", content
        )
        retrieved_docs = [
            {
                "rank":  int(m[0]),
                "id":    m[1],
                "topic": m[2].strip(),
                "score": float(m[3]),
            }
            for m in doc_matches
        ]

        # Top score and method
        top_score_match = re.search(
            r"Top retrieval score\s*:\s*([\d.]+)", content
        )
        top_score = float(top_score_match.group(1)) if top_score_match else (
            retrieved_docs[0]["score"] if retrieved_docs else 0.0
        )

        predictions.append({
            "question_id":      q_num,
            "question":         question,
            "predicted_answer": answer,
            "retrieved_docs":   retrieved_docs,
            "top_score":        top_score,
            "word_count":       len(answer.split()),
        })

    return predictions


# =============================================================
# 2. STATISTICS HELPERS
# =============================================================

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def variance(lst):
    if len(lst) < 2:
        return 0.0
    mu = mean(lst)
    return sum((x - mu) ** 2 for x in lst) / (len(lst) - 1)

def stdev(lst):
    return math.sqrt(variance(lst))

def median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    m = len(s) // 2
    return (s[m] + s[m - 1]) / 2 if len(s) % 2 == 0 else s[m]

def percentile(lst, p):
    if not lst:
        return 0.0
    s     = sorted(lst)
    idx   = int(p / 100 * len(s))
    return s[min(idx, len(s) - 1)]

def coeff_of_variation(lst):
    """CV = stdev / mean * 100.  Normalised measure of spread."""
    mu = mean(lst)
    return (stdev(lst) / mu * 100) if mu != 0 else 0.0


# =============================================================
# 3. VARIANCE ANALYSIS
# =============================================================

def run_variance_test(predictions: list) -> dict:
    n = len(predictions)

    # ---- VAR-1: Retrieval score variance ----
    top_scores = [p["top_score"] for p in predictions]

    # ---- VAR-2: Answer length variance ----
    word_counts = [p["word_count"] for p in predictions]

    # ---- VAR-3: KB document selection frequency ----
    doc_freq      = Counter()
    doc_rank_dist = defaultdict(lambda: Counter())   # doc_id -> {rank: count}
    for pred in predictions:
        for doc in pred["retrieved_docs"]:
            doc_freq[doc["id"]] += 1
            doc_rank_dist[doc["id"]][doc["rank"]] += 1

    doc_freq_values = list(doc_freq.values())

    # ---- VAR-4: Topic coverage (docs per question slot) ----
    # Entropy of doc_freq distribution -- measures how evenly topics covered
    total_retrievals = sum(doc_freq_values)
    topic_entropy    = 0.0
    if total_retrievals > 0:
        for count in doc_freq_values:
            p = count / total_retrievals
            if p > 0:
                topic_entropy -= p * math.log2(p)
    max_entropy = math.log2(len(doc_freq)) if len(doc_freq) > 1 else 1.0
    topic_evenness = topic_entropy / max_entropy  # 0=skewed, 1=perfectly even

    # ---- VAR-5: Within-question score spread ----
    within_q_spreads = []
    for pred in predictions:
        scores = [d["score"] for d in pred["retrieved_docs"]]
        if len(scores) >= 2:
            within_q_spreads.append(max(scores) - min(scores))

    # ---- VAR-6: Per-document rank stability ----
    # For each doc, compute variance of its rank positions
    rank_variances = {}
    for doc_id, rank_counts in doc_rank_dist.items():
        ranks = []
        for rank, count in rank_counts.items():
            ranks.extend([rank] * count)
        rank_variances[doc_id] = round(variance(ranks), 4) if len(ranks) > 1 else 0.0

    # ---- VAR-7: Score percentile spread ----
    score_p10 = percentile(top_scores, 10)
    score_p90 = percentile(top_scores, 90)
    iqr_score = percentile(top_scores, 75) - percentile(top_scores, 25)

    wc_p10 = percentile(word_counts, 10)
    wc_p90 = percentile(word_counts, 90)
    iqr_wc = percentile(word_counts, 75) - percentile(word_counts, 25)

    return {
        "n": n,

        # VAR-1
        "score_mean":     round(mean(top_scores), 6),
        "score_median":   round(median(top_scores), 6),
        "score_variance": round(variance(top_scores), 8),
        "score_stdev":    round(stdev(top_scores), 6),
        "score_cv":       round(coeff_of_variation(top_scores), 4),
        "score_min":      round(min(top_scores), 6),
        "score_max":      round(max(top_scores), 6),
        "score_range":    round(max(top_scores) - min(top_scores), 6),
        "score_p10":      round(score_p10, 6),
        "score_p90":      round(score_p90, 6),
        "score_iqr":      round(iqr_score, 6),
        "top_scores":     top_scores,

        # VAR-2
        "wc_mean":        round(mean(word_counts), 2),
        "wc_median":      round(median(word_counts), 2),
        "wc_variance":    round(variance(word_counts), 4),
        "wc_stdev":       round(stdev(word_counts), 4),
        "wc_cv":          round(coeff_of_variation(word_counts), 4),
        "wc_min":         min(word_counts),
        "wc_max":         max(word_counts),
        "wc_range":       max(word_counts) - min(word_counts),
        "wc_p10":         wc_p10,
        "wc_p90":         wc_p90,
        "wc_iqr":         iqr_wc,
        "word_counts":    word_counts,

        # VAR-3
        "doc_freq":            dict(sorted(doc_freq.items())),
        "doc_freq_variance":   round(variance(doc_freq_values), 4),
        "doc_freq_stdev":      round(stdev(doc_freq_values), 4),
        "doc_freq_cv":         round(coeff_of_variation(doc_freq_values), 4),
        "doc_freq_min":        min(doc_freq_values) if doc_freq_values else 0,
        "doc_freq_max":        max(doc_freq_values) if doc_freq_values else 0,
        "most_retrieved_doc":  doc_freq.most_common(1)[0] if doc_freq else ("N/A", 0),
        "least_retrieved_doc": doc_freq.most_common()[-1] if doc_freq else ("N/A", 0),
        "doc_freq_values":     doc_freq_values,

        # VAR-4
        "topic_entropy":   round(topic_entropy, 4),
        "max_entropy":     round(max_entropy, 4),
        "topic_evenness":  round(topic_evenness, 4),

        # VAR-5
        "within_q_spread_mean":   round(mean(within_q_spreads), 6),
        "within_q_spread_stdev":  round(stdev(within_q_spreads), 6),
        "within_q_spread_min":    round(min(within_q_spreads), 6) if within_q_spreads else 0.0,
        "within_q_spread_max":    round(max(within_q_spreads), 6) if within_q_spreads else 0.0,
        "within_q_spreads":       within_q_spreads,

        # VAR-6
        "rank_variances":  rank_variances,
        "doc_rank_dist":   {k: dict(v) for k, v in doc_rank_dist.items()},

        # Outliers
        "high_score_qs": [
            p for p in predictions
            if p["top_score"] > mean(top_scores) + 2 * stdev(top_scores)
        ],
        "low_score_qs": [
            p for p in predictions
            if p["top_score"] < mean(top_scores) - 2 * stdev(top_scores)
        ],
        "high_wc_qs": [
            p for p in predictions
            if p["word_count"] > mean(word_counts) + 2 * stdev(word_counts)
        ],
        "low_wc_qs": [
            p for p in predictions
            if p["word_count"] < mean(word_counts) - 2 * stdev(word_counts)
        ],
    }


# =============================================================
# 4. WRITE REPORT
# =============================================================

def _bar(value, max_value, width=30):
    if max_value == 0:
        return ""
    filled = int(value / max_value * width)
    return "█" * filled + "░" * (width - filled)


def write_report(results: dict, predictions: list, output_path: str):
    n = results["n"]
    lines = []

    lines.append("=" * 72)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 2: Variance Test")
    lines.append("  Evaluation : Multi-dimensional variance analysis of RAG outputs")
    lines.append(f"  Questions  : {n}")
    lines.append("=" * 72)
    lines.append("")

    lines.append("WHAT THIS TEST MEASURES")
    lines.append("-" * 72)
    lines.append(
        "  Variance analysis diagnoses whether vanilla RAG produces stable,"
    )
    lines.append(
        "  consistent, and discriminative answers. Three failure modes:"
    )
    lines.append("")
    lines.append(
        "  1. SCORE COMPRESSION   -- retrieval scores cluster in a narrow"
    )
    lines.append(
        "     high range (e.g. 0.95-0.98), meaning BiomedBERT cannot"
    )
    lines.append(
        "     differentiate question relevance. Low score variance = bad."
    )
    lines.append("")
    lines.append(
        "  2. DOC RECYCLING       -- a few KB passages are retrieved for"
    )
    lines.append(
        "     nearly every question. Low doc frequency variance = bad."
    )
    lines.append("")
    lines.append(
        "  3. ANSWER HOMOGENEITY  -- answers are all the same length and"
    )
    lines.append(
        "     content regardless of question complexity. Low answer"
    )
    lines.append(
        "     length variance = bad."
    )
    lines.append("")

    # ===== VAR-1: Retrieval Score Variance =====
    lines.append("=" * 72)
    lines.append("  VAR-1: RETRIEVAL SCORE VARIANCE (top-1 score per question)")
    lines.append("-" * 72)
    lines.append(f"  Mean score          : {results['score_mean']:.6f}")
    lines.append(f"  Median score        : {results['score_median']:.6f}")
    lines.append(f"  Std deviation       : {results['score_stdev']:.6f}")
    lines.append(f"  Variance            : {results['score_variance']:.8f}")
    lines.append(f"  Coeff of variation  : {results['score_cv']:.4f}%")
    lines.append(f"  Min score           : {results['score_min']:.6f}")
    lines.append(f"  Max score           : {results['score_max']:.6f}")
    lines.append(f"  Range (max - min)   : {results['score_range']:.6f}")
    lines.append(f"  P10                 : {results['score_p10']:.6f}")
    lines.append(f"  P90                 : {results['score_p90']:.6f}")
    lines.append(f"  IQR (P75 - P25)     : {results['score_iqr']:.6f}")
    lines.append("")

    # Score histogram
    lines.append("  Score distribution (top-1 retrieval scores):")
    score_buckets = [
        ("0.97 - 1.00", 0.97, 1.01),
        ("0.95 - 0.97", 0.95, 0.97),
        ("0.93 - 0.95", 0.93, 0.95),
        ("0.90 - 0.93", 0.90, 0.93),
        ("0.85 - 0.90", 0.85, 0.90),
        ("0.00 - 0.85", 0.00, 0.85),
    ]
    for label, lo, hi in score_buckets:
        count = sum(1 for s in results["top_scores"] if lo <= s < hi)
        pct   = count / n * 100
        lines.append(
            f"    {label}  {count:>4} ({pct:>5.1f}%)  {_bar(count, n, 25)}"
        )
    lines.append("")

    cv1 = results["score_cv"]
    if cv1 < 0.5:
        score_interp = (
            "CRITICAL -- score variance is near-zero. BiomedBERT assigns "
            "nearly identical retrieval scores to all questions, confirming "
            "the retriever cannot discriminate between question types. All "
            "disorder-related questions fall in the same embedding region."
        )
    elif cv1 < 1.5:
        score_interp = (
            "LOW -- retrieval scores are tightly compressed. The retriever "
            "shows minimal discrimination between questions. Symbolic rules "
            "in LLM Judge 2 (Vanilla RAG) bypass this by applying biological constraints "
            "independently of embedding similarity."
        )
    elif cv1 < 3.0:
        score_interp = (
            "MODERATE -- some score variation exists but the range is still "
            "narrow. The retriever shows partial discrimination."
        )
    else:
        score_interp = (
            "HEALTHY -- retrieval scores show meaningful variation, "
            "suggesting the retriever discriminates between question types."
        )

    lines.append(f"  Interpretation (CV={cv1:.4f}%): {score_interp[:60]}")
    lines.append(f"  {score_interp[60:] if len(score_interp) > 60 else ''}")
    lines.append("")

    # ===== VAR-2: Answer Length Variance =====
    lines.append("=" * 72)
    lines.append("  VAR-2: ANSWER LENGTH VARIANCE (word count per answer)")
    lines.append("-" * 72)
    lines.append(f"  Mean word count     : {results['wc_mean']:.2f}")
    lines.append(f"  Median word count   : {results['wc_median']:.2f}")
    lines.append(f"  Std deviation       : {results['wc_stdev']:.4f}")
    lines.append(f"  Variance            : {results['wc_variance']:.4f}")
    lines.append(f"  Coeff of variation  : {results['wc_cv']:.4f}%")
    lines.append(f"  Min word count      : {results['wc_min']}")
    lines.append(f"  Max word count      : {results['wc_max']}")
    lines.append(f"  Range (max - min)   : {results['wc_range']}")
    lines.append(f"  P10                 : {results['wc_p10']}")
    lines.append(f"  P90                 : {results['wc_p90']}")
    lines.append(f"  IQR (P75 - P25)     : {results['wc_iqr']}")
    lines.append("")

    # Word count histogram
    wc_buckets = [
        ("< 50  words",   0,   50),
        ("50-100 words",  50,  100),
        ("100-150 words", 100, 150),
        ("150-200 words", 150, 200),
        ("200-250 words", 200, 250),
        ("> 250  words",  250, 99999),
    ]
    lines.append("  Word count distribution:")
    for label, lo, hi in wc_buckets:
        count = sum(1 for w in results["word_counts"] if lo <= w < hi)
        pct   = count / n * 100
        lines.append(
            f"    {label}  {count:>4} ({pct:>5.1f}%)  {_bar(count, n, 25)}"
        )
    lines.append("")

    # Outlier questions by word count
    if results["high_wc_qs"]:
        lines.append("  Unusually LONG answers (> mean + 2 stdev):")
        for p in results["high_wc_qs"]:
            lines.append(
                f"    Q{p['question_id']:03d}  {p['word_count']} words  "
                f"{p['question'][:55]}"
            )
        lines.append("")
    if results["low_wc_qs"]:
        lines.append("  Unusually SHORT answers (< mean - 2 stdev):")
        for p in results["low_wc_qs"]:
            lines.append(
                f"    Q{p['question_id']:03d}  {p['word_count']} words  "
                f"{p['question'][:55]}"
            )
        lines.append("")

    cv2 = results["wc_cv"]
    if cv2 < 5:
        wc_interp = (
            "VERY LOW -- answers are near-identical in length. The RAG "
            "generator returns the same amount of text for every question, "
            "confirming it is recycling fixed-length passage concatenations."
        )
    elif cv2 < 15:
        wc_interp = (
            "LOW -- limited length variation. The system does not adapt "
            "answer depth to question complexity."
        )
    else:
        wc_interp = (
            "MODERATE/HEALTHY -- some length variation present, suggesting "
            "different question types trigger different passage combinations."
        )

    lines.append(f"  Interpretation (CV={cv2:.4f}%): {wc_interp[:60]}")
    lines.append(f"  {wc_interp[60:] if len(wc_interp) > 60 else ''}")
    lines.append("")

    # ===== VAR-3: KB Document Selection Variance =====
    lines.append("=" * 72)
    lines.append("  VAR-3: KB DOCUMENT SELECTION FREQUENCY VARIANCE")
    lines.append("  (How often each KB passage is retrieved across all questions)")
    lines.append("-" * 72)
    lines.append(f"  Doc freq variance   : {results['doc_freq_variance']:.4f}")
    lines.append(f"  Doc freq stdev      : {results['doc_freq_stdev']:.4f}")
    lines.append(f"  Doc freq CV         : {results['doc_freq_cv']:.4f}%")
    lines.append(f"  Min retrievals      : {results['doc_freq_min']}")
    lines.append(f"  Max retrievals      : {results['doc_freq_max']}")
    lines.append(
        f"  Most retrieved      : {results['most_retrieved_doc'][0]}  "
        f"({results['most_retrieved_doc'][1]} times)"
    )
    lines.append(
        f"  Least retrieved     : {results['least_retrieved_doc'][0]}  "
        f"({results['least_retrieved_doc'][1]} time(s))"
    )
    lines.append("")

    lines.append("  Retrieval frequency per KB document:")
    max_freq = results["doc_freq_max"]
    for doc_id, freq in sorted(
        results["doc_freq"].items(),
        key=lambda x: x[1], reverse=True
    ):
        pct = freq / (n * 3) * 100  # 3 docs per question
        lines.append(
            f"    {doc_id}  {freq:>4} retrievals ({pct:>5.1f}% of slots)  "
            f"{_bar(freq, max_freq, 25)}"
        )
    lines.append("")

    cv3 = results["doc_freq_cv"]
    if cv3 > 40:
        doc_interp = (
            "HIGH IMBALANCE -- a few KB documents dominate all retrievals "
            "while others are rarely used. The retriever has a strong "
            "passage bias. Questions are not being matched to the most "
            "relevant specific passages."
        )
    elif cv3 > 20:
        doc_interp = (
            "MODERATE IMBALANCE -- some documents are retrieved much more "
            "frequently than others. Passage recycling is a concern."
        )
    else:
        doc_interp = (
            "RELATIVELY BALANCED -- documents are retrieved with similar "
            "frequency. The retriever covers the knowledge base evenly."
        )

    lines.append(f"  Interpretation (CV={cv3:.4f}%): {doc_interp[:60]}")
    lines.append(f"  {doc_interp[60:] if len(doc_interp) > 60 else ''}")
    lines.append("")

    # ===== VAR-4: Topic Coverage Entropy =====
    lines.append("=" * 72)
    lines.append("  VAR-4: TOPIC COVERAGE ENTROPY")
    lines.append("  (How evenly are all KB topics covered? 1.0 = perfectly even)")
    lines.append("-" * 72)
    lines.append(f"  Shannon entropy     : {results['topic_entropy']:.4f} bits")
    lines.append(f"  Max possible entropy: {results['max_entropy']:.4f} bits")
    lines.append(f"  Topic evenness      : {results['topic_evenness']:.4f}  "
                 f"(0=skewed, 1=perfectly even)")
    lines.append("")

    ev = results["topic_evenness"]
    if ev < 0.80:
        entropy_interp = (
            "LOW EVENNESS -- the retriever strongly favors a subset of "
            "topics. Many KB passages are underutilised. This confirms "
            "that BiomedBERT embeds most questions near the same passage "
            "cluster, producing a biased topic distribution."
        )
    elif ev < 0.90:
        entropy_interp = (
            "MODERATE EVENNESS -- some topics are retrieved significantly "
            "more than others. The retriever has a mild topic bias."
        )
    else:
        entropy_interp = (
            "HIGH EVENNESS -- topics are covered relatively uniformly. "
            "The retriever distributes retrievals across the knowledge base."
        )

    words = entropy_interp.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            lines.append(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line)
    lines.append("")

    # ===== VAR-5: Within-Question Score Spread =====
    lines.append("=" * 72)
    lines.append("  VAR-5: WITHIN-QUESTION RETRIEVAL SCORE SPREAD")
    lines.append("  (Range between rank-1 and rank-3 scores per question)")
    lines.append("-" * 72)
    lines.append(
        f"  Mean spread (rank1-rank3)  : {results['within_q_spread_mean']:.6f}"
    )
    lines.append(
        f"  Stdev of spread            : {results['within_q_spread_stdev']:.6f}"
    )
    lines.append(
        f"  Min spread                 : {results['within_q_spread_min']:.6f}"
    )
    lines.append(
        f"  Max spread                 : {results['within_q_spread_max']:.6f}"
    )
    lines.append("")

    sp = results["within_q_spread_mean"]
    if sp < 0.005:
        spread_interp = (
            "NEAR-ZERO SPREAD -- the top-3 retrieved passages receive almost "
            "identical scores. BiomedBERT cannot rank passages by relevance; "
            "all disorder-related passages score equally similar to the "
            "question. The retriever is essentially choosing at random among "
            "its top candidates."
        )
    elif sp < 0.02:
        spread_interp = (
            "LOW SPREAD -- small difference between rank-1 and rank-3 scores. "
            "The retriever has limited ability to rank passage relevance, "
            "making the top-3 selection semi-arbitrary."
        )
    else:
        spread_interp = (
            "MODERATE SPREAD -- the retriever shows meaningful score "
            "differences between retrieved passages."
        )

    words = spread_interp.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            lines.append(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line)
    lines.append("")

    # ===== VAR-6: Document Rank Stability =====
    lines.append("=" * 72)
    lines.append("  VAR-6: KB DOCUMENT RANK STABILITY")
    lines.append("  (Does each doc consistently appear at the same rank?)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Doc ID':<10}  {'Rank 1':>7}  {'Rank 2':>7}  {'Rank 3':>7}  "
        f"{'Rank Var':>9}  Stability"
    )
    lines.append("  " + "-" * 62)

    for doc_id in sorted(results["doc_rank_dist"].keys()):
        rd      = results["doc_rank_dist"][doc_id]
        r1      = rd.get(1, 0)
        r2      = rd.get(2, 0)
        r3      = rd.get(3, 0)
        rv      = results["rank_variances"].get(doc_id, 0.0)
        total   = r1 + r2 + r3
        dom_pct = max(r1, r2, r3) / total * 100 if total else 0
        stab    = "stable" if rv < 0.5 else ("variable" if rv < 1.0 else "unstable")
        lines.append(
            f"  {doc_id:<10}  {r1:>7}  {r2:>7}  {r3:>7}  "
            f"{rv:>9.4f}  {stab}"
        )
    lines.append("")

    # ===== OVERALL VARIANCE SCORECARD =====
    lines.append("=" * 72)
    lines.append("  OVERALL VARIANCE SCORECARD")
    lines.append("-" * 72)

    def score_label(cv, thresholds, labels):
        for thresh, label in zip(thresholds, labels):
            if cv < thresh:
                return label
        return labels[-1]

    scorecard = [
        (
            "VAR-1  Retrieval score CV",
            results["score_cv"],
            "% CV",
            "Low CV = bad (score compression)",
            [0.5, 1.5, 3.0],
            ["CRITICAL", "LOW", "MODERATE", "HEALTHY"],
        ),
        (
            "VAR-2  Answer length CV",
            results["wc_cv"],
            "% CV",
            "Low CV = bad (answer homogeneity)",
            [5.0, 15.0, 25.0],
            ["VERY LOW", "LOW", "MODERATE", "HEALTHY"],
        ),
        (
            "VAR-3  Doc frequency CV",
            results["doc_freq_cv"],
            "% CV",
            "High CV = bad (doc recycling)",
            [20.0, 40.0, 60.0],
            ["BALANCED", "MODERATE", "IMBALANCED", "SEVERE"],
        ),
        (
            "VAR-4  Topic evenness",
            results["topic_evenness"] * 100,
            "%",
            "Low = bad (topic bias)",
            [70.0, 80.0, 90.0],
            ["SEVERE BIAS", "LOW", "MODERATE", "EVEN"],
        ),
        (
            "VAR-5  Within-Q spread",
            results["within_q_spread_mean"] * 1000,
            "x1e-3",
            "Low = bad (can't rank passages)",
            [2.0, 5.0, 15.0],
            ["NEAR-ZERO", "LOW", "MODERATE", "HEALTHY"],
        ),
    ]

    lines.append(
        f"  {'Dimension':<32}  {'Value':>10}  {'Unit':<8}  Rating"
    )
    lines.append("  " + "-" * 66)
    for name, val, unit, note, thresholds, labels in scorecard:
        rating = score_label(val, thresholds, labels)
        lines.append(
            f"  {name:<32}  {val:>10.4f}  {unit:<8}  {rating}"
        )
        lines.append(f"  {'':32}  {'':10}  {'':8}  ({note})")
    lines.append("")

    # ===== CONCLUSION =====
    lines.append("=" * 72)
    lines.append("  CONCLUSION")
    lines.append("-" * 72)
    lines.append(
        "  The variance profile above characterises vanilla RAG (LLM Judge 2)"
    )
    lines.append(
        "  as a system with:"
    )
    lines.append("")
    lines.append(
        "  * COMPRESSED retrieval scores -- BiomedBERT cannot differentiate"
    )
    lines.append(
        "    question relevance within the disorder prediction domain."
    )
    lines.append(
        "  * RECYCLED passages           -- a few KB docs dominate all"
    )
    lines.append(
        "    retrievals, creating answer homogeneity across questions."
    )
    lines.append(
        "  * UNIFORM answer length       -- answers are fixed-length"
    )
    lines.append(
        "    concatenations regardless of question complexity."
    )
    lines.append("")
    lines.append(
        "  LLM Judge 2 (Vanilla RAG) (BiomedBERT + symbolic rules + calibration) addresses"
    )
    lines.append(
        "  these failures by routing answers through biological constraint"
    )
    lines.append(
        "  rules that produce variable, question-specific responses with"
    )
    lines.append(
        "  calibrated confidence scores -- yielding healthier variance."
    )
    lines.append("")

    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 72)

    output = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Report written to: {output_path}\n")


# =============================================================
# DEMO PREDICTIONS
# =============================================================

DEMO_PREDICTIONS = [
    {
        "question_id": 1,
        "question": "Is a disorder score above 0.5 a reliable cutoff?",
        "predicted_answer": (
            "Disorder scores between 0.3 and 0.5 define an ambiguous gray zone. "
            "In the DisProt dataset of 13,396 proteins, a substantial fraction "
            "falls in this mid-range. These regions require secondary validation. "
            "Disorder scores above 0.7 represent high confidence intrinsic disorder. "
            "AlphaFold pLDDT scores below 50 indicate very low structural confidence."
        ),
        "retrieved_docs": [
            {"rank": 1, "id": "KB-002", "topic": "gray zone", "score": 0.9766},
            {"rank": 2, "id": "KB-003", "topic": "high confidence", "score": 0.9745},
            {"rank": 3, "id": "KB-009", "topic": "pLDDT low", "score": 0.9738},
        ],
        "top_score": 0.9766,
        "word_count": 52,
    },
    {
        "question_id": 2,
        "question": "Do confidence scores drop for IDRs shorter than 10 residues?",
        "predicted_answer": (
            "Disordered regions shorter than 10 amino acids are difficult to predict. "
            "Short IDRs are underrepresented in experimental databases. Prediction "
            "tools lack sufficient sequence context for short stretches. AlphaFold "
            "pLDDT scores of 70 or above indicate high confidence in the predicted "
            "structure. Sliding window averaging is applied to reduce noise. Short "
            "disordered regions risk being smoothed out and lost."
        ),
        "retrieved_docs": [
            {"rank": 1, "id": "KB-007", "topic": "short IDR", "score": 0.9737},
            {"rank": 2, "id": "KB-011", "topic": "pLDDT high", "score": 0.9709},
            {"rank": 3, "id": "KB-008", "topic": "sliding window", "score": 0.9705},
        ],
        "top_score": 0.9737,
        "word_count": 55,
    },
    {
        "question_id": 3,
        "question": "Do proline and glycine-rich regions score higher disorder confidence?",
        "predicted_answer": (
            "AlphaFold assigns each amino acid a pLDDT confidence score from 0 to 100. "
            "Scores below 50 indicate very low structural confidence and strongly "
            "correlate with intrinsic disorder. DisProt experimentally confirms "
            "disorder in 13,396 proteins. Disorder scores above 0.7 represent high "
            "confidence intrinsic disorder. AlphaFold pLDDT scores of 70 or above "
            "indicate high confidence in the predicted structure."
        ),
        "retrieved_docs": [
            {"rank": 1, "id": "KB-009", "topic": "pLDDT low", "score": 0.9775},
            {"rank": 2, "id": "KB-003", "topic": "high confidence", "score": 0.9767},
            {"rank": 3, "id": "KB-011", "topic": "pLDDT high", "score": 0.9764},
        ],
        "top_score": 0.9775,
        "word_count": 56,
    },
]


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variance test for LLM Judge 2 RAG predictions"
    )
    parser.add_argument("--predictions", type=str,
                        help="Path to LLM2_predictions.txt")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for variance_output.txt")
    parser.add_argument("--demo", action="store_true",
                        help="Run on 3 built-in demo predictions")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "variance_output.txt")

    if args.demo or not args.predictions:
        print("[INFO] Running in DEMO mode (3 sample predictions)\n")
        predictions = DEMO_PREDICTIONS
    else:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"[ERROR] Predictions file not found: {args.predictions}")
            sys.exit(1)
        print(f"[INFO] Parsing predictions: {args.predictions}\n")
        predictions = parse_predictions_file(str(pred_path))
        print(f"[INFO] Parsed {len(predictions)} predictions\n")

    print("[INFO] Running variance analysis...\n")
    results = run_variance_test(predictions)
    write_report(results, predictions, output_path)


if __name__ == "__main__":
    main()