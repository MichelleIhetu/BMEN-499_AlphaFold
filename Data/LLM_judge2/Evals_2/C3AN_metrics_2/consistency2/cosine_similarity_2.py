"""
BMEN-499 AlphaFold -- LLM Judge 2: Cosine Similarity Consistency Test
-----------------------------------------------------------------------
Purpose:
    Measures answer-to-answer cosine similarity across the 100 vanilla
    RAG predictions from LLM_judge2.py to evaluate response consistency.

What This Test Measures:
    Vanilla RAG answers the same question by retrieving the top-K most
    semantically similar passages. If different questions retrieve
    overlapping passage sets, their answers will be highly similar --
    even when the questions ask about different things. This is a key
    weakness of pure neural retrieval without symbolic grounding.

    This test computes:
      1. Pairwise cosine similarity between all 100 predicted answers
         using TF-IDF vector representations.
      2. Mean, median, min, max similarity across all question pairs.
      3. Highly similar pairs (sim > threshold) -- answers that are
         essentially the same despite different questions.
      4. Per-question mean similarity -- how "generic" each answer is.
      5. Similarity heatmap data for visualization.

Why Cosine Similarity Matters for RAG Evaluation:
    High inter-answer similarity signals that the RAG system is
    collapsing diverse questions into a narrow set of retrieved passages.
    LLM Judge 1 (symbolic rules) should produce more discriminative
    answers because rules steer responses based on question type,
    whereas vanilla RAG relies solely on embedding similarity.

Output:
    cosine_similarity_2.txt  -- full report (same folder as this script)

Usage:
    python cosine_similarity_2.py --predictions LLM2_predictions.txt
    python cosine_similarity_2.py --demo
    python cosine_similarity_2.py --predictions LLM2_predictions.txt --threshold 0.85
"""

import re
import os
import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter


# =============================================================
# 1. TF-IDF VECTORIZER (pure stdlib, no sklearn required)
# =============================================================

def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


STOPWORDS = {
    "the", "and", "for", "are", "that", "this", "with", "from", "has",
    "have", "been", "not", "but", "its", "can", "may", "also", "more",
    "all", "each", "they", "their", "these", "which", "when", "where",
    "what", "how", "does", "into", "than", "such", "both", "very",
    "most", "some", "any", "per", "show", "shows", "indicate", "indicates",
    "likely", "using", "used", "use", "within", "below", "above", "across",
    "rather", "while", "would", "could", "should", "between", "same",
    "other", "exist", "exist", "include", "including", "however",
    "therefore", "thus", "hence", "one", "two", "three", "retrieved",
}


def build_tfidf(documents: list) -> tuple:
    """
    Build TF-IDF matrix from a list of text strings.
    Returns (tfidf_vectors, vocabulary) where each vector is a dict
    mapping term -> tfidf weight.
    """
    n_docs = len(documents)

    # Tokenize and compute term frequencies
    tokenized = []
    for doc in documents:
        tokens = [t for t in tokenize(doc) if t not in STOPWORDS]
        tokenized.append(tokens)

    # Document frequency
    df = defaultdict(int)
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    # Build vocabulary: terms appearing in at least 2 documents
    vocab = {term for term, count in df.items() if count >= 2}

    # Compute TF-IDF vectors
    vectors = []
    for tokens in tokenized:
        tf = Counter(t for t in tokens if t in vocab)
        total = sum(tf.values()) or 1
        vec = {}
        for term, count in tf.items():
            tf_val  = count / total
            idf_val = math.log((n_docs + 1) / (df[term] + 1)) + 1
            vec[term] = tf_val * idf_val
        vectors.append(vec)

    return vectors, vocab


def cosine_similarity(v1: dict, v2: dict) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not v1 or not v2:
        return 0.0
    dot   = sum(v1.get(t, 0) * v2.get(t, 0) for t in v1)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# =============================================================
# 2. PARSE PREDICTIONS FILE
# =============================================================

def parse_predictions_file(filepath: str) -> list:
    """
    Parse LLM2_predictions.txt into a list of prediction dicts.
    Each dict has: question_id, question, predicted_answer, retrieved_docs
    """
    text = Path(filepath).read_text(encoding="utf-8")

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
            {"rank": int(m[0]), "id": m[1], "topic": m[2].strip(),
             "score": float(m[3])}
            for m in doc_matches
        ]

        predictions.append({
            "question_id":      q_num,
            "question":         question,
            "predicted_answer": answer,
            "retrieved_docs":   retrieved_docs,
        })

    return predictions


# =============================================================
# 3. PAIRWISE COSINE SIMILARITY ANALYSIS
# =============================================================

def run_cosine_similarity_test(predictions: list,
                                high_sim_threshold: float = 0.90) -> dict:
    """
    Compute pairwise cosine similarities between all predicted answers.
    Returns a comprehensive results dict.
    """
    n = len(predictions)
    answers = [p["predicted_answer"] for p in predictions]

    print(f"[INFO] Building TF-IDF vectors for {n} answers...")
    vectors, vocab = build_tfidf(answers)
    print(f"[INFO] Vocabulary size: {len(vocab)} terms\n")

    # Compute all pairwise similarities (upper triangle)
    print(f"[INFO] Computing {n*(n-1)//2} pairwise cosine similarities...")
    all_sims    = []
    high_pairs  = []
    per_q_sims  = defaultdict(list)   # question_id -> list of sim values

    sim_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            sim = round(sim, 6)
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
            all_sims.append(sim)
            per_q_sims[predictions[i]["question_id"]].append(sim)
            per_q_sims[predictions[j]["question_id"]].append(sim)

            if sim >= high_sim_threshold:
                high_pairs.append({
                    "q_i":   predictions[i]["question_id"],
                    "q_j":   predictions[j]["question_id"],
                    "sim":   sim,
                    "q_i_text": predictions[i]["question"][:80],
                    "q_j_text": predictions[j]["question"][:80],
                    "top_doc_i": (
                        predictions[i]["retrieved_docs"][0]["id"]
                        if predictions[i]["retrieved_docs"] else "N/A"
                    ),
                    "top_doc_j": (
                        predictions[j]["retrieved_docs"][0]["id"]
                        if predictions[j]["retrieved_docs"] else "N/A"
                    ),
                })

    high_pairs.sort(key=lambda x: x["sim"], reverse=True)

    # Summary statistics
    def mean(lst):   return sum(lst) / len(lst) if lst else 0.0
    def median(lst):
        s = sorted(lst)
        m = len(s) // 2
        return (s[m] + s[m - 1]) / 2 if len(s) % 2 == 0 else s[m]
    def stdev(lst):
        if len(lst) < 2:
            return 0.0
        mu = mean(lst)
        return math.sqrt(sum((x - mu) ** 2 for x in lst) / (len(lst) - 1))

    # Distribution buckets
    buckets = {
        "0.95 - 1.00 (near-identical)": 0,
        "0.90 - 0.95 (very high)":      0,
        "0.80 - 0.90 (high)":           0,
        "0.70 - 0.80 (moderate-high)":  0,
        "0.50 - 0.70 (moderate)":       0,
        "0.00 - 0.50 (low)":            0,
    }
    for s in all_sims:
        if s >= 0.95:   buckets["0.95 - 1.00 (near-identical)"] += 1
        elif s >= 0.90: buckets["0.90 - 0.95 (very high)"]      += 1
        elif s >= 0.80: buckets["0.80 - 0.90 (high)"]           += 1
        elif s >= 0.70: buckets["0.70 - 0.80 (moderate-high)"]  += 1
        elif s >= 0.50: buckets["0.50 - 0.70 (moderate)"]       += 1
        else:           buckets["0.00 - 0.50 (low)"]            += 1

    # Per-question mean similarity (higher = more generic answer)
    per_q_stats = []
    for pred in predictions:
        qid   = pred["question_id"]
        sims  = per_q_sims[qid]
        per_q_stats.append({
            "question_id": qid,
            "question":    pred["question"],
            "mean_sim":    round(mean(sims), 6),
            "max_sim":     round(max(sims), 6) if sims else 0.0,
            "top_doc":     (
                pred["retrieved_docs"][0]["id"]
                if pred["retrieved_docs"] else "N/A"
            ),
        })
    per_q_stats.sort(key=lambda x: x["mean_sim"], reverse=True)

    # Most generic answers (top 10 by mean similarity to all other answers)
    most_generic   = per_q_stats[:10]
    most_distinct  = sorted(per_q_stats, key=lambda x: x["mean_sim"])[:10]

    # KB document overlap analysis
    doc_pair_counts = defaultdict(int)
    for pred in predictions:
        docs = [d["id"] for d in pred["retrieved_docs"]]
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pair = tuple(sorted([docs[i], docs[j]]))
                doc_pair_counts[pair] += 1

    top_doc_pairs = sorted(
        doc_pair_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "n_predictions":        n,
        "n_pairs":              len(all_sims),
        "vocab_size":           len(vocab),
        "high_sim_threshold":   high_sim_threshold,
        "mean_sim":             round(mean(all_sims), 6),
        "median_sim":           round(median(all_sims), 6),
        "stdev_sim":            round(stdev(all_sims), 6),
        "min_sim":              round(min(all_sims), 6) if all_sims else 0.0,
        "max_sim":              round(max(all_sims), 6) if all_sims else 0.0,
        "n_high_sim_pairs":     len(high_pairs),
        "pct_high_sim_pairs":   round(
            len(high_pairs) / len(all_sims) * 100, 2
        ) if all_sims else 0.0,
        "distribution":         buckets,
        "high_sim_pairs":       high_pairs[:50],   # top 50 for report
        "most_generic_answers": most_generic,
        "most_distinct_answers": most_distinct,
        "top_doc_pairs":        top_doc_pairs,
        "all_sims":             all_sims,           # for percentile calc
    }


# =============================================================
# 4. WRITE REPORT
# =============================================================

def write_report(results: dict, output_path: str):
    lines = []
    total_pairs = results["n_pairs"]

    lines.append("=" * 72)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 2: Cosine Similarity Test")
    lines.append("  Evaluation : Inter-answer cosine similarity (TF-IDF)")
    lines.append("  Purpose    : Measure answer diversity / genericity of RAG")
    lines.append(f"  Questions  : {results['n_predictions']}")
    lines.append(f"  Pairs      : {results['n_pairs']:,}")
    lines.append(f"  Vocabulary : {results['vocab_size']} terms")
    lines.append("=" * 72)
    lines.append("")

    lines.append("WHAT THIS TEST MEASURES")
    lines.append("-" * 72)
    lines.append(
        "  Cosine similarity between all pairs of predicted answers reveals"
    )
    lines.append(
        "  whether vanilla RAG produces discriminative, question-specific"
    )
    lines.append(
        "  responses or generic, passage-recycling answers. High pairwise"
    )
    lines.append(
        "  similarity indicates the retriever is returning overlapping"
    )
    lines.append(
        "  passage sets for different questions -- a hallmark failure of"
    )
    lines.append(
        "  pure neural retrieval without symbolic grounding."
    )
    lines.append("")
    lines.append(
        "  TF-IDF weighting down-weights common biomedical terms (pLDDT,"
    )
    lines.append(
        "  DisProt, disorder) that appear in all answers, so similarity"
    )
    lines.append(
        "  scores reflect actual content overlap rather than shared"
    )
    lines.append(
        "  vocabulary. Scores near 1.0 mean near-identical answers."
    )
    lines.append("")

    # ---- SUMMARY STATISTICS ----
    lines.append("=" * 72)
    lines.append("  SUMMARY STATISTICS")
    lines.append("-" * 72)
    lines.append(f"  Mean cosine similarity       : {results['mean_sim']:.6f}")
    lines.append(f"  Median cosine similarity     : {results['median_sim']:.6f}")
    lines.append(f"  Std deviation                : {results['stdev_sim']:.6f}")
    lines.append(f"  Min similarity               : {results['min_sim']:.6f}")
    lines.append(f"  Max similarity (excl. self)  : {results['max_sim']:.6f}")
    lines.append(f"  Threshold for 'high sim'     : >= {results['high_sim_threshold']:.2f}")
    lines.append(
        f"  Pairs above threshold        : "
        f"{results['n_high_sim_pairs']:,} / {total_pairs:,} "
        f"({results['pct_high_sim_pairs']:.2f}%)"
    )
    lines.append("")

    # Interpret mean similarity
    m = results["mean_sim"]
    if m >= 0.90:
        interp_label = "CRITICAL -- answers are near-identical"
        interp = (
            "The vast majority of predicted answers share nearly identical "
            "content. The RAG retriever is collapsing almost all questions "
            "into the same small passage set. The system provides essentially "
            "one answer regardless of the question asked."
        )
    elif m >= 0.80:
        interp_label = "HIGH -- severe answer recycling"
        interp = (
            "Answers are highly similar across questions. The vanilla RAG "
            "pipeline retrieves heavily overlapping passage sets because "
            "BiomedBERT embeds disorder-related questions in a tight semantic "
            "cluster. Symbolic rules (LLM Judge 1) would produce more "
            "discriminative responses by applying question-specific reasoning."
        )
    elif m >= 0.70:
        interp_label = "MODERATE-HIGH -- significant answer overlap"
        interp = (
            "Substantial content overlap exists between answers. The retriever "
            "is not differentiating well between question subtypes. A symbolic "
            "layer would reduce this overlap by routing questions to "
            "appropriate rule sets."
        )
    elif m >= 0.50:
        interp_label = "MODERATE -- some answer recycling"
        interp = (
            "Moderate similarity suggests partial answer recycling. Some "
            "question types are retrieving distinct passages while others "
            "share content. The symbolic rules in LLM Judge 1 should "
            "further improve differentiation."
        )
    else:
        interp_label = "LOW -- answers are reasonably diverse"
        interp = (
            "TF-IDF similarity is relatively low, suggesting the RAG system "
            "produces reasonably diverse answers. However, this may reflect "
            "vocabulary variation rather than substantive content differences."
        )

    lines.append(f"  Interpretation: {interp_label}")
    lines.append("")
    words = interp.split()
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

    # ---- DISTRIBUTION ----
    lines.append("=" * 72)
    lines.append("  SIMILARITY DISTRIBUTION")
    lines.append("-" * 72)
    lines.append(
        f"  {'Similarity Range':<35}  {'Count':>7}  {'% of pairs':>10}"
    )
    lines.append("  " + "-" * 58)
    for bucket, count in results["distribution"].items():
        pct = count / total_pairs * 100 if total_pairs else 0
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        lines.append(
            f"  {bucket:<35}  {count:>7,}  {pct:>9.2f}%  {bar}"
        )
    lines.append("")

    # ---- TOP DOC PAIRS ----
    lines.append("=" * 72)
    lines.append("  TOP CO-RETRIEVED KNOWLEDGE BASE DOCUMENT PAIRS")
    lines.append("  (Most frequently retrieved together -- drives answer similarity)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Doc Pair':<22}  {'Co-occurrences':>14}  "
        f"{'% of Questions':>14}"
    )
    lines.append("  " + "-" * 60)
    for (d1, d2), count in results["top_doc_pairs"]:
        pct = count / results["n_predictions"] * 100
        lines.append(
            f"  {d1} + {d2:<10}  {count:>14,}  {pct:>13.1f}%"
        )
    lines.append("")
    lines.append(
        "  NOTE: High co-occurrence means these passage pairs appear"
    )
    lines.append(
        "  together in many answers, contributing to answer similarity."
    )
    lines.append(
        "  Vanilla RAG has no mechanism to prevent this recycling."
    )
    lines.append("")

    # ---- HIGH SIMILARITY PAIRS ----
    lines.append("=" * 72)
    lines.append(
        f"  HIGH SIMILARITY PAIRS  (cosine >= {results['high_sim_threshold']:.2f})"
    )
    lines.append(
        f"  {results['n_high_sim_pairs']} pairs found"
    )
    lines.append("-" * 72)

    if results["high_sim_pairs"]:
        for pair in results["high_sim_pairs"][:30]:
            lines.append(
                f"\n  Q{pair['q_i']:03d} vs Q{pair['q_j']:03d}  "
                f"sim={pair['sim']:.6f}"
            )
            lines.append(f"    Q{pair['q_i']:03d}: {pair['q_i_text']}")
            lines.append(f"    Q{pair['q_j']:03d}: {pair['q_j_text']}")
            lines.append(
                f"    Top doc Q{pair['q_i']:03d}: {pair['top_doc_i']}  |  "
                f"Top doc Q{pair['q_j']:03d}: {pair['top_doc_j']}"
            )
        if results["n_high_sim_pairs"] > 30:
            lines.append(
                f"\n  ... and {results['n_high_sim_pairs'] - 30} more pairs "
                f"above threshold (showing top 30)"
            )
    else:
        lines.append(
            f"\n  No pairs exceed the {results['high_sim_threshold']:.2f} threshold."
        )
    lines.append("")

    # ---- MOST GENERIC ANSWERS ----
    lines.append("=" * 72)
    lines.append("  TOP 10 MOST GENERIC ANSWERS")
    lines.append("  (Highest mean similarity to all other answers)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Rank':<5}  {'Q#':<5}  {'Mean Sim':>9}  "
        f"{'Top Doc':<10}  Question"
    )
    lines.append("  " + "-" * 68)
    for rank, entry in enumerate(results["most_generic_answers"], 1):
        q_text = entry["question"][:50]
        lines.append(
            f"  {rank:<5}  Q{entry['question_id']:<4}  "
            f"{entry['mean_sim']:>9.6f}  "
            f"{entry['top_doc']:<10}  {q_text}"
        )
    lines.append("")

    # ---- MOST DISTINCT ANSWERS ----
    lines.append("=" * 72)
    lines.append("  TOP 10 MOST DISTINCT ANSWERS")
    lines.append("  (Lowest mean similarity to all other answers)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Rank':<5}  {'Q#':<5}  {'Mean Sim':>9}  "
        f"{'Top Doc':<10}  Question"
    )
    lines.append("  " + "-" * 68)
    for rank, entry in enumerate(results["most_distinct_answers"], 1):
        q_text = entry["question"][:50]
        lines.append(
            f"  {rank:<5}  Q{entry['question_id']:<4}  "
            f"{entry['mean_sim']:>9.6f}  "
            f"{entry['top_doc']:<10}  {q_text}"
        )
    lines.append("")

    # ---- PERCENTILE TABLE ----
    lines.append("=" * 72)
    lines.append("  SIMILARITY PERCENTILE TABLE")
    lines.append("-" * 72)
    sorted_sims = sorted(results["all_sims"])
    total = len(sorted_sims)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    lines.append(
        f"  {'Percentile':<15}  {'Cosine Similarity':>18}"
    )
    lines.append("  " + "-" * 36)
    for p in percentiles:
        idx = int(p / 100 * total)
        idx = min(idx, total - 1)
        lines.append(
            f"  P{p:<14}  {sorted_sims[idx]:>18.6f}"
        )
    lines.append("")

    # ---- IMPLICATIONS ----
    lines.append("=" * 72)
    lines.append("  IMPLICATIONS FOR LLM JUDGE 1 VS JUDGE 2 COMPARISON")
    lines.append("-" * 72)
    lines.append(
        "  This cosine similarity test quantifies a core limitation of"
    )
    lines.append(
        "  vanilla RAG: answer homogeneity. When BiomedBERT encodes all"
    )
    lines.append(
        "  disorder-related questions into a tight embedding cluster,"
    )
    lines.append(
        "  the retriever returns overlapping passage sets, producing"
    )
    lines.append(
        "  highly similar answers regardless of question specificity."
    )
    lines.append("")
    lines.append(
        "  LLM Judge 1 addresses this through symbolic rules that:"
    )
    lines.append(
        "    (1) Route questions to rule-specific answer branches"
    )
    lines.append(
        "    (2) Apply pLDDT / composition / threshold rules selectively"
    )
    lines.append(
        "    (3) Produce calibrated confidence scores that vary by case"
    )
    lines.append(
        "    (4) Prevent passage recycling through hard constraints"
    )
    lines.append("")
    lines.append(
        "  Expected: LLM Judge 1 inter-answer cosine similarity should"
    )
    lines.append(
        "  be significantly lower than LLM Judge 2, demonstrating that"
    )
    lines.append(
        "  symbolic rules improve answer discriminability."
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
        "question":    "Is a disorder score above 0.5 a reliable cutoff for calling a region disordered?",
        "predicted_answer": (
            "Disorder scores between 0.3 and 0.5 define an ambiguous gray "
            "zone where proteins cannot be confidently classified as ordered "
            "or disordered. In the DisProt dataset of 13,396 proteins, a "
            "substantial fraction falls in this mid-range. These regions "
            "require secondary validation using sequence composition or "
            "experimental methods. Disorder scores above 0.7 represent high "
            "confidence intrinsic disorder. These regions consistently "
            "correspond to experimentally validated IDRs in DisProt. "
            "AlphaFold assigns each amino acid a pLDDT confidence score from "
            "0 to 100. Scores below 50 indicate very low structural confidence "
            "and strongly correlate with intrinsic disorder."
        ),
        "retrieved_docs": [{"id": "KB-002"}, {"id": "KB-003"}, {"id": "KB-009"}],
    },
    {
        "question_id": 2,
        "question":    "Do confidence scores drop for IDRs shorter than 10 residues?",
        "predicted_answer": (
            "Disordered regions shorter than 10 amino acids are difficult to "
            "predict reliably. Short IDRs are underrepresented in experimental "
            "databases because prediction tools lack sufficient sequence "
            "context for short stretches. AlphaFold pLDDT scores of 70 or "
            "above indicate high confidence in the predicted structure. "
            "Regions with these scores are likely ordered. Where DisProt "
            "experimental annotations exist, experimental data should take "
            "precedence over computational predictions. Sliding window "
            "averaging is applied to per-residue disorder scores to reduce "
            "noise. Short disordered regions risk being smoothed out and lost."
        ),
        "retrieved_docs": [{"id": "KB-007"}, {"id": "KB-011"}, {"id": "KB-008"}],
    },
    {
        "question_id": 3,
        "question":    "Do proline and glycine-rich regions consistently score higher disorder confidence?",
        "predicted_answer": (
            "AlphaFold assigns each amino acid a pLDDT confidence score from "
            "0 to 100. Scores below 50 indicate very low structural confidence "
            "and strongly correlate with intrinsic disorder. DisProt "
            "experimentally confirms disorder in 13,396 proteins. Disorder "
            "scores above 0.7 represent high confidence intrinsic disorder. "
            "AlphaFold pLDDT scores of 70 or above indicate high confidence "
            "in the predicted structure. Regions with these scores are likely "
            "ordered and not intrinsically disordered. Where DisProt "
            "experimental annotations exist, experimental data should take "
            "precedence over computational predictions."
        ),
        "retrieved_docs": [{"id": "KB-009"}, {"id": "KB-003"}, {"id": "KB-011"}],
    },
    {
        "question_id": 4,
        "question":    "Does applying a sliding window smooth out confidence scores?",
        "predicted_answer": (
            "Disorder scores above 0.7 represent high confidence intrinsic "
            "disorder. These regions consistently correspond to experimentally "
            "validated IDRs in DisProt. AlphaFold pLDDT scores between 50 and "
            "70 indicate low but not absent structural confidence. These "
            "regions may be conditionally disordered unstructured in isolation "
            "but folding upon binding to a partner molecule. Sliding window "
            "averaging is applied to per-residue disorder scores to reduce "
            "noise. Short disordered regions risk being smoothed out and lost. "
            "Window size must be chosen carefully to balance noise reduction "
            "against signal preservation."
        ),
        "retrieved_docs": [{"id": "KB-003"}, {"id": "KB-010"}, {"id": "KB-008"}],
    },
    {
        "question_id": 5,
        "question":    "How do AlphaFold pLDDT scores correlate with known disordered regions?",
        "predicted_answer": (
            "AlphaFold assigns each amino acid a pLDDT confidence score from "
            "0 to 100. Scores below 50 indicate very low structural confidence "
            "and strongly correlate with intrinsic disorder. DisProt "
            "experimentally confirms disorder in 13,396 proteins; regions "
            "annotated as disordered in DisProt consistently show pLDDT below "
            "50 in AlphaFold predictions, making this the most reliable single "
            "computational signal. Disorder scores above 0.7 represent high "
            "confidence intrinsic disorder. AlphaFold pLDDT scores of 70 or "
            "above indicate high confidence in the predicted structure."
        ),
        "retrieved_docs": [{"id": "KB-009"}, {"id": "KB-003"}, {"id": "KB-011"}],
    },
]


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity consistency test for LLM Judge 2"
    )
    parser.add_argument(
        "--predictions", type=str,
        help="Path to LLM2_predictions.txt"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for cosine_similarity_2.txt"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Cosine similarity threshold for high-similarity pairs (default: 0.90)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on 5 built-in demo predictions"
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "cosine_similarity_2.txt")

    if args.demo or not args.predictions:
        print("[INFO] Running in DEMO mode (5 sample predictions)\n")
        predictions = DEMO_PREDICTIONS
    else:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"[ERROR] Predictions file not found: {args.predictions}")
            sys.exit(1)
        print(f"[INFO] Parsing predictions file: {args.predictions}\n")
        predictions = parse_predictions_file(str(pred_path))
        print(f"[INFO] Parsed {len(predictions)} predictions\n")

    results = run_cosine_similarity_test(
        predictions, high_sim_threshold=args.threshold
    )

    write_report(results, output_path)


if __name__ == "__main__":
    main()