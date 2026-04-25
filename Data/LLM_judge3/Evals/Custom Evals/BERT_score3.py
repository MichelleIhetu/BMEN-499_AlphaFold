"""
BMEN-499 AlphaFold -- LLM Judge 3: BERT Score Analysis
--------------------------------------------------------
File    : BERT_score3.py
Output  : BERT_score3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Computes a BERTScore-style semantic similarity metric between
    each predicted answer and a reference answer derived from the
    DisProt knowledge base passages.

    Since the real BERTScore library requires PyTorch + transformers
    (~2GB download), this script implements a lightweight TF-IDF
    approximation of BERTScore that captures the same three metrics:

        Precision (P) : proportion of answer tokens that are
                        semantically grounded in the reference

        Recall    (R) : proportion of reference tokens that are
                        covered by the answer

        F1            : harmonic mean of P and R
                        F1 = 2 * P * R / (P + R)

    Reference answers are constructed from the ground-truth DisProt
    KB passages that are most relevant to each question (matched
    by keyword overlap -- same logic BiomedBERT uses).

    Score interpretation (F1):
        >= 0.85  : Excellent  -- near-reference quality
        >= 0.70  : Good       -- strong semantic overlap
        >= 0.55  : Moderate   -- partial coverage
        >= 0.40  : Weak       -- limited alignment
        <  0.40  : Poor       -- minimal semantic match

Output: BERT_score3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "BERT_score3_output.txt")

# ── DisProt knowledge base reference passages ─────────────────────────
# These are the ground-truth reference texts used as BERTScore targets.
# Each passage is a factual, validated statement from the DisProt KB.
KNOWLEDGE_BASE = {
    "KB-001": "The 0.5 disorder score threshold classifies protein regions as intrinsically disordered. Of 13,396 DisProt proteins, 29.1% exceed this threshold with a mean disorder score of 0.378. However, 44.2% exceed 0.3, meaning many true IDRs fall below 0.5 and would be missed by a strict cutoff.",
    "KB-002": "Disorder scores between 0.3 and 0.5 define an ambiguous gray zone where proteins cannot be confidently classified as ordered or disordered. These regions require secondary validation using sequence composition or experimental methods.",
    "KB-003": "Disorder scores above 0.7 represent high confidence intrinsic disorder, consistently matching experimentally validated IDRs in DisProt and correlating with AlphaFold pLDDT scores below 50.",
    "KB-004": "Proline content with a DisProt mean of 6.0% strongly predicts intrinsic disorder. Proline's rigid pyrrolidine ring disrupts alpha-helices and beta-sheets, preventing regular secondary structure formation.",
    "KB-005": "Glycine with a DisProt mean of 7.0% adds backbone conformational freedom. It is a weaker independent disorder predictor than proline and should be evaluated in combination with other disorder signals.",
    "KB-006": "Elevated proline and glycine together form a strong composite disorder signal. Proline introduces backbone kinks while glycine adds excess conformational freedom, both disrupting regular folding.",
    "KB-007": "Of the annotated disordered regions in DisProt, a percentage are shorter than 10 residues with a mean region length of approximately 50 amino acids. Short IDRs are hard to predict reliably due to limited sequence context.",
    "KB-008": "Sliding window averaging smooths per-residue disorder scores to reduce noise. Windows larger than the mean region length risk smoothing out short IDRs entirely, requiring careful window size selection.",
    "KB-009": "AlphaFold pLDDT below 50 strongly indicates intrinsic disorder. DisProt-annotated disordered regions in 13,396 proteins consistently show pLDDT below 50, making this the most reliable single computational disorder signal.",
    "KB-010": "pLDDT scores of 50 to 70 indicate ambiguous structure. Regions may be conditionally disordered, known as Molecular Recognition Features or MoRFs, which are unstructured alone but fold upon binding to a partner molecule.",
    "KB-011": "pLDDT of 70 or above indicates confident AlphaFold structure prediction. These regions are likely ordered and not intrinsically disordered. DisProt experimental annotations take precedence over computational predictions.",
    "KB-012": "92.6% of DisProt proteins contain Pfam structured domains alongside disordered regions, confirming that IDRs and structured domains frequently co-occur. Each region must be evaluated independently.",
    "KB-013": "Proteins with no detectable Pfam domains and disorder content above 0.5 are classified as Intrinsically Disordered Proteins or IDPs. These are common in signaling, transcription regulation, and hub protein networks.",
    "KB-014": "DisProt contains 13,396 experimentally validated proteins with a mean disorder content of 0.378. There are thousands of annotated disordered regions with 29.1% of proteins exceeding the 0.5 threshold.",
}

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","this","that","these","those","it","its",
    "as","not","no","so","if","than","then","when","which","who",
    "what","how","also","more","most","after","before","between",
    "into","through","while","both","each","further","once","above",
    "below","up","down","out","about","over","under","again","here",
    "there","where","why","all","any","few","very","just","because",
    "such","only","their","they","them","we","our","can","cannot",
    "per","i","mean","well","does","across","without","whether",
    "among","within","against","along","during",
}

# ── Math helpers ──────────────────────────────────────────────────────

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std(lst):
    mu = mean(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / len(lst)) if lst else 0.0

def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

def percentile(lst, p):
    s   = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]

# ── TF-IDF BERTScore approximation ───────────────────────────────────

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def tf_idf_vector(tokens, idf_scores):
    counts = Counter(tokens)
    total  = len(tokens) if tokens else 1
    return {w: (counts[w] / total) * idf_scores.get(w, 1.0) for w in counts}


def build_idf(all_token_lists):
    n  = len(all_token_lists)
    df = Counter()
    for tokens in all_token_lists:
        for w in set(tokens):
            df[w] += 1
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}


def bert_score_approx(candidate_tokens, reference_tokens, idf_scores):
    """
    TF-IDF weighted BERTScore approximation.
    Precision: sum of max similarities for each candidate token.
    Recall   : sum of max similarities for each reference token.
    F1       : harmonic mean.
    """
    if not candidate_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0

    cand_vec = tf_idf_vector(candidate_tokens, idf_scores)
    ref_vec  = tf_idf_vector(reference_tokens,  idf_scores)

    # Precision: for each candidate token, find best match in reference
    p_scores = []
    for cw, cw_val in cand_vec.items():
        # Exact match scores 1.0 (weighted), partial via shared substrings
        if cw in ref_vec:
            p_scores.append(cw_val * ref_vec[cw])
        else:
            # Soft match: longest common substring fraction
            best = max(
                (len(set(cw) & set(rw)) / max(len(set(cw) | set(rw)), 1)
                 for rw in ref_vec),
                default=0.0
            )
            p_scores.append(cw_val * best * 0.5)

    # Recall: for each reference token, find best match in candidate
    r_scores = []
    for rw, rw_val in ref_vec.items():
        if rw in cand_vec:
            r_scores.append(rw_val * cand_vec[rw])
        else:
            best = max(
                (len(set(rw) & set(cw)) / max(len(set(rw) | set(cw)), 1)
                 for cw in cand_vec),
                default=0.0
            )
            r_scores.append(rw_val * best * 0.5)

    # Normalize
    cand_norm = math.sqrt(sum(v ** 2 for v in cand_vec.values())) or 1.0
    ref_norm  = math.sqrt(sum(v ** 2 for v in ref_vec.values()))  or 1.0

    P = sum(p_scores) / cand_norm
    R = sum(r_scores) / ref_norm
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return round(min(P, 1.0), 4), round(min(R, 1.0), 4), round(min(F, 1.0), 4)


# ── Reference selection ───────────────────────────────────────────────

def select_reference(question, top_doc_id):
    """
    Select the best reference passage for a question.
    Prefer the top retrieved doc if available, otherwise fall back
    to keyword-based selection from the KB.
    """
    # Use top retrieved doc if it's a valid KB entry
    if top_doc_id and top_doc_id in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[top_doc_id]

    # Keyword fallback
    q_lower   = question.lower()
    best_kb   = "KB-014"   # default: summary passage
    best_score = 0

    for kb_id, passage in KNOWLEDGE_BASE.items():
        q_tokens = set(tokenize(q_lower))
        p_tokens = set(tokenize(passage))
        score    = len(q_tokens & p_tokens)
        if score > best_score:
            best_score = score
            best_kb    = kb_id

    return KNOWLEDGE_BASE[best_kb]


def score_label(f1):
    if f1 >= 0.85:   return "EXCELLENT"
    elif f1 >= 0.70: return "GOOD     "
    elif f1 >= 0.55: return "MODERATE "
    elif f1 >= 0.40: return "WEAK     "
    else:            return "POOR     "


# ── Parse predictions ─────────────────────────────────────────────────

def parse_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    blocks      = re.split(r"={6,}", text)
    predictions = []
    q_pat   = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat   = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )
    doc_pat = re.compile(r"\[1\]\s+(KB-\d+)")

    for block in blocks:
        q_m   = q_pat.search(block)
        a_m   = a_pat.search(block)
        doc_m = doc_pat.search(block)
        if q_m and a_m:
            predictions.append({
                "q_num":    int(q_m.group(1)),
                "question": q_m.group(2).strip(),
                "answer":   re.sub(r"\s+", " ", a_m.group(1)).strip(),
                "top_doc":  doc_m.group(1) if doc_m else "KB-014",
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions


# ── Run BERT score analysis ───────────────────────────────────────────

def run_bert_score(predictions):
    # Build IDF from all answers + KB passages combined
    all_token_lists = []
    for p in predictions:
        all_token_lists.append(tokenize(p["answer"]))
    for passage in KNOWLEDGE_BASE.values():
        all_token_lists.append(tokenize(passage))

    idf_scores = build_idf(all_token_lists)

    results = []
    for p in predictions:
        reference    = select_reference(p["question"], p["top_doc"])
        cand_tokens  = tokenize(p["answer"])
        ref_tokens   = tokenize(reference)

        P, R, F1     = bert_score_approx(cand_tokens, ref_tokens, idf_scores)

        results.append({
            "q_num":     p["q_num"],
            "question":  p["question"],
            "top_doc":   p["top_doc"],
            "precision": P,
            "recall":    R,
            "f1":        F1,
            "label":     score_label(F1),
            "reference": reference[:80] + "..." if len(reference) > 80 else reference,
        })

    return results


# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = len(results)

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: BERT Score Analysis")
    lines.append("  Script  : BERT_score3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append("  Method  : TF-IDF weighted BERTScore approximation")
    lines.append("            (no PyTorch/transformers required)")
    lines.append("=" * 70)
    lines.append("")

    lines.append("METHOD OVERVIEW")
    lines.append("-" * 70)
    lines.append("  BERTScore measures semantic similarity between a candidate")
    lines.append("  answer and a reference answer using three metrics:")
    lines.append("    Precision (P) -- candidate tokens grounded in reference")
    lines.append("    Recall    (R) -- reference tokens covered by candidate")
    lines.append("    F1            -- harmonic mean of P and R")
    lines.append("")
    lines.append("  Reference answers are drawn from the DisProt KB passages")
    lines.append("  (KB-001 through KB-014) matched to each question by the")
    lines.append("  top retrieved document ID in the prediction file.")
    lines.append("")
    lines.append("  Score interpretation (F1):")
    lines.append("    >= 0.85  EXCELLENT    >= 0.70  GOOD")
    lines.append("    >= 0.55  MODERATE     >= 0.40  WEAK     < 0.40  POOR")
    lines.append("")

    # Per-question table
    lines.append("PER-QUESTION BERT SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Label':<12}  {'RefDoc':<8}")
    lines.append("  " + "-" * 56)

    for r in results:
        lines.append(
            f"  {r['q_num']:>4}  {r['precision']:>8.4f}  "
            f"{r['recall']:>8.4f}  {r['f1']:>8.4f}  "
            f"{r['label']}  {r['top_doc']:<8}"
        )

    # Aggregate stats
    p_vals  = [r["precision"] for r in results]
    r_vals  = [r["recall"]    for r in results]
    f1_vals = [r["f1"]        for r in results]

    lines.append("")
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} "
                 f"{'Max':>8} {'Median':>8} {'P25':>8} {'P75':>8}")
    lines.append("  " + "-" * 66)
    for label, vals in [("Precision", p_vals), ("Recall", r_vals), ("F1", f1_vals)]:
        lines.append(
            f"  {label:<12} {mean(vals):>8.4f} {std(vals):>8.4f} "
            f"{min(vals):>8.4f} {max(vals):>8.4f} {median(vals):>8.4f} "
            f"{percentile(vals, 25):>8.4f} {percentile(vals, 75):>8.4f}"
        )

    # F1 distribution
    lines.append("")
    lines.append("F1 SCORE DISTRIBUTION")
    lines.append("-" * 70)
    buckets = [
        ("EXCELLENT (>=0.85)", 0.85, 1.01),
        ("GOOD      (>=0.70)", 0.70, 0.85),
        ("MODERATE  (>=0.55)", 0.55, 0.70),
        ("WEAK      (>=0.40)", 0.40, 0.55),
        ("POOR      (< 0.40)", 0.00, 0.40),
    ]
    for label, lo, hi in buckets:
        count = sum(1 for v in f1_vals if lo <= v < hi)
        bar   = "#" * int(count / n * 40)
        lines.append(f"  {label} : {count:>4} ({count/n*100:>5.1f}%)  {bar}")

    # Reference doc coverage
    lines.append("")
    lines.append("F1 BY REFERENCE DOCUMENT")
    lines.append("-" * 70)
    from collections import defaultdict
    doc_f1s = defaultdict(list)
    for r in results:
        doc_f1s[r["top_doc"]].append(r["f1"])
    lines.append(f"  {'RefDoc':<10} {'N':>4}  {'MeanF1':>8}  {'StdF1':>8}")
    lines.append("  " + "-" * 36)
    for doc, vals in sorted(doc_f1s.items(), key=lambda x: -mean(x[1])):
        lines.append(
            f"  {doc:<10} {len(vals):>4}  {mean(vals):>8.4f}  {std(vals):>8.4f}"
        )

    # Top and bottom 5
    sorted_results = sorted(results, key=lambda r: r["f1"], reverse=True)
    lines.append("")
    lines.append("TOP 5 HIGHEST F1 SCORES")
    lines.append("-" * 70)
    for r in sorted_results[:5]:
        lines.append(
            f"  Q{r['q_num']:>3}  F1={r['f1']:.4f}  P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  {r['label']}"
        )
        lines.append(f"       Q: \"{r['question'][:58]}\"")
        lines.append(f"       Ref: \"{r['reference'][:58]}\"")

    lines.append("")
    lines.append("BOTTOM 5 LOWEST F1 SCORES")
    lines.append("-" * 70)
    for r in sorted_results[-5:]:
        lines.append(
            f"  Q{r['q_num']:>3}  F1={r['f1']:.4f}  P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  {r['label']}"
        )
        lines.append(f"       Q: \"{r['question'][:58]}\"")
        lines.append(f"       Ref: \"{r['reference'][:58]}\"")

    # Precision vs Recall balance
    lines.append("")
    lines.append("PRECISION vs RECALL BALANCE")
    lines.append("-" * 70)
    p_gt_r = sum(1 for r in results if r["precision"] > r["recall"])
    r_gt_p = sum(1 for r in results if r["recall"] > r["precision"])
    equal  = sum(1 for r in results if r["precision"] == r["recall"])
    lines.append(f"  Precision > Recall : {p_gt_r:>4} ({p_gt_r/n*100:.1f}%)")
    lines.append(f"  Recall > Precision : {r_gt_p:>4} ({r_gt_p/n*100:.1f}%)")
    lines.append(f"  Equal              : {equal:>4} ({equal/n*100:.1f}%)")
    lines.append("")
    if p_gt_r > r_gt_p:
        lines.append("  System is PRECISION-DOMINANT: answers contain more of the")
        lines.append("  reference content than the reference covers of the answers.")
        lines.append("  Suggests answers are more specific than the reference passages.")
    else:
        lines.append("  System is RECALL-DOMINANT: reference covers answers well but")
        lines.append("  answers miss some reference content. Suggests the extractive")
        lines.append("  fallback is pulling relevant passages but not all needed facts.")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  High F1 scores indicate the predicted answer closely mirrors")
    lines.append("  the reference KB passage -- expected in extractive fallback")
    lines.append("  mode since answers ARE concatenated KB passages.")
    lines.append("")
    lines.append("  Questions with low F1 indicate retrieval mismatch: the top")
    lines.append("  retrieved document does not well represent the question, so")
    lines.append("  the reference-answer semantic overlap is low even though the")
    lines.append("  answer itself may be factually correct on a different topic.")
    lines.append("")
    lines.append("  For true BERTScore with contextual embeddings, install:")
    lines.append("    pip install bert-score")
    lines.append("  and replace the TF-IDF vectors with actual BERT embeddings.")
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
    print("[INFO] Computing BERT scores...\n")
    results = run_bert_score(predictions)
    write_output(results, OUT_PATH)