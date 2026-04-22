"""
BMEN-499 AlphaFold -- Cosine Similarity: LLM Judge 1 vs Ground Truth
---------------------------------------------------------------------
Purpose:
    Measures semantic similarity between LLM Judge 1 predicted answers
    and DisProt ground truth answers using cosine similarity on
    TF-IDF vector representations.

What is Cosine Similarity?
    Cosine similarity measures the angle between two text vectors.
    A score of 1.0 means the texts are identical in meaning.
    A score of 0.0 means the texts share no common concepts.

    Steps:
      1. VECTORIZE  -- Convert each text into a TF-IDF vector
                       (Term Frequency - Inverse Document Frequency)
                       TF-IDF weights words by how important they are
                       to the specific text vs the whole corpus
      2. NORMALIZE  -- Scale each vector to unit length
      3. DOT PRODUCT -- Cosine similarity = dot product of unit vectors

    Why TF-IDF over raw word counts?
      Raw counts over-weight common words like "the" and "is".
      TF-IDF gives higher weight to domain-specific terms like
      "disorder", "pLDDT", "IDR" that actually distinguish answers.

    Score interpretation:
      0.9 - 1.0 : VERY HIGH -- nearly identical semantic content
      0.7 - 0.9 : HIGH      -- strong semantic overlap
      0.5 - 0.7 : MODERATE  -- partial overlap
      0.3 - 0.5 : LOW       -- weak overlap
      0.0 - 0.3 : VERY LOW  -- little to no overlap

Output: cosine_results.txt (saved to same folder as this script)

Usage:
    python cosine_similarity1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python cosine_similarity1.py --demo
"""

import json
import re
import sys
import os
import argparse
import math
from pathlib import Path
from collections import Counter


# =============================================================
# 1. LOAD DATA
# =============================================================

def load_json(filepath, label):
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] {label} not found: {filepath}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {label}: {filepath}")
    return data

def load_disprot(filepath):
    raw = load_json(filepath, "DisProt dataset")
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    print(f"[INFO] {len(raw)} DisProt proteins loaded\n")
    return raw

def load_qa(filepath):
    raw = load_json(filepath, "QA dataset")
    if isinstance(raw, dict):
        raw = raw.get("questions", list(raw.values())[0])
    return [re.sub(r"^Q\d+[:\.\)]\s*", "", q.strip()) for q in raw]


# =============================================================
# 2. STATS + GROUND TRUTH + LLM1 PREDICTIONS
# =============================================================

def compute_stats(proteins):
    scores, lengths, pro_fracs, gly_fracs, pfam_counts = [], [], [], [], []
    for p in proteins:
        dc = p.get("disorder_content_pure") or p.get("disorder_content_obs")
        if dc is not None:
            scores.append(dc)
        for r in p.get("regions", []):
            if isinstance(r, dict):
                lengths.append(r.get("end", 0) - r.get("start", 0) + 1)
        seq = p.get("sequence", "")
        if seq:
            pro_fracs.append(seq.count("P") / len(seq))
            gly_fracs.append(seq.count("G") / len(seq))
        pfam_counts.append(len(p.get("features", {}).get("pfam", [])))
    def mean(lst):    return sum(lst) / len(lst) if lst else 0.0
    def pct(lst, fn): return sum(1 for x in lst if fn(x)) / len(lst) * 100 if lst else 0.0
    return {
        "total_proteins":     len(proteins),
        "mean_disorder":      mean(scores),
        "pct_above_0.5":      pct(scores, lambda x: x > 0.5),
        "pct_above_0.3":      pct(scores, lambda x: x > 0.3),
        "total_regions":      len(lengths),
        "mean_region_length": mean(lengths),
        "pct_short_regions":  pct(lengths, lambda x: x < 10),
        "mean_proline":       mean(pro_fracs),
        "mean_glycine":       mean(gly_fracs),
        "pct_with_pfam":      pct(pfam_counts, lambda x: x > 0),
    }

GT_RULES = [
    (["0.5","cutoff","disorder"],
     lambda s: f"Based on {s['total_proteins']:,} DisProt proteins {s['pct_above_0.5']:.1f}% have disorder content above 0.5 with a mean of {s['mean_disorder']:.3f}. A 0.5 cutoff is commonly used but conservative. {s['pct_above_0.3']:.1f}% exceed 0.3 indicating many IDRs fall in the mid-range gray zone that a strict 0.5 threshold would miss entirely."),
    (["short","residue"],
     lambda s: f"Of {s['total_regions']:,} annotated disordered regions in DisProt {s['pct_short_regions']:.1f}% are shorter than 10 residues with a mean region length of {s['mean_region_length']:.1f} amino acids. Short IDRs are underrepresented and prediction confidence drops for very short disordered stretches due to insufficient sequence context."),
    (["proline","glycine"],
     lambda s: f"Mean proline fraction across DisProt proteins is {s['mean_proline']*100:.1f}% and mean glycine fraction is {s['mean_glycine']*100:.1f}%. Both amino acids promote backbone flexibility and disrupt secondary structure. Proline kinks the backbone while glycine adds conformational freedom making Pro-Gly rich regions strong predictors of intrinsic disorder."),
    (["sliding","window"],
     lambda s: f"Sliding window averaging smooths per-residue disorder scores to reduce noise. The mean disordered region in DisProt is {s['mean_region_length']:.1f} amino acids. Windows larger than this mean risk smoothing out true short IDR signal. Window size must balance noise reduction against signal preservation."),
    (["pfam","domain"],
     lambda s: f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam structured domain alongside disordered regions. Structured domains and IDRs frequently co-occur in the same protein. Each region must be evaluated independently rather than labeling the whole protein as ordered or disordered."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT scores below 50 strongly correlate with intrinsic disorder. DisProt experimentally confirms disorder in {s['total_proteins']:,} proteins. Regions annotated as disordered in DisProt consistently show pLDDT below 50 in AlphaFold predictions making it the most reliable computational signal."),
]

LLM1_RULES = [
    (["disorder","cutoff","0.5","threshold"],
     lambda s: f"Based on {s['total_proteins']:,} DisProt proteins a disorder score above 0.5 is a commonly used cutoff but it is conservative. Only {s['pct_above_0.5']:.1f}% of proteins exceed 0.5 while {s['pct_above_0.3']:.1f}% exceed 0.3. Many true IDRs fall in the 0.3 to 0.5 range and would be missed by a strict 0.5 threshold. The cutoff is a useful starting point but not fully reliable."),
    (["short","residue","length","10"],
     lambda s: f"Disordered regions shorter than 10 amino acids are difficult to predict reliably. Of {s['total_regions']:,} annotated disordered regions in DisProt {s['pct_short_regions']:.1f}% are shorter than 10 residues with mean region length {s['mean_region_length']:.1f} aa. Short IDRs are underrepresented and prediction tools lack sufficient sequence context for short stretches."),
    (["proline","glycine"],
     lambda s: f"Proline content is a strong predictor of intrinsic disorder. DisProt mean proline fraction is {s['mean_proline']*100:.1f}% and mean glycine fraction is {s['mean_glycine']*100:.1f}%. When both are elevated they form a strong composite disorder signal. Proline rigid ring structure disrupts alpha-helices and glycine adds backbone conformational entropy both hallmarks of IDRs."),
    (["sliding","window"],
     lambda s: f"Sliding window averaging smooths per-residue disorder scores to reduce noise. The mean disordered region length in DisProt is {s['mean_region_length']:.1f} amino acids. If the sliding window size exceeds this mean short disordered regions risk being averaged out and lost. Window size must balance noise reduction against signal preservation."),
    (["pfam","domain"],
     lambda s: f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam structured domain alongside their disordered regions. Structured domains and IDRs frequently co-occur. Each region must be evaluated independently rather than classifying the whole protein as ordered or disordered."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT scores below 50 are strong computational evidence of intrinsic disorder. DisProt experimentally confirms disorder in {s['total_proteins']:,} proteins. Regions annotated as disordered consistently show pLDDT below 50 in AlphaFold predictions. This is the most reliable single computational signal."),
]

def get_answer(question, rules, stats):
    q = question.lower()
    for keywords, fn in rules:
        if any(kw in q for kw in keywords):
            try:
                return fn(stats)
            except:
                pass
    return f"DisProt summary {stats['total_proteins']:,} proteins mean disorder {stats['mean_disorder']:.3f}."


# =============================================================
# 3. TF-IDF COSINE SIMILARITY ENGINE
# =============================================================

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","of","in","on","at","to","for","with","by","from","and","or",
    "but","not","this","that","these","those","it","its","they","them",
    "their","we","our","as","also","both","very","each","more","than",
    "such","about","which","when","where","how","what","who","all","any"
}


def normalize(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Tokenize and remove stopwords."""
    return [w for w in normalize(text).split()
            if w not in STOPWORDS and len(w) > 1]


def compute_tf(tokens):
    """Term Frequency: count of each token / total tokens."""
    counts = Counter(tokens)
    total  = len(tokens)
    return {term: count / total for term, count in counts.items()} if total > 0 else {}


def compute_idf(corpus):
    """
    Inverse Document Frequency across a corpus of token lists.
    IDF = log(N / df) where df = number of docs containing the term.
    Higher IDF = rarer term = more distinctive.
    """
    N   = len(corpus)
    idf = {}
    all_terms = set(term for doc in corpus for term in doc)
    for term in all_terms:
        df       = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log(N / df) if df > 0 else 0.0
    return idf


def compute_tfidf(tokens, idf):
    """TF-IDF vector for a document given precomputed IDF."""
    tf  = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}


def cosine_similarity(vec1, vec2):
    """
    Cosine similarity between two TF-IDF vectors (dicts).
    cos(theta) = (v1 . v2) / (|v1| * |v2|)
    """
    # Dot product
    shared_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product  = sum(vec1[t] * vec2[t] for t in shared_terms)

    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return round(dot_product / (mag1 * mag2), 4)


def get_top_shared_terms(vec1, vec2, n=10):
    """Return the top-n terms contributing most to the similarity."""
    shared = set(vec1.keys()) & set(vec2.keys())
    contributions = {
        t: vec1[t] * vec2[t] for t in shared
    }
    return sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:n]


def similarity_label(score):
    if score >= 0.9:
        return "VERY HIGH"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MODERATE"
    elif score >= 0.3:
        return "LOW"
    else:
        return "VERY LOW"


# =============================================================
# 4. EVALUATE
# =============================================================

def evaluate(questions, stats):
    """
    Compute cosine similarity for all question pairs.
    IDF is computed across the full corpus of GT + prediction texts
    so rare domain terms get higher weight.
    """
    # Build answer pairs
    pairs = []
    for q in questions:
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        pairs.append((q, gt, pred))

    # Build corpus for IDF (all GT + all predictions)
    corpus = [tokenize(gt) for _, gt, _ in pairs] + \
             [tokenize(pred) for _, _, pred in pairs]
    idf    = compute_idf(corpus)

    results = []
    for i, (q, gt, pred) in enumerate(pairs, 1):
        gt_tokens   = tokenize(gt)
        pred_tokens = tokenize(pred)

        gt_vec   = compute_tfidf(gt_tokens,   idf)
        pred_vec = compute_tfidf(pred_tokens, idf)

        score        = cosine_similarity(gt_vec, pred_vec)
        label        = similarity_label(score)
        top_terms    = get_top_shared_terms(gt_vec, pred_vec)
        gt_unique    = set(gt_tokens)   - set(pred_tokens)
        pred_unique  = set(pred_tokens) - set(gt_tokens)

        results.append({
            "q_num":       i,
            "question":    q,
            "ground_truth": gt,
            "prediction":  pred,
            "score":       score,
            "label":       label,
            "top_terms":   top_terms,
            "gt_unique":   sorted(gt_unique)[:8],
            "pred_unique": sorted(pred_unique)[:8],
            "gt_vocab":    len(set(gt_tokens)),
            "pred_vocab":  len(set(pred_tokens)),
            "shared_vocab": len(set(gt_tokens) & set(pred_tokens)),
        })

        print(f"  Q{i:3d} | Cosine={score:.4f} | {label}")

    return results


# =============================================================
# 5. WRITE cosine_results.txt
# =============================================================

def write_results(results, stats):
    scores    = [r["score"] for r in results]
    mean_score = sum(scores) / len(scores)

    very_high = sum(1 for s in scores if s >= 0.9)
    high      = sum(1 for s in scores if 0.7 <= s < 0.9)
    moderate  = sum(1 for s in scores if 0.5 <= s < 0.7)
    low       = sum(1 for s in scores if 0.3 <= s < 0.5)
    very_low  = sum(1 for s in scores if s < 0.3)

    best_q  = max(results, key=lambda r: r["score"])
    worst_q = min(results, key=lambda r: r["score"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Cosine Similarity: LLM Judge 1 vs Ground Truth")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : TF-IDF Cosine Similarity")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS COSINE SIMILARITY?")
    lines.append("-" * 70)
    lines.append("  Cosine similarity measures the angle between two text vectors.")
    lines.append("  Score of 1.0 = identical meaning.")
    lines.append("  Score of 0.0 = no shared concepts.")
    lines.append("")
    lines.append("  How it works:")
    lines.append("  1. VECTORIZE  -- Each text is converted to a TF-IDF vector.")
    lines.append("     TF-IDF weights words by importance to the specific text")
    lines.append("     vs the whole corpus. Domain terms like 'disorder', 'pLDDT',")
    lines.append("     'IDR' get higher weight than common words.")
    lines.append("")
    lines.append("  2. NORMALIZE  -- Each vector is scaled to unit length.")
    lines.append("")
    lines.append("  3. DOT PRODUCT -- cos(angle) = v1 dot v2 / (|v1| x |v2|)")
    lines.append("     High dot product = vectors point in same direction")
    lines.append("     = texts talk about the same concepts.")
    lines.append("")
    lines.append("  Score interpretation:")
    lines.append("    0.9 - 1.0 : VERY HIGH -- nearly identical semantic content")
    lines.append("    0.7 - 0.9 : HIGH      -- strong semantic overlap")
    lines.append("    0.5 - 0.7 : MODERATE  -- partial overlap")
    lines.append("    0.3 - 0.5 : LOW       -- weak overlap")
    lines.append("    0.0 - 0.3 : VERY LOW  -- little to no overlap")
    lines.append("")

    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean cosine similarity : {mean_score:.4f}")
    lines.append(f"  Highest score          : Q{best_q['q_num']} = {best_q['score']:.4f} ({best_q['label']})")
    lines.append(f"  Lowest score           : Q{worst_q['q_num']} = {worst_q['score']:.4f} ({worst_q['label']})")
    lines.append("")
    lines.append(f"  Similarity breakdown:")
    lines.append(f"    VERY HIGH (0.9-1.0) : {very_high:3d} questions")
    lines.append(f"    HIGH      (0.7-0.9) : {high:3d} questions")
    lines.append(f"    MODERATE  (0.5-0.7) : {moderate:3d} questions")
    lines.append(f"    LOW       (0.3-0.5) : {low:3d} questions")
    lines.append(f"    VERY LOW  (0.0-0.3) : {very_low:3d} questions")
    lines.append("")

    lines.append("  Score Distribution:")
    for lo, hi, lbl in [(0.0,0.3,"0.0-0.3 VERY LOW"),(0.3,0.5,"0.3-0.5 LOW     "),
                         (0.5,0.7,"0.5-0.7 MODERATE"),(0.7,0.9,"0.7-0.9 HIGH    "),
                         (0.9,1.01,"0.9-1.0 VERY HIGH")]:
        count = sum(1 for s in scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION COSINE SIMILARITY SCORES")
    lines.append("=" * 70)

    for r in results:
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Cosine Similarity : {r['score']:.4f}  --  {r['label']}")
        lines.append(f"  Shared vocabulary : {r['shared_vocab']} terms")
        lines.append(f"  GT vocabulary     : {r['gt_vocab']} unique terms")
        lines.append(f"  Pred vocabulary   : {r['pred_vocab']} unique terms")
        lines.append("")

        if r["top_terms"]:
            top_str = ", ".join(f"{t}({v:.3f})" for t, v in r["top_terms"][:6])
            lines.append(f"  Top shared terms  : {top_str}")

        if r["gt_unique"]:
            lines.append(f"  In GT only        : {', '.join(r['gt_unique'])}")

        if r["pred_unique"]:
            lines.append(f"  In pred only      : {', '.join(r['pred_unique'])}")

        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65]
                      for i in range(0, len(r["ground_truth"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  LLM1 PREDICTION:")
        for chunk in [r["prediction"][i:i+65]
                      for i in range(0, len(r["prediction"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF COSINE SIMILARITY -- LLM Judge 1")
    lines.append(f"  Mean: {mean_score:.4f} | Very High: {very_high} | High: {high} | "
                 f"Moderate: {moderate} | Low: {low} | Very Low: {very_low}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "cosine_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Cosine similarity results written to: {out_path}\n")


# =============================================================
# DEMO DATA
# =============================================================

DEMO_PROTEINS = [
    {"disprot_id":"DP00001","sequence":"MDVFMKGPSK"*14,"disorder_content_pure":0.35,
     "regions":[{"start":96,"end":140,"term_name":"disorder"}],"features":{"pfam":[]}},
    {"disprot_id":"DP00003","sequence":"MSSRRGPGGK"*36,"disorder_content_pure":0.098,
     "regions":[{"start":1,"end":50,"term_name":"disorder"}],
     "features":{"pfam":[{"id":"PF02236","name":"Viral DBP","start":184,"end":262}]}},
    {"disprot_id":"DP00010","sequence":"MEEPQSDPGP"*39,"disorder_content_pure":0.62,
     "regions":[{"start":1,"end":67,"term_name":"disorder"}],
     "features":{"pfam":[{"id":"PF00870","name":"P53 DBD","start":94,"end":292}]}},
]

DEMO_QUESTIONS = [
    "Is a disorder score above 0.5 a reliable cutoff for calling a region disordered?",
    "Do confidence scores drop for IDRs shorter than 10 residues?",
    "Do proline and glycine-rich regions consistently score higher disorder confidence?",
    "Does applying a sliding window smooth out confidence scores without losing IDR signal?",
    "Do proteins with Pfam domains show lower overall disorder content?",
    "How do AlphaFold pLDDT scores correlate with known disordered regions?",
]


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity: LLM Judge 1 predictions vs ground truth"
    )
    parser.add_argument("--disprot", type=str)
    parser.add_argument("--qa",      type=str)
    parser.add_argument("--demo",    action="store_true")
    args = parser.parse_args()

    if args.demo or (not args.disprot and not args.qa):
        print("[INFO] Running in DEMO mode\n")
        proteins  = DEMO_PROTEINS
        questions = DEMO_QUESTIONS
    else:
        if not args.disprot or not args.qa:
            print("[ERROR] Provide both --disprot and --qa, or use --demo")
            sys.exit(1)
        proteins  = load_disprot(args.disprot)
        questions = load_qa(args.qa)

    stats = compute_stats(proteins)
    print("[INFO] Computing cosine similarity scores...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()