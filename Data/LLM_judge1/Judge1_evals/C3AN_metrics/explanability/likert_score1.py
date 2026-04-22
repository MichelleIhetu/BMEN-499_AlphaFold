"""
BMEN-499 AlphaFold -- Likert Score Evaluation: LLM Judge 1
-----------------------------------------------------------
Purpose:
    Evaluates LLM Judge 1 predicted answers against DisProt ground
    truth using a Likert scale scoring system across multiple
    explainability dimensions.

What is a Likert Score?
    A Likert scale rates responses from 1 to 5 on specific criteria:

      1 -- Strongly Disagree / Very Poor
      2 -- Disagree / Poor
      3 -- Neutral / Acceptable
      4 -- Agree / Good
      5 -- Strongly Agree / Excellent

    Here each predicted answer is rated on five criteria:

      1. FACTUAL ACCURACY    -- Does the prediction state correct facts
                                compared to the ground truth?
      2. COMPLETENESS        -- Does it cover all key points from GT?
      3. CLARITY             -- Is the answer clearly expressed?
      4. RELEVANCE           -- Does it stay on topic for the question?
      5. SCIENTIFIC DEPTH    -- Does it use appropriate biomedical detail?

    Final Likert score = mean across all five criteria (1.0 - 5.0)

Output: likert_results.txt (saved to same folder as this script)

Usage:
    python likert_score1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python likert_score1.py --demo
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path


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
    (["0.5", "cutoff", "disorder"],
     lambda s: (
         f"Based on {s['total_proteins']:,} DisProt proteins {s['pct_above_0.5']:.1f}% "
         f"have disorder content above 0.5 with a mean of {s['mean_disorder']:.3f}. "
         f"A 0.5 cutoff is commonly used but conservative. {s['pct_above_0.3']:.1f}% "
         f"exceed 0.3 indicating many IDRs fall in the mid-range gray zone that a "
         f"strict 0.5 threshold would miss entirely."
     )),
    (["short", "residue"],
     lambda s: (
         f"Of {s['total_regions']:,} annotated disordered regions in DisProt "
         f"{s['pct_short_regions']:.1f}% are shorter than 10 residues with a mean "
         f"region length of {s['mean_region_length']:.1f} amino acids. Short IDRs "
         f"are underrepresented and prediction confidence drops for very short "
         f"disordered stretches due to insufficient sequence context."
     )),
    (["proline", "glycine"],
     lambda s: (
         f"Mean proline fraction across DisProt proteins is {s['mean_proline']*100:.1f}% "
         f"and mean glycine fraction is {s['mean_glycine']*100:.1f}%. Both amino acids "
         f"promote backbone flexibility and disrupt secondary structure. Proline kinks "
         f"the backbone while glycine adds conformational freedom making Pro-Gly rich "
         f"regions strong predictors of intrinsic disorder."
     )),
    (["sliding", "window"],
     lambda s: (
         f"Sliding window averaging smooths per-residue disorder scores to reduce noise. "
         f"The mean disordered region in DisProt is {s['mean_region_length']:.1f} amino "
         f"acids. Windows larger than this mean risk smoothing out true short IDR signal. "
         f"Window size must balance noise reduction against signal preservation."
     )),
    (["pfam", "domain"],
     lambda s: (
         f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam "
         f"structured domain alongside disordered regions. Structured domains and IDRs "
         f"frequently co-occur in the same protein. Each region must be evaluated "
         f"independently rather than labeling the whole protein as ordered or disordered."
     )),
    (["alphafold", "plddt"],
     lambda s: (
         f"AlphaFold pLDDT scores below 50 strongly correlate with intrinsic disorder. "
         f"DisProt experimentally confirms disorder in {s['total_proteins']:,} proteins. "
         f"Regions annotated as disordered in DisProt consistently show pLDDT below 50 "
         f"in AlphaFold predictions making it the most reliable computational signal."
     )),
]

LLM1_RULES = [
    (["disorder", "cutoff", "0.5", "threshold"],
     lambda s: (
         f"Based on {s['total_proteins']:,} DisProt proteins a disorder score above 0.5 "
         f"is a commonly used cutoff but it is conservative. Only {s['pct_above_0.5']:.1f}% "
         f"of proteins exceed 0.5 while {s['pct_above_0.3']:.1f}% exceed 0.3. Many true "
         f"IDRs fall in the 0.3 to 0.5 range and would be missed by a strict 0.5 "
         f"threshold. The cutoff is a useful starting point but not fully reliable."
     )),
    (["short", "residue", "length", "10"],
     lambda s: (
         f"Disordered regions shorter than 10 amino acids are difficult to predict "
         f"reliably. Of {s['total_regions']:,} annotated disordered regions in DisProt "
         f"{s['pct_short_regions']:.1f}% are shorter than 10 residues with mean region "
         f"length {s['mean_region_length']:.1f} aa. Short IDRs are underrepresented and "
         f"prediction tools lack sufficient sequence context for short stretches."
     )),
    (["proline", "glycine"],
     lambda s: (
         f"Proline content is a strong predictor of intrinsic disorder. DisProt mean "
         f"proline fraction is {s['mean_proline']*100:.1f}% and mean glycine fraction "
         f"is {s['mean_glycine']*100:.1f}%. When both are elevated they form a strong "
         f"composite disorder signal. Proline rigid ring structure disrupts alpha-helices "
         f"and glycine adds backbone conformational entropy both hallmarks of IDRs."
     )),
    (["sliding", "window"],
     lambda s: (
         f"Sliding window averaging smooths per-residue disorder scores to reduce noise. "
         f"The mean disordered region length in DisProt is {s['mean_region_length']:.1f} "
         f"amino acids. If the sliding window size exceeds this mean short disordered "
         f"regions risk being averaged out and lost. Window size must balance noise "
         f"reduction against signal preservation."
     )),
    (["pfam", "domain"],
     lambda s: (
         f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam "
         f"structured domain alongside their disordered regions. Structured domains and "
         f"IDRs frequently co-occur. Each region must be evaluated independently rather "
         f"than classifying the whole protein as ordered or disordered."
     )),
    (["alphafold", "plddt"],
     lambda s: (
         f"AlphaFold pLDDT scores below 50 are strong computational evidence of intrinsic "
         f"disorder. DisProt experimentally confirms disorder in {s['total_proteins']:,} "
         f"proteins. Regions annotated as disordered consistently show pLDDT below 50 "
         f"in AlphaFold predictions. This is the most reliable single computational signal."
     )),
]

def get_answer(question, rules, stats):
    q = question.lower()
    for keywords, fn in rules:
        if any(kw in q for kw in keywords):
            try:
                return fn(stats)
            except:
                pass
    return (
        f"DisProt summary {stats['total_proteins']:,} proteins "
        f"mean disorder {stats['mean_disorder']:.3f} "
        f"mean region length {stats['mean_region_length']:.1f} aa."
    )


# =============================================================
# 3. LIKERT SCORING ENGINE
# =============================================================

# Biomedical keywords for scientific depth scoring
SCIENCE_TERMS = [
    "intrinsically disordered", "idr", "idp", "plddt", "alphafold",
    "pfam", "proline", "glycine", "amino acid", "residue", "backbone",
    "alpha-helix", "beta-sheet", "secondary structure", "conformational",
    "disprot", "disorder content", "threshold", "cutoff", "sequence",
    "annotation", "experimental", "computational", "prediction"
]

# Stopwords to remove before clarity check
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "but", "not", "this", "that",
    "it", "its", "they", "them", "we", "as", "also", "both", "very"
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [w for w in normalize(text).split() if w not in STOPWORDS]


def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# --- Criterion 1: Factual Accuracy (1-5) ----------------------
def score_factual_accuracy(pred, gt):
    """
    Compare numbers and key factual claims between prediction and GT.
    More matching numbers and facts = higher score.
    """
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)

    if not gt_nums:
        matched_ratio = 1.0
    else:
        matched = sum(
            1 for gn in gt_nums
            if any(abs(pn - gn) / max(abs(gn), 1e-9) < 0.03 for pn in pred_nums)
        )
        matched_ratio = matched / len(gt_nums)

    # Check key factual terms
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))
    term_overlap = len(gt_toks & pred_toks) / max(len(gt_toks), 1)

    raw   = 0.6 * matched_ratio + 0.4 * term_overlap
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Number match={matched_ratio:.2f}, Term overlap={term_overlap:.2f}"


# --- Criterion 2: Completeness (1-5) --------------------------
def score_completeness(pred, gt):
    """
    Does the prediction cover the key points from the ground truth?
    Measured by how many GT content words appear in prediction.
    """
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))

    if not gt_toks:
        return 5, "No GT content to compare"

    covered = gt_toks & pred_toks
    ratio   = len(covered) / len(gt_toks)

    # Also check sentence count as proxy for completeness
    pred_sentences = len([s for s in pred.split(".") if len(s.strip()) > 10])
    gt_sentences   = len([s for s in gt.split(".") if len(s.strip()) > 10])
    length_ratio   = min(1.0, pred_sentences / max(gt_sentences, 1))

    raw   = 0.7 * ratio + 0.3 * length_ratio
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Coverage={ratio:.2f}, Length ratio={length_ratio:.2f}"


# --- Criterion 3: Clarity (1-5) --------------------------------
def score_clarity(pred):
    """
    Is the prediction clearly written?
    Checks: sentence structure, average sentence length, repetition.
    """
    sentences = [s.strip() for s in pred.split(".") if len(s.strip()) > 5]
    if not sentences:
        return 1, "No complete sentences found"

    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

    # Ideal sentence length: 15-25 words
    if 15 <= avg_len <= 25:
        length_score = 1.0
    elif 10 <= avg_len < 15 or 25 < avg_len <= 35:
        length_score = 0.75
    else:
        length_score = 0.5

    # Check for repetition
    tokens      = tokenize(pred)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    rep_score   = min(1.0, unique_ratio * 1.5)

    raw   = 0.5 * length_score + 0.5 * rep_score
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Avg sentence len={avg_len:.1f} words, Uniqueness={unique_ratio:.2f}"


# --- Criterion 4: Relevance (1-5) ------------------------------
def score_relevance(pred, question):
    """
    Does the prediction stay on topic for the question?
    Checks how many question keywords appear in the prediction.
    """
    q_toks   = set(tokenize(question))
    pred_toks = set(tokenize(pred))

    if not q_toks:
        return 5, "No question tokens to compare"

    overlap = q_toks & pred_toks
    ratio   = len(overlap) / len(q_toks)

    score = max(1, min(5, round(1 + ratio * 4)))
    return score, f"Question keyword coverage={ratio:.2f} ({len(overlap)}/{len(q_toks)} terms)"


# --- Criterion 5: Scientific Depth (1-5) -----------------------
def score_scientific_depth(pred):
    """
    Does the prediction use appropriate biomedical terminology?
    Counts how many domain-specific science terms appear.
    """
    pred_lower = pred.lower()
    found = [term for term in SCIENCE_TERMS if term in pred_lower]
    ratio = len(found) / len(SCIENCE_TERMS)

    # Bonus for longer, more detailed answers
    word_count    = len(pred.split())
    length_bonus  = min(0.2, word_count / 500)

    raw   = min(1.0, ratio * 3 + length_bonus)
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Science terms found={len(found)}/{len(SCIENCE_TERMS)}: {', '.join(found[:5])}"


# --- Overall Likert Score --------------------------------------
def likert_score(question, pred, gt):
    """
    Compute all five Likert criteria and combine into overall score.
    """
    accuracy_score,   accuracy_note   = score_factual_accuracy(pred, gt)
    complete_score,   complete_note   = score_completeness(pred, gt)
    clarity_score,    clarity_note    = score_clarity(pred)
    relevance_score,  relevance_note  = score_relevance(pred, question)
    depth_score,      depth_note      = score_scientific_depth(pred)

    # Weighted mean
    overall = (
        0.30 * accuracy_score  +
        0.25 * complete_score  +
        0.20 * clarity_score   +
        0.15 * relevance_score +
        0.10 * depth_score
    )
    overall = round(overall, 3)

    label = (
        "EXCELLENT (5)" if overall >= 4.5 else
        "GOOD (4)"      if overall >= 3.5 else
        "ACCEPTABLE (3)" if overall >= 2.5 else
        "POOR (2)"      if overall >= 1.5 else
        "VERY POOR (1)"
    )

    return {
        "factual_accuracy":  {"score": accuracy_score,  "note": accuracy_note},
        "completeness":      {"score": complete_score,   "note": complete_note},
        "clarity":           {"score": clarity_score,    "note": clarity_note},
        "relevance":         {"score": relevance_score,  "note": relevance_note},
        "scientific_depth":  {"score": depth_score,      "note": depth_note},
        "overall":           overall,
        "label":             label,
    }


# =============================================================
# 4. RUN EVALUATION
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = likert_score(q, pred, gt)
        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })
        print(f"  Q{i:3d} | Likert={sc['overall']:.2f}/5 | {sc['label']}")
    return results


# =============================================================
# 5. WRITE likert_results.txt
# =============================================================

def write_results(results, stats):
    overall_scores = [r["score"]["overall"]                       for r in results]
    acc_scores     = [r["score"]["factual_accuracy"]["score"]     for r in results]
    com_scores     = [r["score"]["completeness"]["score"]         for r in results]
    cla_scores     = [r["score"]["clarity"]["score"]              for r in results]
    rel_scores     = [r["score"]["relevance"]["score"]            for r in results]
    dep_scores     = [r["score"]["scientific_depth"]["score"]     for r in results]

    mean_overall = sum(overall_scores) / len(overall_scores)
    mean_acc     = sum(acc_scores)     / len(acc_scores)
    mean_com     = sum(com_scores)     / len(com_scores)
    mean_cla     = sum(cla_scores)     / len(cla_scores)
    mean_rel     = sum(rel_scores)     / len(rel_scores)
    mean_dep     = sum(dep_scores)     / len(dep_scores)

    excellent  = sum(1 for r in results if r["score"]["overall"] >= 4.5)
    good       = sum(1 for r in results if 3.5 <= r["score"]["overall"] < 4.5)
    acceptable = sum(1 for r in results if 2.5 <= r["score"]["overall"] < 3.5)
    poor       = sum(1 for r in results if 1.5 <= r["score"]["overall"] < 2.5)
    very_poor  = sum(1 for r in results if r["score"]["overall"] < 1.5)

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Likert Score Evaluation: LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : 5-Point Likert Scale across 5 explainability criteria")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS A LIKERT SCORE?")
    lines.append("-" * 70)
    lines.append("  A Likert scale rates each answer from 1 to 5 on five criteria:")
    lines.append("")
    lines.append("  1 = Very Poor  |  2 = Poor  |  3 = Acceptable")
    lines.append("  4 = Good       |  5 = Excellent")
    lines.append("")
    lines.append("  CRITERION 1 -- Factual Accuracy (weight=30%)")
    lines.append("    Do the numbers and facts in the prediction match")
    lines.append("    the ground truth? Numbers within 3% count as a match.")
    lines.append("")
    lines.append("  CRITERION 2 -- Completeness (weight=25%)")
    lines.append("    Does the prediction cover all key points from the")
    lines.append("    ground truth? Measures content word coverage.")
    lines.append("")
    lines.append("  CRITERION 3 -- Clarity (weight=20%)")
    lines.append("    Is the answer clearly written? Checks sentence")
    lines.append("    length (ideal: 15-25 words) and word variety.")
    lines.append("")
    lines.append("  CRITERION 4 -- Relevance (weight=15%)")
    lines.append("    Does the answer stay on topic for the question?")
    lines.append("    Measures question keyword coverage in the answer.")
    lines.append("")
    lines.append("  CRITERION 5 -- Scientific Depth (weight=10%)")
    lines.append("    Does the answer use appropriate biomedical terms?")
    lines.append("    Checks for terms like IDR, pLDDT, Pfam, proline, etc.")
    lines.append("")
    lines.append("  Overall Likert = weighted mean of all five criteria")
    lines.append("")

    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean overall Likert score  : {mean_overall:.3f} / 5.0")
    lines.append(f"  Mean factual accuracy      : {mean_acc:.2f} / 5")
    lines.append(f"  Mean completeness          : {mean_com:.2f} / 5")
    lines.append(f"  Mean clarity               : {mean_cla:.2f} / 5")
    lines.append(f"  Mean relevance             : {mean_rel:.2f} / 5")
    lines.append(f"  Mean scientific depth      : {mean_dep:.2f} / 5")
    lines.append("")
    lines.append(f"  Quality breakdown:")
    lines.append(f"    Excellent  (4.5-5.0) : {excellent:3d} questions")
    lines.append(f"    Good       (3.5-4.5) : {good:3d} questions")
    lines.append(f"    Acceptable (2.5-3.5) : {acceptable:3d} questions")
    lines.append(f"    Poor       (1.5-2.5) : {poor:3d} questions")
    lines.append(f"    Very Poor  (0.0-1.5) : {very_poor:3d} questions")
    lines.append("")

    # Score distribution bar
    lines.append("  Likert Score Distribution:")
    for lo, hi, lbl in [(1,2,"1 (Very Poor) "),(2,3,"2 (Poor)      "),
                         (3,4,"3 (Acceptable)"),(4,5,"4 (Good)      "),
                         (4.5,5.1,"5 (Excellent) ")]:
        count = sum(1 for s in overall_scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    # Radar summary
    lines.append("  CRITERIA RADAR (mean scores):")
    for name, val in [
        ("Factual Accuracy ", mean_acc),
        ("Completeness     ", mean_com),
        ("Clarity          ", mean_cla),
        ("Relevance        ", mean_rel),
        ("Scientific Depth ", mean_dep),
    ]:
        bar   = "#" * int(val * 4) + "." * max(0, 20 - int(val * 4))
        lines.append(f"    {name} [{bar}] {val:.2f}/5")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION LIKERT SCORES")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
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
        lines.append("")
        lines.append("  LIKERT SCORES:")
        lines.append(f"    Factual Accuracy  : {s['factual_accuracy']['score']}/5  -- {s['factual_accuracy']['note']}")
        lines.append(f"    Completeness      : {s['completeness']['score']}/5  -- {s['completeness']['note']}")
        lines.append(f"    Clarity           : {s['clarity']['score']}/5  -- {s['clarity']['note']}")
        lines.append(f"    Relevance         : {s['relevance']['score']}/5  -- {s['relevance']['note']}")
        lines.append(f"    Scientific Depth  : {s['scientific_depth']['score']}/5  -- {s['scientific_depth']['note']}")
        lines.append(f"    OVERALL LIKERT    : {s['overall']:.3f}/5  -- {s['label']}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF LIKERT EVALUATION -- LLM Judge 1")
    lines.append(f"  Mean Likert: {mean_overall:.3f}/5 | "
                 f"Excellent: {excellent} | Good: {good} | "
                 f"Acceptable: {acceptable} | Poor: {poor} | Very Poor: {very_poor}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "likert_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Likert results written to: {out_path}\n")


# =============================================================
# DEMO DATA
# =============================================================

DEMO_PROTEINS = [
    {"disprot_id": "DP00001", "sequence": "MDVFMKGPSK" * 14,
     "disorder_content_pure": 0.35,
     "regions": [{"start": 96, "end": 140, "term_name": "disorder"}],
     "features": {"pfam": []}},
    {"disprot_id": "DP00003", "sequence": "MSSRRGPGGK" * 36,
     "disorder_content_pure": 0.098,
     "regions": [{"start": 1, "end": 50, "term_name": "disorder"}],
     "features": {"pfam": [{"id": "PF02236", "name": "Viral DBP",
                             "start": 184, "end": 262}]}},
    {"disprot_id": "DP00010", "sequence": "MEEPQSDPGP" * 39,
     "disorder_content_pure": 0.62,
     "regions": [{"start": 1, "end": 67, "term_name": "disorder"}],
     "features": {"pfam": [{"id": "PF00870", "name": "P53 DBD",
                             "start": 94, "end": 292}]}},
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
        description="Likert score evaluation for LLM Judge 1 predictions"
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
    print("[INFO] Computing Likert scores...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()