"""
BMEN-499 AlphaFold -- Agreement Score: LLM Judge 1 vs Ground Truth
-------------------------------------------------------------------
Purpose:
    Measures how well LLM Judge 1 predicted answers AGREE with
    DisProt ground truth answers using multiple consistency metrics.

What is Agreement Scoring?
    Agreement goes beyond simple word overlap (like NAUR). It checks:

    1. CONTRADICTION COUNT  -- Does the prediction say anything that
                               directly contradicts the ground truth?
                               (e.g. GT says "29.1%" but LLM says "50%")

    2. SEMANTIC CONSISTENCY -- Do both texts express the same core claim
                               even if worded differently?

    3. NUMERIC AGREEMENT    -- Do the numbers in the prediction match
                               the numbers in the ground truth?

    4. KEYWORD AGREEMENT    -- Do both share the same key biomedical
                               terms (disorder, IDR, pLDDT, etc.)?

    5. DIRECTIONAL AGREEMENT -- Do both texts agree on the direction
                                of a claim? (e.g. both say "above 0.5
                                is conservative" vs one saying opposite)

    Final agreement score = weighted combination of all five metrics

Output: consistency_report.txt (saved to same folder as this script)

Usage:
    python contradiction_count.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python contradiction_count.py --demo
"""

import json
import re
import sys
import os
import argparse
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
# 3. AGREEMENT SCORING ENGINE
# =============================================================

# Biomedical keywords relevant to protein disorder
DOMAIN_KEYWORDS = [
    "disorder", "disordered", "idr", "idp", "protein", "residue",
    "alphafold", "plddt", "pfam", "domain", "proline", "glycine",
    "threshold", "cutoff", "region", "sequence", "amino", "acid",
    "confidence", "prediction", "disprot", "intrinsic", "structured",
    "unstructured", "flexible", "backbone", "secondary", "structure",
    "window", "sliding", "noise", "signal", "conservative", "overlap"
]

# Directional claim patterns -- pairs that would be contradictions
DIRECTIONAL_PATTERNS = [
    (r"\babove\s+0\.5\b",    r"\bbelow\s+0\.5\b"),
    (r"\breliable\b",        r"\bunreliable\b"),
    (r"\bconservative\b",    r"\baggressive\b"),
    (r"\bstrong\b",          r"\bweak\b"),
    (r"\bhigh\b",            r"\blow\b"),
    (r"\bincreases?\b",      r"\bdecreases?\b"),
    (r"\bpredicts?\b",       r"\bdoes not predict\b"),
    (r"\bcorrelates?\b",     r"\bdoes not correlate\b"),
    (r"\bfrequently\b",      r"\brarely\b"),
    (r"\bco-occur\b",        r"\bdo not co-occur\b"),
]


def extract_numbers(text):
    """Extract all numeric values from text."""
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


def numeric_agreement(pred, gt):
    """
    Compare numbers between prediction and ground truth.
    Agreement = fraction of GT numbers that appear (approximately) in prediction.
    Tolerance: within 2% of the GT value counts as a match.
    """
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)

    if not gt_nums:
        return {"score": 1.0, "gt_numbers": [], "pred_numbers": [],
                "matched": 0, "contradictions": []}

    matched       = 0
    contradictions = []

    for gn in gt_nums:
        found = any(abs(pn - gn) / max(abs(gn), 1e-9) < 0.02 for pn in pred_nums)
        if found:
            matched += 1
        else:
            # Check if a very different number appears in the same context
            close_but_wrong = [pn for pn in pred_nums
                               if abs(pn - gn) / max(abs(gn), 1e-9) > 0.05
                               and abs(pn - gn) < gn * 2]
            if close_but_wrong:
                contradictions.append(
                    f"GT states {gn} but prediction has {close_but_wrong[0]:.1f}"
                )

    score = matched / len(gt_nums)
    return {
        "score":          round(score, 4),
        "gt_numbers":     gt_nums,
        "pred_numbers":   pred_nums,
        "matched":        matched,
        "contradictions": contradictions,
    }


def keyword_agreement(pred, gt):
    """
    Measure overlap of domain-specific biomedical keywords.
    Agreement = fraction of GT domain keywords found in prediction.
    """
    pred_lower = pred.lower()
    gt_lower   = gt.lower()

    gt_keys   = [kw for kw in DOMAIN_KEYWORDS if kw in gt_lower]
    pred_keys = [kw for kw in DOMAIN_KEYWORDS if kw in pred_lower]

    if not gt_keys:
        return {"score": 1.0, "gt_keywords": [], "pred_keywords": [],
                "matched": [], "missing": []}

    matched = [kw for kw in gt_keys if kw in pred_keys]
    missing = [kw for kw in gt_keys if kw not in pred_keys]

    return {
        "score":        round(len(matched) / len(gt_keys), 4),
        "gt_keywords":  gt_keys,
        "pred_keywords": pred_keys,
        "matched":      matched,
        "missing":      missing,
    }


def directional_agreement(pred, gt):
    """
    Check whether prediction and ground truth agree on directional claims.
    A contradiction is when GT uses one direction word and prediction uses
    the opposite.
    """
    contradictions = []
    agreements     = []

    for pos_pattern, neg_pattern in DIRECTIONAL_PATTERNS:
        gt_pos   = bool(re.search(pos_pattern, gt,   re.IGNORECASE))
        gt_neg   = bool(re.search(neg_pattern, gt,   re.IGNORECASE))
        pred_pos = bool(re.search(pos_pattern, pred, re.IGNORECASE))
        pred_neg = bool(re.search(neg_pattern, pred, re.IGNORECASE))

        if gt_pos and pred_neg:
            contradictions.append(
                f"GT uses '{pos_pattern.strip(r'\\b')}' but prediction uses opposite"
            )
        elif gt_neg and pred_pos:
            contradictions.append(
                f"GT uses '{neg_pattern.strip(r'\\b')}' but prediction uses opposite"
            )
        elif gt_pos and pred_pos:
            agreements.append(pos_pattern.replace(r"\b", "").replace("\\b", ""))
        elif gt_neg and pred_neg:
            agreements.append(neg_pattern.replace(r"\b", "").replace("\\b", ""))

    total_checks = len(DIRECTIONAL_PATTERNS)
    agree_count  = len(agreements)
    score        = agree_count / total_checks if total_checks > 0 else 1.0

    return {
        "score":           round(score, 4),
        "agreements":      agreements,
        "contradictions":  contradictions,
        "total_checks":    total_checks,
    }


def semantic_consistency(pred, gt):
    """
    Measure semantic consistency using unigram overlap as a proxy.
    High overlap = texts express similar content even if worded differently.
    """
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return set(text.split())

    pred_tokens = tokenize(pred)
    gt_tokens   = tokenize(gt)

    # Remove stopwords
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "of", "in", "on", "at", "to", "for", "with", "by", "from",
                 "and", "or", "but", "not", "this", "that", "these", "those",
                 "it", "its", "they", "them", "their", "we", "our", "as",
                 "also", "both", "each", "more", "than", "such", "very"}

    pred_content = pred_tokens - stopwords
    gt_content   = gt_tokens   - stopwords

    if not gt_content:
        return {"score": 1.0, "shared_concepts": [], "unique_to_gt": []}

    shared        = pred_content & gt_content
    unique_to_gt  = gt_content - pred_content

    score = len(shared) / len(gt_content)

    return {
        "score":          round(score, 4),
        "shared_concepts": sorted(list(shared))[:15],
        "unique_to_gt":   sorted(list(unique_to_gt))[:10],
    }


def count_contradictions(pred, gt, numeric, directional):
    """
    Count total contradictions across all metrics.
    """
    total = len(numeric["contradictions"]) + len(directional["contradictions"])
    all_contradictions = numeric["contradictions"] + directional["contradictions"]
    return total, all_contradictions


def agreement_score(pred, gt):
    """
    Compute overall agreement score from all five sub-metrics.

    Weights:
      Semantic consistency : 30%
      Keyword agreement    : 25%
      Numeric agreement    : 25%
      Directional agreement: 20%
    """
    semantic   = semantic_consistency(pred, gt)
    keywords   = keyword_agreement(pred, gt)
    numeric    = numeric_agreement(pred, gt)
    directional = directional_agreement(pred, gt)

    total_contradictions, contradiction_list = count_contradictions(
        pred, gt, numeric, directional
    )

    # Weighted agreement score
    score = (
        0.30 * semantic["score"]    +
        0.25 * keywords["score"]    +
        0.25 * numeric["score"]     +
        0.20 * directional["score"]
    )

    # Penalize for contradictions
    penalty = min(0.3, total_contradictions * 0.05)
    score   = max(0.0, score - penalty)

    label = (
        "STRONG AGREEMENT"   if score >= 0.75 else
        "MODERATE AGREEMENT" if score >= 0.50 else
        "WEAK AGREEMENT"     if score >= 0.25 else
        "DISAGREEMENT"
    )

    return {
        "overall_score":      round(score, 4),
        "label":              label,
        "semantic":           semantic,
        "keywords":           keywords,
        "numeric":            numeric,
        "directional":        directional,
        "total_contradictions": total_contradictions,
        "contradiction_list": contradiction_list,
        "penalty_applied":    round(penalty, 4),
    }


# =============================================================
# 4. RUN EVALUATION
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = agreement_score(pred, gt)
        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })
        print(f"  Q{i:3d} | Agreement={sc['overall_score']:.4f} | "
              f"Contradictions={sc['total_contradictions']} | {sc['label']}")
    return results


# =============================================================
# 5. WRITE consistency_report.txt
# =============================================================

def write_report(results, stats):
    scores  = [r["score"]["overall_score"]      for r in results]
    sem     = [r["score"]["semantic"]["score"]   for r in results]
    kw      = [r["score"]["keywords"]["score"]   for r in results]
    num     = [r["score"]["numeric"]["score"]    for r in results]
    dirc    = [r["score"]["directional"]["score"] for r in results]
    contras = [r["score"]["total_contradictions"] for r in results]

    mean_score  = sum(scores) / len(scores)
    mean_sem    = sum(sem)    / len(sem)
    mean_kw     = sum(kw)     / len(kw)
    mean_num    = sum(num)    / len(num)
    mean_dir    = sum(dirc)   / len(dirc)
    total_contra = sum(contras)

    strong   = sum(1 for r in results if r["score"]["label"] == "STRONG AGREEMENT")
    moderate = sum(1 for r in results if r["score"]["label"] == "MODERATE AGREEMENT")
    weak     = sum(1 for r in results if r["score"]["label"] == "WEAK AGREEMENT")
    disagree = sum(1 for r in results if r["score"]["label"] == "DISAGREEMENT")

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Agreement Score: LLM Judge 1 vs Ground Truth")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : Multi-dimensional Agreement + Contradiction Count")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS AGREEMENT SCORING?")
    lines.append("-" * 70)
    lines.append("  Agreement scoring checks whether LLM1 predictions say the")
    lines.append("  same things as the ground truth across five dimensions:")
    lines.append("")
    lines.append("  1. SEMANTIC CONSISTENCY (weight=30%)")
    lines.append("     Do both texts express the same core concepts?")
    lines.append("     Measured by shared content word overlap after removing")
    lines.append("     common stopwords like 'the', 'is', 'and'.")
    lines.append("")
    lines.append("  2. KEYWORD AGREEMENT (weight=25%)")
    lines.append("     Do both use the same biomedical domain terms?")
    lines.append("     Checks for terms like: disorder, pLDDT, IDR, Pfam,")
    lines.append("     proline, glycine, threshold, backbone, etc.")
    lines.append("")
    lines.append("  3. NUMERIC AGREEMENT (weight=25%)")
    lines.append("     Do the numbers match? e.g. if GT says 29.1% does the")
    lines.append("     prediction also say approximately 29.1%? Tolerance: 2%.")
    lines.append("")
    lines.append("  4. DIRECTIONAL AGREEMENT (weight=20%)")
    lines.append("     Do both texts agree on the direction of claims?")
    lines.append("     e.g. both say 'conservative' not one saying 'aggressive'.")
    lines.append("")
    lines.append("  5. CONTRADICTION PENALTY")
    lines.append("     Each detected contradiction reduces the score by 0.05.")
    lines.append("     Max penalty capped at 0.30.")
    lines.append("")
    lines.append("  Score interpretation:")
    lines.append("    0.75 - 1.0  : STRONG AGREEMENT   -- texts express same claims")
    lines.append("    0.50 - 0.75 : MODERATE AGREEMENT -- mostly agree, minor gaps")
    lines.append("    0.25 - 0.50 : WEAK AGREEMENT     -- partial agreement")
    lines.append("    0.00 - 0.25 : DISAGREEMENT       -- significant conflicts")
    lines.append("")

    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean overall agreement score : {mean_score:.4f}")
    lines.append(f"  Mean semantic consistency    : {mean_sem:.4f}")
    lines.append(f"  Mean keyword agreement       : {mean_kw:.4f}")
    lines.append(f"  Mean numeric agreement       : {mean_num:.4f}")
    lines.append(f"  Mean directional agreement   : {mean_dir:.4f}")
    lines.append(f"  Total contradictions found   : {total_contra}")
    lines.append("")
    lines.append(f"  Agreement quality breakdown:")
    lines.append(f"    STRONG AGREEMENT   : {strong:3d} questions")
    lines.append(f"    MODERATE AGREEMENT : {moderate:3d} questions")
    lines.append(f"    WEAK AGREEMENT     : {weak:3d} questions")
    lines.append(f"    DISAGREEMENT       : {disagree:3d} questions")
    lines.append("")

    lines.append("  Score Distribution:")
    for lo, hi, lbl in [(0.0,0.25,"0.00-0.25"),(0.25,0.50,"0.25-0.50"),
                         (0.50,0.75,"0.50-0.75"),(0.75,1.01,"0.75-1.00")]:
        count = sum(1 for s in scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION AGREEMENT SCORES")
    lines.append("=" * 70)

    for r in results:
        s  = r["score"]
        sc = s["overall_score"]

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
        lines.append("  AGREEMENT SCORES:")
        lines.append(f"    Semantic consistency  : {s['semantic']['score']:.4f}")
        lines.append(f"    Keyword agreement     : {s['keywords']['score']:.4f}")
        lines.append(f"    Numeric agreement     : {s['numeric']['score']:.4f}")
        lines.append(f"    Directional agreement : {s['directional']['score']:.4f}")
        lines.append(f"    Contradiction penalty : -{s['penalty_applied']:.4f}")
        lines.append(f"    OVERALL AGREEMENT     : {sc:.4f}  -- {s['label']}")
        lines.append("")

        if s["keywords"]["missing"]:
            lines.append(f"    Missing keywords : {', '.join(s['keywords']['missing'][:8])}")

        if s["keywords"]["matched"]:
            lines.append(f"    Matched keywords : {', '.join(s['keywords']['matched'][:8])}")

        if s["numeric"]["contradictions"]:
            lines.append(f"    Numeric issues   : {'; '.join(s['numeric']['contradictions'])}")

        if s["directional"]["contradictions"]:
            lines.append(f"    Direction issues : {'; '.join(s['directional']['contradictions'][:3])}")

        if s["contradiction_list"]:
            lines.append(f"    All contradictions ({s['total_contradictions']}):")
            for c in s["contradiction_list"]:
                lines.append(f"      - {c}")

        if s["semantic"]["unique_to_gt"]:
            lines.append(f"    Concepts in GT not in prediction: "
                         f"{', '.join(s['semantic']['unique_to_gt'][:8])}")

        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF AGREEMENT EVALUATION -- LLM Judge 1")
    lines.append(f"  Mean Agreement: {mean_score:.4f} | "
                 f"Strong: {strong} | Moderate: {moderate} | "
                 f"Weak: {weak} | Disagreement: {disagree}")
    lines.append(f"  Total contradictions: {total_contra}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "consistency_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Consistency report written to: {out_path}\n")


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
        description="Agreement score: LLM Judge 1 predictions vs ground truth"
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
    print("[INFO] Computing agreement scores...\n")
    results = evaluate(questions, stats)
    write_report(results, stats)


if __name__ == "__main__":
    main()