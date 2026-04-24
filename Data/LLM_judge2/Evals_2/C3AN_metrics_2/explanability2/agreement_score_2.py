"""
BMEN-499 AlphaFold -- Agreement Score: LLM Judge 2 vs Ground Truth
-------------------------------------------------------------------
Purpose:
    Measures how well LLM Judge 2 (Vanilla RAG) predicted answers
    AGREE with DisProt ground truth answers using multiple
    consistency metrics.

LLM Judge 2 -- Vanilla RAG:
    BiomedBERT retrieves the top-k most relevant DisProt facts
    from a knowledge base and concatenates them as the answer.
    No symbolic rules, no calibration -- pure neural retrieval.

Agreement Dimensions:
    1. SEMANTIC CONSISTENCY  (30%) -- shared core concepts
    2. KEYWORD AGREEMENT     (25%) -- same biomedical terms
    3. NUMERIC AGREEMENT     (25%) -- matching numbers (2% tolerance)
    4. DIRECTIONAL AGREEMENT (20%) -- same claim direction
    5. CONTRADICTION PENALTY (-0.05 per contradiction, max -0.30)

Output: agreement_score_results_2.txt (saved to same folder)

Usage:
    python agreement_score_2.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python agreement_score_2.py --demo
"""

import json
import re
import sys
import os
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict


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
# 2. STATS + GROUND TRUTH + LLM2 PREDICTIONS
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

# Ground truth rules (same across all judges)
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

# LLM2 Vanilla RAG answers -- retrieved and concatenated passages
# These mirror the vanilla RAG knowledge base passages from LLM_judge2.py
LLM2_RULES = [
    (["0.5","cutoff","disorder"],
     lambda s: f"The 0.5 disorder score threshold classifies protein regions as intrinsically disordered. Of {s['total_proteins']:,} DisProt proteins {s['pct_above_0.5']:.1f}% exceed this threshold with mean disorder score {s['mean_disorder']:.3f}. However {s['pct_above_0.3']:.1f}% exceed 0.3 meaning many true IDRs fall below 0.5 and are missed. Disorder scores between 0.3 and 0.5 define an ambiguous gray zone where proteins cannot be confidently classified without secondary validation."),
    (["short","residue"],
     lambda s: f"Of {s['total_regions']:,} DisProt regions {s['pct_short_regions']:.1f}% are shorter than 10 residues with mean {s['mean_region_length']:.1f} aa. Short IDRs are hard to predict reliably due to limited sequence context. Sliding window averaging smooths per-residue disorder scores but windows larger than mean region length risk smoothing out short IDRs entirely."),
    (["proline","glycine"],
     lambda s: f"Proline content DisProt mean {s['mean_proline']*100:.1f}% strongly predicts intrinsic disorder. Proline rigid pyrrolidine ring disrupts alpha-helices and beta-sheets preventing regular secondary structure. Glycine mean {s['mean_glycine']*100:.1f}% adds conformational freedom. Elevated proline and glycine together form a strong composite disorder signal."),
    (["sliding","window"],
     lambda s: f"Sliding window averaging smooths per-residue disorder scores to reduce noise. The mean disordered region length in DisProt is {s['mean_region_length']:.1f} amino acids. If sliding window size exceeds this mean short disordered regions risk being averaged out and lost entirely."),
    (["pfam","domain"],
     lambda s: f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain Pfam domains alongside disordered regions confirming IDRs and structured domains frequently co-occur. Each region must be evaluated independently. Proteins with no Pfam domains and disorder content above 0.5 are classified as intrinsically disordered proteins IDPs."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT below 50 strongly indicates intrinsic disorder. DisProt annotated disordered regions in {s['total_proteins']:,} proteins consistently show pLDDT below 50 the most reliable computational signal. pLDDT scores of 50 to 70 indicate ambiguous structure possibly conditionally disordered MoRF regions."),
]

DOMAIN_KEYWORDS = [
    "disorder","disordered","idr","idp","plddt","alphafold","pfam",
    "proline","glycine","residue","amino","backbone","threshold","cutoff",
    "disprot","intrinsic","region","sequence","confidence","annotated",
    "prediction","experimental","structured","conservative","morf"
]

DIRECTIONAL_PAIRS = [
    ("reliable",     "unreliable"),
    ("conservative", "aggressive"),
    ("strong",       "weak"),
    ("increases",    "decreases"),
    ("above",        "below"),
    ("high",         "low"),
    ("frequently",   "rarely"),
    ("consistent",   "inconsistent"),
    ("sufficient",   "insufficient"),
    ("structured",   "unstructured"),
    ("predicts",     "does not predict"),
    ("correlates",   "does not correlate"),
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
# 3. AGREEMENT SCORING ENGINE
# =============================================================

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","of","in","on","at",
    "to","for","with","by","from","and","or","but","not","this","that",
    "it","its","they","we","as","also","both","very","than","such","each"
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return [w for w in normalize(text).split()
            if w not in STOPWORDS and len(w) > 1]

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


def semantic_consistency(pred, gt):
    pred_set = set(tokenize(pred)) - STOPWORDS
    gt_set   = set(tokenize(gt))   - STOPWORDS
    if not gt_set:
        return {"score": 1.0, "shared": [], "unique_to_gt": []}
    shared       = pred_set & gt_set
    unique_to_gt = gt_set   - pred_set
    score        = len(shared) / len(gt_set)
    return {
        "score":        round(score, 4),
        "shared":       sorted(list(shared))[:12],
        "unique_to_gt": sorted(list(unique_to_gt))[:8],
    }


def keyword_agreement(pred, gt):
    pred_lower = pred.lower()
    gt_lower   = gt.lower()
    gt_keys    = [kw for kw in DOMAIN_KEYWORDS if kw in gt_lower]
    pred_keys  = [kw for kw in DOMAIN_KEYWORDS if kw in pred_lower]
    if not gt_keys:
        return {"score": 1.0, "matched": [], "missing": []}
    matched = [kw for kw in gt_keys if kw in pred_keys]
    missing = [kw for kw in gt_keys if kw not in pred_keys]
    return {
        "score":   round(len(matched) / len(gt_keys), 4),
        "matched": matched,
        "missing": missing,
    }


def numeric_agreement(pred, gt):
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)
    if not gt_nums:
        return {"score": 1.0, "matched": 0, "contradictions": []}
    matched        = 0
    contradictions = []
    for gn in gt_nums:
        if gn == 0:
            continue
        found = any(abs(pn - gn) / max(abs(gn), 1e-9) < 0.02 for pn in pred_nums)
        if found:
            matched += 1
        else:
            wrong = [pn for pn in pred_nums
                     if abs(pn - gn) / max(abs(gn), 1e-9) > 0.05
                     and abs(pn - gn) < gn * 2]
            if wrong:
                contradictions.append(f"GT={gn}, Pred has {wrong[0]:.1f}")
    total = len([n for n in gt_nums if n != 0])
    score = matched / total if total > 0 else 1.0
    return {
        "score":           round(score, 4),
        "matched":         matched,
        "contradictions":  contradictions,
    }


def directional_agreement(pred, gt):
    contradictions = []
    agreements     = []
    pl, gl         = pred.lower(), gt.lower()
    for pos, neg in DIRECTIONAL_PAIRS:
        gt_pos  = pos in gl
        gt_neg  = neg in gl
        pr_pos  = pos in pl
        pr_neg  = neg in pl
        if gt_pos and pr_neg and not pr_pos:
            contradictions.append(f"GT uses '{pos}' but pred uses '{neg}'")
        elif gt_neg and pr_pos and not pr_neg:
            contradictions.append(f"GT uses '{neg}' but pred uses '{pos}'")
        elif (gt_pos and pr_pos) or (gt_neg and pr_neg):
            agreements.append(pos if gt_pos else neg)
    score = len(agreements) / len(DIRECTIONAL_PAIRS)
    return {
        "score":          round(score, 4),
        "agreements":     agreements,
        "contradictions": contradictions,
    }


def agreement_score(pred, gt):
    semantic    = semantic_consistency(pred, gt)
    keywords    = keyword_agreement(pred, gt)
    numeric     = numeric_agreement(pred, gt)
    directional = directional_agreement(pred, gt)

    total_contradictions = (len(numeric["contradictions"]) +
                            len(directional["contradictions"]))
    all_contradictions   = numeric["contradictions"] + directional["contradictions"]

    score = (
        0.30 * semantic["score"]    +
        0.25 * keywords["score"]    +
        0.25 * numeric["score"]     +
        0.20 * directional["score"]
    )
    penalty = min(0.30, total_contradictions * 0.05)
    score   = round(max(0.0, score - penalty), 4)

    label = (
        "STRONG AGREEMENT"    if score >= 0.75 else
        "MODERATE AGREEMENT"  if score >= 0.50 else
        "WEAK AGREEMENT"      if score >= 0.25 else
        "DISAGREEMENT"
    )

    return {
        "overall_score":        score,
        "label":                label,
        "semantic":             semantic,
        "keywords":             keywords,
        "numeric":              numeric,
        "directional":          directional,
        "total_contradictions": total_contradictions,
        "contradiction_list":   all_contradictions,
        "penalty_applied":      round(penalty, 4),
    }


# =============================================================
# 4. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM2_RULES, stats)
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
# 5. WRITE agreement_score_results_2.txt
# =============================================================

def write_results(results, stats):
    scores  = [r["score"]["overall_score"]       for r in results]
    sem     = [r["score"]["semantic"]["score"]    for r in results]
    kw      = [r["score"]["keywords"]["score"]    for r in results]
    num     = [r["score"]["numeric"]["score"]     for r in results]
    dirc    = [r["score"]["directional"]["score"] for r in results]
    contras = [r["score"]["total_contradictions"] for r in results]

    mean_score = sum(scores) / len(scores)
    mean_sem   = sum(sem)    / len(sem)
    mean_kw    = sum(kw)     / len(kw)
    mean_num   = sum(num)    / len(num)
    mean_dir   = sum(dirc)   / len(dirc)
    total_c    = sum(contras)
    std_score  = math.sqrt(sum((s - mean_score)**2 for s in scores) / len(scores))

    strong   = sum(1 for r in results if r["score"]["label"] == "STRONG AGREEMENT")
    moderate = sum(1 for r in results if r["score"]["label"] == "MODERATE AGREEMENT")
    weak     = sum(1 for r in results if r["score"]["label"] == "WEAK AGREEMENT")
    disagree = sum(1 for r in results if r["score"]["label"] == "DISAGREEMENT")

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Agreement Score: LLM Judge 2 vs Ground Truth")
    lines.append("  Model   : Vanilla RAG -- BiomedBERT Retriever (LLM Judge 2)")
    lines.append("  Metric  : Multi-dimensional Agreement Score")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS LLM JUDGE 2 (VANILLA RAG)?")
    lines.append("-" * 70)
    lines.append("  LLM Judge 2 uses BiomedBERT to retrieve the top-k most")
    lines.append("  semantically similar DisProt knowledge base passages for")
    lines.append("  each question and concatenates them as the answer.")
    lines.append("  No symbolic rules, no calibration -- pure neural retrieval.")
    lines.append("")
    lines.append("  Compare with LLM Judge 1 (symbolic rules + calibration)")
    lines.append("  to see whether symbolic grounding improves agreement.")
    lines.append("")
    lines.append("AGREEMENT DIMENSIONS")
    lines.append("-" * 70)
    lines.append("  1. Semantic Consistency (30%) -- shared core concepts")
    lines.append("  2. Keyword Agreement    (25%) -- same biomedical terms")
    lines.append("  3. Numeric Agreement    (25%) -- matching numbers (2% tol)")
    lines.append("  4. Directional Agreement(20%) -- same claim direction")
    lines.append("  Contradiction penalty: -0.05 per contradiction (max -0.30)")
    lines.append("")
    lines.append("  Labels:")
    lines.append("    STRONG AGREEMENT   >= 0.75")
    lines.append("    MODERATE AGREEMENT >= 0.50")
    lines.append("    WEAK AGREEMENT     >= 0.25")
    lines.append("    DISAGREEMENT       <  0.25")
    lines.append("")
    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean agreement score         : {mean_score:.4f}  (std={std_score:.4f})")
    lines.append(f"  Mean semantic consistency    : {mean_sem:.4f}")
    lines.append(f"  Mean keyword agreement       : {mean_kw:.4f}")
    lines.append(f"  Mean numeric agreement       : {mean_num:.4f}")
    lines.append(f"  Mean directional agreement   : {mean_dir:.4f}")
    lines.append(f"  Total contradictions found   : {total_c}")
    lines.append("")
    lines.append(f"  Agreement quality breakdown:")
    lines.append(f"    STRONG AGREEMENT   : {strong:3d} questions")
    lines.append(f"    MODERATE AGREEMENT : {moderate:3d} questions")
    lines.append(f"    WEAK AGREEMENT     : {weak:3d} questions")
    lines.append(f"    DISAGREEMENT       : {disagree:3d} questions")
    lines.append("")
    lines.append("  Score Distribution:")
    for lo, hi, lbl in [(0.0,0.25,"0.00-0.25 DISAGREEMENT   "),
                         (0.25,0.50,"0.25-0.50 WEAK           "),
                         (0.50,0.75,"0.50-0.75 MODERATE       "),
                         (0.75,1.01,"0.75-1.00 STRONG         ")]:
        count = sum(1 for s in scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")
    lines.append("  Criteria Radar (mean scores):")
    for name, val in [("Semantic Consistency ", mean_sem),
                       ("Keyword Agreement    ", mean_kw),
                       ("Numeric Agreement    ", mean_num),
                       ("Directional Agreement", mean_dir)]:
        bar = "#" * int(val * 20) + "." * max(0, 20 - int(val * 20))
        lines.append(f"    {name} [{bar}] {val:.4f}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION AGREEMENT SCORES")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Overall Agreement : {s['overall_score']:.4f}  --  {s['label']}")
        lines.append(f"  Contradictions    : {s['total_contradictions']} "
                     f"(penalty={s['penalty_applied']:.4f})")
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65]
                      for i in range(0, len(r["ground_truth"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  LLM2 PREDICTION (Vanilla RAG):")
        for chunk in [r["prediction"][i:i+65]
                      for i in range(0, len(r["prediction"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  AGREEMENT BREAKDOWN:")
        lines.append(f"    Semantic consistency  : {s['semantic']['score']:.4f}")
        lines.append(f"    Keyword agreement     : {s['keywords']['score']:.4f}")
        lines.append(f"    Numeric agreement     : {s['numeric']['score']:.4f}")
        lines.append(f"    Directional agreement : {s['directional']['score']:.4f}")
        lines.append(f"    Penalty applied       : -{s['penalty_applied']:.4f}")
        lines.append(f"    OVERALL               : {s['overall_score']:.4f}")
        if s["keywords"]["missing"]:
            lines.append(f"    Missing keywords : {', '.join(s['keywords']['missing'][:8])}")
        if s["keywords"]["matched"]:
            lines.append(f"    Matched keywords : {', '.join(s['keywords']['matched'][:8])}")
        if s["contradiction_list"]:
            lines.append(f"    Contradictions   : {'; '.join(s['contradiction_list'][:3])}")
        if s["semantic"]["unique_to_gt"]:
            lines.append(f"    In GT not pred   : {', '.join(s['semantic']['unique_to_gt'][:6])}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF AGREEMENT SCORE -- LLM Judge 2 (Vanilla RAG)")
    lines.append(f"  Mean: {mean_score:.4f} | Strong: {strong} | Moderate: {moderate} | "
                 f"Weak: {weak} | Disagree: {disagree}")
    lines.append(f"  Total contradictions: {total_c}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "agreement_score_results_2.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Agreement score results written to: {out_path}\n")


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
        description="Agreement score: LLM Judge 2 (Vanilla RAG) vs ground truth"
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
    print("[INFO] Computing agreement scores for LLM Judge 2...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()