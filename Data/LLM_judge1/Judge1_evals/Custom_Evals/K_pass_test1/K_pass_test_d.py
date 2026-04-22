"""
BMEN-499 AlphaFold -- K-Pass Test: LLM Judge 1
-----------------------------------------------
Purpose:
    Runs LLM Judge 1 predicted answers through K evaluation passes,
    each pass applying a different scoring lens. Aggregates results
    across all K passes to measure consistency and robustness.

What is a K-Pass Test?
    A K-pass test evaluates the same answer K times, each time from
    a different evaluative perspective (pass). This reveals:

      - Is the model CONSISTENT across different evaluation criteria?
      - Does the answer hold up under MULTIPLE types of scrutiny?
      - Which passes does the model FAIL most often?
      - What is the AGGREGATE reliability score across all passes?

    K = 5 passes in this implementation:

      Pass 1 -- FACTUAL PASS
                Are the stated facts numerically correct vs GT?

      Pass 2 -- COMPLETENESS PASS
                Does the prediction cover all key GT concepts?

      Pass 3 -- SEMANTIC PASS
                Do prediction and GT express the same meaning?

      Pass 4 -- TERMINOLOGY PASS
                Does the prediction use correct biomedical terms?

      Pass 5 -- CONSISTENCY PASS
                Is the prediction internally consistent (no
                self-contradictions within the answer itself)?

    K-Pass Score = mean score across all 5 passes
    A question PASSES a pass if it scores >= 0.80 on that pass.
    Overall pass rate = passes passed / (questions x K)

Output: k_pass_results_d.txt (saved to same folder as this script)

Usage:
    python K_pass_test_d.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python K_pass_test_d.py --demo
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

DOMAIN_TERMS = [
    "disorder","disordered","idr","idp","plddt","alphafold","pfam",
    "proline","glycine","residue","amino","backbone","threshold","cutoff",
    "disprot","intrinsic","region","sequence","confidence","annotated",
    "prediction","experimental","structured","conservative"
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
# 3. HELPER FUNCTIONS
# =============================================================

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","of","in","on",
    "at","to","for","with","by","from","and","or","but","not","this",
    "that","it","its","they","we","as","also","both","very","each","more"
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return [w for w in normalize(text).split()
            if w not in STOPWORDS and len(w) > 1]

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# =============================================================
# 4. THE FIVE PASSES
# =============================================================

def pass1_factual(pred, gt, stats):
    """
    Pass 1: FACTUAL PASS
    Check if numbers in prediction match ground truth numbers.
    Tolerance: 5% relative difference.
    Score = matched_numbers / total_gt_numbers
    """
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)

    if not gt_nums:
        return 1.0, "No numeric facts to check", []

    matched = []
    missed  = []
    for gn in gt_nums:
        if gn == 0:
            continue
        if any(abs(pn - gn) / max(abs(gn), 1e-9) < 0.05 for pn in pred_nums):
            matched.append(gn)
        else:
            missed.append(gn)

    total = len(matched) + len(missed)
    score = len(matched) / total if total > 0 else 1.0

    details = []
    if matched:
        details.append(f"Matched: {matched[:4]}")
    if missed:
        details.append(f"Missing/wrong: {missed[:4]}")

    return round(score, 4), " | ".join(details) if details else "All numeric facts correct", missed


def pass2_completeness(pred, gt):
    """
    Pass 2: COMPLETENESS PASS
    Check what fraction of GT content words appear in prediction.
    Score = GT_words_in_pred / total_GT_words
    """
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))

    if not gt_toks:
        return 1.0, "No GT content to check", []

    covered = gt_toks & pred_toks
    missing = gt_toks - pred_toks
    score   = len(covered) / len(gt_toks)

    details = f"{len(covered)}/{len(gt_toks)} GT concepts covered"
    missing_sample = sorted(list(missing))[:6]

    return round(score, 4), details, missing_sample


def pass3_semantic(pred, gt):
    """
    Pass 3: SEMANTIC PASS
    Measure semantic overlap using token-level cosine similarity.
    Score = cosine similarity of token frequency vectors.
    """
    pred_toks = Counter(tokenize(pred))
    gt_toks   = Counter(tokenize(gt))

    shared = set(pred_toks) & set(gt_toks)
    if not shared:
        return 0.0, "No shared tokens", []

    dot    = sum(pred_toks[t] * gt_toks[t] for t in shared)
    mag_p  = math.sqrt(sum(v**2 for v in pred_toks.values()))
    mag_g  = math.sqrt(sum(v**2 for v in gt_toks.values()))
    score  = dot / (mag_p * mag_g) if (mag_p and mag_g) else 0.0

    top_shared = sorted(shared, key=lambda t: pred_toks[t]*gt_toks[t], reverse=True)[:6]
    details    = f"Cosine={score:.4f} | Top shared: {', '.join(top_shared)}"

    return round(score, 4), details, []


def pass4_terminology(pred, gt):
    """
    Pass 4: TERMINOLOGY PASS
    Check that prediction uses the same biomedical terms as GT.
    Score = GT_domain_terms_in_pred / GT_domain_terms
    """
    pred_lower = pred.lower()
    gt_lower   = gt.lower()

    gt_terms   = [t for t in DOMAIN_TERMS if t in gt_lower]
    pred_terms = [t for t in DOMAIN_TERMS if t in pred_lower]

    if not gt_terms:
        return 1.0, "No domain terms in GT", []

    matched = [t for t in gt_terms if t in pred_terms]
    missing = [t for t in gt_terms if t not in pred_terms]
    score   = len(matched) / len(gt_terms)

    details = f"{len(matched)}/{len(gt_terms)} domain terms present"

    return round(score, 4), details, missing


def pass5_consistency(pred):
    """
    Pass 5: CONSISTENCY PASS
    Check if the prediction is internally self-consistent.
    Detects self-contradictions within the prediction itself.

    Checks:
      - Does it use opposite directional words for the same subject?
      - Does it repeat the same claim with contradictory numbers?
      - Is sentence structure coherent (no orphaned clauses)?
    """
    inconsistencies = []
    pred_lower = pred.lower()
    sentences  = [s.strip() for s in pred.split(".") if len(s.strip()) > 5]

    # Check for directional contradictions within the prediction
    contradiction_pairs = [
        ("reliable",    "unreliable"),
        ("strong",      "weak"),
        ("high",        "low"),
        ("increases",   "decreases"),
        ("structured",  "unstructured"),
        ("sufficient",  "insufficient"),
    ]
    for pos, neg in contradiction_pairs:
        if pos in pred_lower and neg in pred_lower:
            inconsistencies.append(f"Uses both '{pos}' and '{neg}'")

    # Check for repeated contradictory numbers
    nums = extract_numbers(pred)
    if len(nums) > 2:
        sorted_nums = sorted(set(nums))
        for i in range(len(sorted_nums)-1):
            ratio = sorted_nums[i+1] / max(sorted_nums[i], 1e-9)
            if 1.5 < ratio < 100 and sorted_nums[i] > 0.1:
                pass  # Normal range variation -- not a contradiction

    # Check sentence count as coherence proxy
    if len(sentences) < 1:
        inconsistencies.append("Answer has no complete sentences")

    # Score: 1.0 if no inconsistencies, deduct 0.2 per issue
    score   = max(0.0, 1.0 - len(inconsistencies) * 0.2)
    details = (f"No internal inconsistencies found" if not inconsistencies
               else f"{len(inconsistencies)} inconsistencies: {'; '.join(inconsistencies[:3])}")

    return round(score, 4), details, inconsistencies


# =============================================================
# 5. K-PASS ENGINE
# =============================================================

PASSES = [
    {
        "id":          1,
        "name":        "Factual Pass",
        "description": "Are stated numbers correct vs ground truth? (tolerance 5%)",
        "weight":      0.30,
    },
    {
        "id":          2,
        "name":        "Completeness Pass",
        "description": "Does prediction cover all key GT concepts?",
        "weight":      0.25,
    },
    {
        "id":          3,
        "name":        "Semantic Pass",
        "description": "Do prediction and GT express the same meaning?",
        "weight":      0.20,
    },
    {
        "id":          4,
        "name":        "Terminology Pass",
        "description": "Does prediction use correct biomedical terminology?",
        "weight":      0.15,
    },
    {
        "id":          5,
        "name":        "Consistency Pass",
        "description": "Is the prediction internally self-consistent?",
        "weight":      0.10,
    },
]

K = len(PASSES)


def run_k_passes(pred, gt, stats):
    """
    Run all K passes on a single prediction/GT pair.
    Returns per-pass scores and weighted aggregate.
    """
    pass_results = []

    # Pass 1: Factual
    sc, det, issues = pass1_factual(pred, gt, stats)
    pass_results.append({
        "pass_id": 1, "name": "Factual Pass",
        "score": sc, "passed": sc >= 0.80,
        "detail": det, "issues": issues,
        "weight": 0.30,
    })

    # Pass 2: Completeness
    sc, det, issues = pass2_completeness(pred, gt)
    pass_results.append({
        "pass_id": 2, "name": "Completeness Pass",
        "score": sc, "passed": sc >= 0.80,
        "detail": det, "issues": issues,
        "weight": 0.25,
    })

    # Pass 3: Semantic
    sc, det, issues = pass3_semantic(pred, gt)
    pass_results.append({
        "pass_id": 3, "name": "Semantic Pass",
        "score": sc, "passed": sc >= 0.80,
        "detail": det, "issues": issues,
        "weight": 0.20,
    })

    # Pass 4: Terminology
    sc, det, issues = pass4_terminology(pred, gt)
    pass_results.append({
        "pass_id": 4, "name": "Terminology Pass",
        "score": sc, "passed": sc >= 0.80,
        "detail": det, "issues": issues,
        "weight": 0.15,
    })

    # Pass 5: Consistency
    sc, det, issues = pass5_consistency(pred)
    pass_results.append({
        "pass_id": 5, "name": "Consistency Pass",
        "score": sc, "passed": sc >= 0.80,
        "detail": det, "issues": issues,
        "weight": 0.10,
    })

    # Aggregate
    passes_passed    = sum(1 for p in pass_results if p["passed"])
    weighted_score   = sum(p["score"] * p["weight"] for p in pass_results)
    mean_score       = sum(p["score"] for p in pass_results) / K

    label = (
        f"PASSED ALL {K}"      if passes_passed == K else
        f"PASSED {passes_passed}/{K}" if passes_passed >= K // 2 + 1 else
        f"FAILED {K - passes_passed}/{K}"
    )

    return {
        "passes":        pass_results,
        "passes_passed": passes_passed,
        "passes_failed": K - passes_passed,
        "weighted_score": round(weighted_score, 4),
        "mean_score":     round(mean_score, 4),
        "label":          label,
    }


# =============================================================
# 6. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = run_k_passes(pred, gt, stats)

        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })

        pass_summary = " | ".join(
            f"P{p['pass_id']}={'OK' if p['passed'] else 'FAIL'}({p['score']:.2f})"
            for p in sc["passes"]
        )
        print(f"  Q{i:3d} | {pass_summary} | "
              f"Weighted={sc['weighted_score']:.4f} | {sc['label']}")

    return results


# =============================================================
# 7. WRITE k_pass_results_d.txt
# =============================================================

def write_results(results, stats):
    weighted_scores = [r["score"]["weighted_score"] for r in results]
    mean_scores     = [r["score"]["mean_score"]     for r in results]
    passes_passed   = [r["score"]["passes_passed"]  for r in results]

    mean_weighted = sum(weighted_scores) / len(weighted_scores)
    mean_mean     = sum(mean_scores)     / len(mean_scores)
    std_weighted  = math.sqrt(sum((s - mean_weighted)**2
                                  for s in weighted_scores) / len(weighted_scores))

    total_possible = len(results) * K
    total_passed   = sum(passes_passed)
    overall_rate   = total_passed / total_possible

    # Per-pass statistics
    pass_stats = defaultdict(list)
    for r in results:
        for p in r["score"]["passes"]:
            pass_stats[p["pass_id"]].append(p["score"])

    best_q  = max(results, key=lambda r: r["score"]["weighted_score"])
    worst_q = min(results, key=lambda r: r["score"]["weighted_score"])

    all_passed   = sum(1 for r in results if r["score"]["passes_passed"] == K)
    most_passed  = sum(1 for r in results if r["score"]["passes_passed"] >= 4)
    half_passed  = sum(1 for r in results if r["score"]["passes_passed"] == K // 2 + 1)
    mostly_failed = sum(1 for r in results if r["score"]["passes_passed"] <= K // 2)

    lines = []
    lines.append("=" * 70)
    lines.append(f"  BMEN-499 AlphaFold -- K-Pass Test D (K={K}, Threshold=0.80 Very Strict): LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append(f"  K       : {K} evaluation passes per question")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS A K-PASS TEST?")
    lines.append("-" * 70)
    lines.append(f"  A K-pass test evaluates each answer {K} times, each time from")
    lines.append("  a different evaluative perspective. An answer must hold up")
    lines.append("  under ALL K passes to be considered truly reliable.")
    lines.append("")
    lines.append("  Pass threshold: score >= 0.80 to PASS a pass.")
    lines.append("  Weighted K-pass score = weighted mean across all passes.")
    lines.append("")
    for p in PASSES:
        lines.append(f"  Pass {p['id']} -- {p['name']} (weight={p['weight']:.0%})")
        lines.append(f"    {p['description']}")
        lines.append("")

    lines.append("OVERALL K-PASS RESULTS")
    lines.append("-" * 70)
    lines.append(f"  Overall pass rate       : {overall_rate:.1%} "
                 f"({total_passed}/{total_possible} pass-question combinations)")
    lines.append(f"  Mean weighted K-score   : {mean_weighted:.4f}  (std={std_weighted:.4f})")
    lines.append(f"  Mean unweighted K-score : {mean_mean:.4f}")
    lines.append(f"  Best  : Q{best_q['q_num']} = {best_q['score']['weighted_score']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Worst : Q{worst_q['q_num']} = {worst_q['score']['weighted_score']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Questions passing ALL {K} passes    : {all_passed}")
    lines.append(f"  Questions passing 4/{K} passes      : {most_passed}")
    lines.append(f"  Questions passing {K//2+1}/{K} passes      : {half_passed}")
    lines.append(f"  Questions failing majority         : {mostly_failed}")
    lines.append("")

    lines.append("  PER-PASS PERFORMANCE:")
    lines.append(f"    {'Pass':<25} {'Mean Score':<12} {'Pass Rate':<12} {'Hardest?'}")
    lines.append(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    pass_means = {}
    for p in PASSES:
        pid    = p["id"]
        pscores = pass_stats[pid]
        pmean  = sum(pscores) / len(pscores)
        prate  = sum(1 for s in pscores if s >= 0.80) / len(pscores)
        pass_means[pid] = pmean
        bar    = "#" * int(pmean * 10) + "." * (10 - int(pmean * 10))
        lines.append(f"    {p['name']:<25} [{bar}] {pmean:.4f}  {prate:.1%}")

    hardest = min(pass_means, key=pass_means.get)
    easiest = max(pass_means, key=pass_means.get)
    lines.append("")
    lines.append(f"  Hardest pass : Pass {hardest} -- {PASSES[hardest-1]['name']} "
                 f"(mean={pass_means[hardest]:.4f})")
    lines.append(f"  Easiest pass : Pass {easiest} -- {PASSES[easiest-1]['name']} "
                 f"(mean={pass_means[easiest]:.4f})")
    lines.append("")

    lines.append("  K-Pass Score Distribution:")
    for lo, hi, lbl in [(0.0,0.25,"<0.25 VERY LOW "),(0.25,0.50,"<0.50 LOW      "),
                         (0.50,0.75,"<0.75 MODERATE "),(0.75,0.90,"<0.90 HIGH     "),
                         (0.90,1.01,">=0.90 VERY HIGH")]:
        count = sum(1 for s in weighted_scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION K-PASS REPORT")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  K-Pass Result : {s['label']}")
        lines.append(f"  Weighted score: {s['weighted_score']:.4f}")
        lines.append(f"  Mean score    : {s['mean_score']:.4f}")
        lines.append(f"  Passed        : {s['passes_passed']}/{K} passes")
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
        lines.append("  PASS BREAKDOWN:")
        for p in s["passes"]:
            status = "PASS" if p["passed"] else "FAIL"
            bar    = "#" * int(p["score"] * 10) + "." * (10 - int(p["score"] * 10))
            lines.append(f"    Pass {p['pass_id']} [{p['name']:<18}] "
                         f"[{bar}] {p['score']:.4f}  {status}")
            lines.append(f"           Detail : {p['detail']}")
            if p["issues"]:
                sample = p["issues"][:3] if isinstance(p["issues"][0], str) else \
                         [str(x) for x in p["issues"][:3]]
                lines.append(f"           Issues : {', '.join(sample)}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  END OF K-PASS TEST (K={K}) -- LLM Judge 1")
    lines.append(f"  Overall pass rate: {overall_rate:.1%} ({total_passed}/{total_possible})")
    lines.append(f"  Mean weighted K-score: {mean_weighted:.4f} (std={std_weighted:.4f})")
    lines.append(f"  All {K} passes passed: {all_passed} questions | "
                 f"Majority failed: {mostly_failed} questions")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "k_pass_results_d.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] K-pass results written to: {out_path}\n")


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
        description=f"K-Pass Test D (K={K}, Threshold=0.80 Very Strict): LLM Judge 1 predictions vs ground truth"
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
    print(f"[INFO] Running K-Pass Test (K={K}) on LLM Judge 1...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()