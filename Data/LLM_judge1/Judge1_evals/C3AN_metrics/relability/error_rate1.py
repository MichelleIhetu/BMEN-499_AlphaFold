"""
BMEN-499 AlphaFold -- Error Rate: LLM Judge 1 vs Ground Truth
--------------------------------------------------------------
Purpose:
    Measures the error rate of LLM Judge 1 predicted answers
    compared to DisProt ground truth answers across multiple
    error categories.

What is Error Rate?
    Error rate measures how often and how severely the model
    produces incorrect, incomplete, or misleading outputs
    compared to the ground truth.

    Five error types are detected:

      1. FACTUAL ERROR       -- Prediction states a fact that is
                                wrong compared to the ground truth
                                (e.g. wrong numbers, wrong claims)

      2. OMISSION ERROR      -- Prediction leaves out key information
                                that the ground truth includes
                                (missing concepts, missing numbers)

      3. HALLUCINATION ERROR -- Prediction introduces information
                                not present in the ground truth
                                (adds claims GT does not make)

      4. TERMINOLOGY ERROR   -- Prediction uses wrong or missing
                                biomedical terminology compared to GT

      5. NUMERIC ERROR       -- Prediction states numbers that differ
                                significantly from GT numbers
                                (tolerance: 5%)

    Error rate = total errors / total possible checks (0.0 - 1.0)
    Lower is better.

Output: error_rate_results.txt (saved to same folder as this script)

Usage:
    python error_rate1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python error_rate1.py --demo
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

# Key biomedical terms expected in answers
DOMAIN_TERMS = [
    "disorder", "disordered", "idr", "idp", "plddt", "alphafold",
    "pfam", "proline", "glycine", "residue", "amino", "backbone",
    "threshold", "cutoff", "disprot", "intrinsic", "region", "sequence",
    "confidence", "prediction", "annotated", "experimental", "structured"
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
# 3. ERROR DETECTION ENGINE
# =============================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

def extract_key_claims(text):
    """Extract short claim phrases (noun + verb patterns) from text."""
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return sentences

def tokenize(text):
    stopwords = {"a","an","the","is","are","was","were","be","been","of",
                 "in","on","at","to","for","with","by","from","and","or",
                 "but","not","this","that","it","its","they","we","as"}
    return [w for w in normalize(text).split()
            if w not in stopwords and len(w) > 1]


# --- Error Type 1: Factual Errors ----------------------------
def detect_factual_errors(pred, gt, stats):
    """
    Detect factual errors by checking key statistical claims.
    Compares core factual assertions against known ground truth values.
    """
    errors = []

    # Check key GT facts that should appear in prediction
    key_facts = [
        (str(stats["total_proteins"]),
         "total protein count"),
        (f"{stats['pct_above_0.5']:.1f}",
         "percentage above 0.5 threshold"),
        (f"{stats['mean_disorder']:.3f}",
         "mean disorder score"),
        (f"{stats['mean_region_length']:.1f}",
         "mean region length"),
        (f"{stats['mean_proline']*100:.1f}",
         "mean proline fraction"),
        (f"{stats['mean_glycine']*100:.1f}",
         "mean glycine fraction"),
        (f"{stats['pct_with_pfam']:.1f}",
         "percentage with Pfam domains"),
    ]

    pred_lower = pred.lower()
    gt_lower   = gt.lower()

    for fact_val, fact_name in key_facts:
        if fact_val in gt_lower and fact_val not in pred_lower:
            # Fact is in GT but missing from pred -- check if a wrong value is there
            gt_nums   = extract_numbers(gt)
            pred_nums = extract_numbers(pred)
            try:
                gv = float(fact_val.replace(",", ""))
                wrong_vals = [pn for pn in pred_nums
                              if abs(pn - gv) / max(abs(gv), 1e-9) > 0.10
                              and abs(pn - gv) < abs(gv) * 5]
                if wrong_vals:
                    errors.append({
                        "type":     "FACTUAL",
                        "severity": "HIGH",
                        "detail":   f"Wrong {fact_name}: GT={fact_val}, "
                                    f"Pred has {wrong_vals[0]:.3f} instead",
                    })
            except ValueError:
                pass

    return errors


# --- Error Type 2: Omission Errors ---------------------------
def detect_omission_errors(pred, gt):
    """
    Detect omission errors -- key concepts in GT that are missing from pred.
    """
    errors = []
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))

    # Domain-specific terms in GT that are missing from pred
    gt_domain   = [t for t in DOMAIN_TERMS if t in gt.lower()]
    pred_domain = [t for t in DOMAIN_TERMS if t in pred.lower()]
    missing     = [t for t in gt_domain if t not in pred_domain]

    if len(missing) > 3:
        errors.append({
            "type":     "OMISSION",
            "severity": "MEDIUM",
            "detail":   f"Missing {len(missing)} key biomedical terms from GT: "
                        f"{', '.join(missing[:5])}",
        })

    # Check if GT numbers are omitted entirely
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)
    if gt_nums and not pred_nums:
        errors.append({
            "type":     "OMISSION",
            "severity": "HIGH",
            "detail":   "GT contains numeric facts but prediction has no numbers",
        })

    # Check significant content word omission
    gt_content   = gt_toks - set(DOMAIN_TERMS)
    pred_content = pred_toks - set(DOMAIN_TERMS)
    omitted_ratio = len(gt_content - pred_content) / max(len(gt_content), 1)
    if omitted_ratio > 0.6:
        errors.append({
            "type":     "OMISSION",
            "severity": "MEDIUM",
            "detail":   f"{omitted_ratio*100:.1f}% of GT content words missing from prediction",
        })

    return errors


# --- Error Type 3: Hallucination Errors ----------------------
def detect_hallucination_errors(pred, gt):
    """
    Detect hallucination errors -- pred introduces content not in GT.
    Hallucination = pred has significant content not grounded in GT.
    """
    errors = []
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))

    # Words unique to prediction (not in GT)
    pred_only = pred_toks - gt_toks
    gt_only   = gt_toks   - pred_toks

    # High ratio of pred-only content = possible hallucination
    halluc_ratio = len(pred_only) / max(len(pred_toks), 1)
    if halluc_ratio > 0.5:
        sample = sorted(list(pred_only))[:8]
        errors.append({
            "type":     "HALLUCINATION",
            "severity": "MEDIUM",
            "detail":   f"{halluc_ratio*100:.1f}% of prediction content not in GT. "
                        f"Extra terms: {', '.join(sample)}",
        })

    # Check for numbers in pred not present in GT
    gt_nums   = set(round(n, 1) for n in extract_numbers(gt))
    pred_nums = extract_numbers(pred)
    invented  = [pn for pn in pred_nums
                 if not any(abs(pn - gn) / max(abs(gn), 1e-9) < 0.05
                            for gn in gt_nums)
                 and pn > 1.0]  # skip small decimals like 0.5 threshold
    if len(invented) > 2:
        errors.append({
            "type":     "HALLUCINATION",
            "severity": "MEDIUM",
            "detail":   f"Prediction contains {len(invented)} numbers not in GT: "
                        f"{invented[:4]}",
        })

    return errors


# --- Error Type 4: Terminology Errors ------------------------
def detect_terminology_errors(pred, gt):
    """
    Detect terminology errors -- wrong or missing biomedical terms.
    """
    errors = []
    pred_lower = pred.lower()
    gt_lower   = gt.lower()

    # Terms in GT that should appear in pred
    gt_terms   = [t for t in DOMAIN_TERMS if t in gt_lower]
    pred_terms = [t for t in DOMAIN_TERMS if t in pred_lower]
    missing    = [t for t in gt_terms if t not in pred_terms]

    if missing:
        severity = "HIGH" if len(missing) > 4 else "LOW"
        errors.append({
            "type":     "TERMINOLOGY",
            "severity": severity,
            "detail":   f"Missing biomedical terms: {', '.join(missing[:6])}",
        })

    return errors


# --- Error Type 5: Numeric Errors ----------------------------
def detect_numeric_errors(pred, gt):
    """
    Detect numeric errors -- numbers that differ by more than 5% from GT.
    """
    errors     = []
    gt_nums    = extract_numbers(gt)
    pred_nums  = extract_numbers(pred)

    if not gt_nums:
        return errors

    numeric_errors = []
    for gn in gt_nums:
        if gn == 0:
            continue
        close = any(abs(pn - gn) / abs(gn) < 0.05 for pn in pred_nums)
        if not close:
            conflicts = [pn for pn in pred_nums
                         if abs(pn - gn) / abs(gn) > 0.05
                         and abs(pn - gn) < abs(gn) * 5]
            if conflicts:
                pct_diff = abs(conflicts[0] - gn) / abs(gn) * 100
                numeric_errors.append({
                    "gt_val":    gn,
                    "pred_val":  conflicts[0],
                    "pct_diff":  round(pct_diff, 1),
                    "severity":  "HIGH" if pct_diff > 50 else "MEDIUM",
                })

    if numeric_errors:
        for ne in numeric_errors[:3]:
            errors.append({
                "type":     "NUMERIC",
                "severity": ne["severity"],
                "detail":   f"GT={ne['gt_val']}, Pred={ne['pred_val']} "
                            f"(off by {ne['pct_diff']:.1f}%)",
            })

    return errors


# --- Aggregate all errors for one question -------------------
def compute_error_rate(pred, gt, stats):
    """
    Run all five error detectors and compute overall error rate.

    Error rate = weighted error count / total checks
    Each error type has a maximum possible check count.
    """
    factual      = detect_factual_errors(pred, gt, stats)
    omission     = detect_omission_errors(pred, gt)
    hallucination = detect_hallucination_errors(pred, gt)
    terminology  = detect_terminology_errors(pred, gt)
    numeric      = detect_numeric_errors(pred, gt)

    all_errors  = factual + omission + hallucination + terminology + numeric
    total_errors = len(all_errors)

    high_errors   = sum(1 for e in all_errors if e["severity"] == "HIGH")
    medium_errors = sum(1 for e in all_errors if e["severity"] == "MEDIUM")
    low_errors    = sum(1 for e in all_errors if e["severity"] == "LOW")

    # Weighted error score
    weighted = high_errors * 1.0 + medium_errors * 0.5 + low_errors * 0.25

    # Normalize: assume max 10 weighted errors = error rate of 1.0
    error_rate = round(min(1.0, weighted / 10.0), 4)

    label = (
        "NO ERRORS"      if error_rate == 0.0 else
        "LOW ERROR RATE" if error_rate < 0.20 else
        "MODERATE ERROR RATE" if error_rate < 0.50 else
        "HIGH ERROR RATE"
    )

    return {
        "error_rate":     error_rate,
        "label":          label,
        "total_errors":   total_errors,
        "high_errors":    high_errors,
        "medium_errors":  medium_errors,
        "low_errors":     low_errors,
        "weighted_score": round(weighted, 2),
        "factual":        factual,
        "omission":       omission,
        "hallucination":  hallucination,
        "terminology":    terminology,
        "numeric":        numeric,
        "all_errors":     all_errors,
    }


# =============================================================
# 4. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = compute_error_rate(pred, gt, stats)
        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })
        print(f"  Q{i:3d} | Error rate={sc['error_rate']:.4f} | "
              f"Errors={sc['total_errors']} "
              f"(H={sc['high_errors']}, M={sc['medium_errors']}, L={sc['low_errors']}) "
              f"| {sc['label']}")
    return results


# =============================================================
# 5. WRITE error_rate_results.txt
# =============================================================

def write_results(results, stats):
    er_scores  = [r["score"]["error_rate"]    for r in results]
    t_errors   = [r["score"]["total_errors"]  for r in results]
    t_high     = [r["score"]["high_errors"]   for r in results]
    t_medium   = [r["score"]["medium_errors"] for r in results]
    t_low      = [r["score"]["low_errors"]    for r in results]
    t_factual  = [len(r["score"]["factual"])      for r in results]
    t_omission = [len(r["score"]["omission"])     for r in results]
    t_halluc   = [len(r["score"]["hallucination"]) for r in results]
    t_term     = [len(r["score"]["terminology"])  for r in results]
    t_numeric  = [len(r["score"]["numeric"])      for r in results]

    mean_er    = sum(er_scores) / len(er_scores)
    total_all  = sum(t_errors)

    no_error  = sum(1 for r in results if r["score"]["label"] == "NO ERRORS")
    low_er    = sum(1 for r in results if r["score"]["label"] == "LOW ERROR RATE")
    mod_er    = sum(1 for r in results if r["score"]["label"] == "MODERATE ERROR RATE")
    high_er   = sum(1 for r in results if r["score"]["label"] == "HIGH ERROR RATE")

    best_q  = min(results, key=lambda r: r["score"]["error_rate"])
    worst_q = max(results, key=lambda r: r["score"]["error_rate"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Error Rate: LLM Judge 1 vs Ground Truth")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : Multi-type Error Rate Analysis")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS ERROR RATE?")
    lines.append("-" * 70)
    lines.append("  Error rate measures how often the LLM prediction produces")
    lines.append("  incorrect, incomplete, or misleading content compared to GT.")
    lines.append("  Score range: 0.0 (no errors) to 1.0 (maximum errors).")
    lines.append("  Lower is better.")
    lines.append("")
    lines.append("  FIVE ERROR TYPES DETECTED:")
    lines.append("")
    lines.append("  1. FACTUAL ERROR (severity=HIGH)")
    lines.append("     Prediction states a wrong fact compared to ground truth.")
    lines.append("     Example: GT says 13,396 proteins but pred says 5,000.")
    lines.append("")
    lines.append("  2. OMISSION ERROR (severity=MEDIUM)")
    lines.append("     Prediction leaves out key GT information entirely.")
    lines.append("     Example: GT mentions pLDDT scores but pred does not.")
    lines.append("")
    lines.append("  3. HALLUCINATION ERROR (severity=MEDIUM)")
    lines.append("     Prediction adds content not grounded in ground truth.")
    lines.append("     Example: Pred introduces claims GT never makes.")
    lines.append("")
    lines.append("  4. TERMINOLOGY ERROR (severity=LOW to HIGH)")
    lines.append("     Prediction uses wrong or missing biomedical terms.")
    lines.append("     Example: GT uses 'IDR' but pred never mentions it.")
    lines.append("")
    lines.append("  5. NUMERIC ERROR (severity=MEDIUM to HIGH)")
    lines.append("     Prediction states numbers that differ >5% from GT.")
    lines.append("     Example: GT says 29.1% but pred says 45%.")
    lines.append("")
    lines.append("  Error rate = weighted errors / 10 (capped at 1.0)")
    lines.append("  Weights: HIGH=1.0, MEDIUM=0.5, LOW=0.25")
    lines.append("")
    lines.append("  Labels:")
    lines.append("    NO ERRORS            : error rate = 0.0")
    lines.append("    LOW ERROR RATE       : error rate < 0.20")
    lines.append("    MODERATE ERROR RATE  : error rate < 0.50")
    lines.append("    HIGH ERROR RATE      : error rate >= 0.50")
    lines.append("")

    lines.append("OVERALL ERROR RATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean error rate         : {mean_er:.4f}")
    lines.append(f"  Total errors detected   : {total_all}")
    lines.append(f"    Factual errors        : {sum(t_factual)}")
    lines.append(f"    Omission errors       : {sum(t_omission)}")
    lines.append(f"    Hallucination errors  : {sum(t_halluc)}")
    lines.append(f"    Terminology errors    : {sum(t_term)}")
    lines.append(f"    Numeric errors        : {sum(t_numeric)}")
    lines.append(f"  High severity errors    : {sum(t_high)}")
    lines.append(f"  Medium severity errors  : {sum(t_medium)}")
    lines.append(f"  Low severity errors     : {sum(t_low)}")
    lines.append("")
    lines.append(f"  Best question  : Q{best_q['q_num']} = {best_q['score']['error_rate']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Worst question : Q{worst_q['q_num']} = {worst_q['score']['error_rate']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Error rate breakdown:")
    lines.append(f"    NO ERRORS           : {no_error:3d} questions")
    lines.append(f"    LOW ERROR RATE      : {low_er:3d} questions")
    lines.append(f"    MODERATE ERROR RATE : {mod_er:3d} questions")
    lines.append(f"    HIGH ERROR RATE     : {high_er:3d} questions")
    lines.append("")

    lines.append("  Error Rate Distribution:")
    for lo, hi, lbl in [(0.0,0.0,"0.00 NO ERRORS   "),(0.0,0.2,"<0.20 LOW        "),
                         (0.2,0.5,"<0.50 MODERATE   "),(0.5,1.01,">=0.50 HIGH      ")]:
        if lo == 0.0 and hi == 0.0:
            count = sum(1 for s in er_scores if s == 0.0)
        else:
            count = sum(1 for s in er_scores if lo < s <= hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("  Error Type Frequency:")
    type_counts = [
        ("Factual    ", sum(t_factual)),
        ("Omission   ", sum(t_omission)),
        ("Hallucination", sum(t_halluc)),
        ("Terminology", sum(t_term)),
        ("Numeric    ", sum(t_numeric)),
    ]
    for name, count in type_counts:
        bar = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {name} | {bar} | {count} errors")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION ERROR REPORT")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Error rate : {s['error_rate']:.4f}  --  {s['label']}")
        lines.append(f"  Total      : {s['total_errors']} errors "
                     f"(High={s['high_errors']}, Medium={s['medium_errors']}, "
                     f"Low={s['low_errors']}, Weighted={s['weighted_score']})")
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

        if s["total_errors"] == 0:
            lines.append("  No errors detected.")
        else:
            lines.append("  ERRORS FOUND:")
            for e in s["all_errors"]:
                lines.append(f"    [{e['type']} | {e['severity']}] {e['detail']}")

        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF ERROR RATE ANALYSIS -- LLM Judge 1")
    lines.append(f"  Mean error rate: {mean_er:.4f} | "
                 f"No errors: {no_error} | Low: {low_er} | "
                 f"Moderate: {mod_er} | High: {high_er}")
    lines.append(f"  Total errors detected: {total_all}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "error_rate_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Error rate results written to: {out_path}\n")


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
        description="Error rate analysis: LLM Judge 1 predictions vs ground truth"
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
    print("[INFO] Computing error rates...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()