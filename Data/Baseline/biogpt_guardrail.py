"""
BMEN-499 AlphaFold -- BioGPT Guardrail System
----------------------------------------------
Purpose:
    Applies a multi-layer guardrail to BioGPT generated answers
    to detect and reduce hallucination before answers are used
    as the baseline in the QA evaluation pipeline.

What is a Guardrail?
    A guardrail is a validation layer that sits between BioGPT's
    raw output and the evaluation pipeline. It checks each generated
    answer against the DisProt ground truth and either:
      - PASSES  the answer if it is sufficiently grounded
      - WARNS   if the answer has minor hallucination risk
      - BLOCKS  the answer if hallucination risk is too high
                and replaces it with a grounded fallback

Five Guardrail Layers:
    Layer 1 -- NUMERIC GUARD
               Checks all numbers in BioGPT output against
               known DisProt statistics. Flags numbers that
               deviate more than 10% from ground truth values.

    Layer 2 -- HALLUCINATION GUARD
               Checks if BioGPT introduces claims not grounded
               in the DisProt knowledge base. High ratio of
               ungrounded content triggers a block.

    Layer 3 -- CONTRADICTION GUARD
               Detects direct contradictions between BioGPT
               output and ground truth (directional opposites,
               negated claims).

    Layer 4 -- DOMAIN TERM GUARD
               Ensures BioGPT uses correct biomedical terminology.
               Flags missing or incorrect domain terms.

    Layer 5 -- CONFIDENCE GUARD
               Checks BioGPT output length and specificity.
               Very short or vague answers get flagged as
               low-confidence and replaced with GT fallback.

Guardrail Actions:
    PASS    -- answer is grounded, use as-is
    WARN    -- minor issues, use with warning annotation
    BLOCK   -- high hallucination risk, replace with GT fallback

Output: guardrail_results.txt (saved to same folder)

Usage:
    python biogpt_guardrail.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python biogpt_guardrail.py --demo
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
# 2. COMPUTE STATS + GROUND TRUTH
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

def get_ground_truth(question, stats):
    q = question.lower()
    for keywords, fn in GT_RULES:
        if any(kw in q for kw in keywords):
            try:
                return fn(stats)
            except:
                pass
    return (f"DisProt dataset summary: {stats['total_proteins']:,} proteins, "
            f"mean disorder={stats['mean_disorder']:.3f}, "
            f"mean region length={stats['mean_region_length']:.1f} aa.")


# Simulated BioGPT answers -- in production replace with actual BioGPT output
def get_biogpt_answer(question, stats):
    """
    In production this function calls BioGPT:
        from transformers import BioGptTokenizer, BioGptForCausalLM
        outputs = model.generate(...)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    For demo purposes we simulate answers with varying hallucination levels.
    """
    q = question.lower()
    simulated = {
        "0.5":      f"A disorder score above 0.5 is generally considered reliable for IDR classification. Studies show approximately {stats['pct_above_0.5']:.1f}% of proteins exceed this threshold. However the exact cutoff may vary by predictor used.",
        "short":    f"Short IDRs under 10 residues present challenges for prediction algorithms. The average disordered region is around {stats['mean_region_length']:.0f} residues. Confidence scores typically decrease significantly for very short regions due to limited context.",
        "proline":  f"Proline and glycine rich regions show elevated disorder scores. Mean proline content is approximately {stats['mean_proline']*100:.1f}% in disordered proteins. These amino acids destabilize secondary structure elements through steric clashes.",
        "sliding":  f"A sliding window approach can smooth local fluctuations in disorder scores. Window sizes of 5 to 9 residues are commonly used. However excessively large windows may mask genuine short disordered regions in the sequence.",
        "pfam":     f"Proteins containing Pfam domains often show mixed structural profiles. About {stats['pct_with_pfam']:.0f}% of annotated proteins have both structured domains and disordered regions. The presence of a Pfam domain does not preclude intrinsic disorder elsewhere.",
        "alphafold":f"AlphaFold confidence scores correlate inversely with disorder propensity. Regions with pLDDT below 70 are often intrinsically disordered. The pLDDT metric was trained on structured proteins so very low scores strongly suggest disorder.",
    }
    for kw, answer in simulated.items():
        if kw in q:
            return answer
    return f"Intrinsically disordered proteins lack stable tertiary structure under physiological conditions. The DisProt database catalogues {stats['total_proteins']:,} experimentally validated disordered proteins with diverse functional roles."


# =============================================================
# 3. GUARDRAIL LAYERS
# =============================================================

DOMAIN_TERMS = [
    "disorder","disordered","idr","idp","plddt","alphafold","pfam",
    "proline","glycine","residue","amino","backbone","threshold","cutoff",
    "disprot","intrinsic","region","sequence","confidence","annotated"
]

DIRECTIONAL_PAIRS = [
    ("reliable",    "unreliable"),
    ("strong",      "weak"),
    ("increases",   "decreases"),
    ("above",       "below"),
    ("high",        "low"),
    ("consistent",  "inconsistent"),
    ("predicts",    "does not predict"),
    ("correlates",  "does not correlate"),
]

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    sw = {"a","an","the","is","are","was","were","be","been","of","in","on",
          "at","to","for","with","by","from","and","or","but","not","this",
          "that","it","its","they","we","as"}
    return [w for w in normalize(text).split() if w not in sw and len(w) > 1]


# --- Layer 1: Numeric Guard ----------------------------------
def layer1_numeric(biogpt_ans, gt, stats):
    """
    Check if BioGPT numbers are grounded in DisProt statistics.
    Tolerance: 10% relative difference.
    """
    issues     = []
    gt_nums    = extract_numbers(gt)
    biogpt_nums = extract_numbers(biogpt_ans)

    # Known valid DisProt numbers
    valid_nums = set(round(n, 1) for n in gt_nums)
    valid_nums.update([
        round(stats["total_proteins"], -2),
        round(stats["mean_disorder"], 2),
        round(stats["pct_above_0.5"], 1),
        round(stats["pct_above_0.3"], 1),
        round(stats["mean_region_length"], 1),
        round(stats["mean_proline"]*100, 1),
        round(stats["mean_glycine"]*100, 1),
        round(stats["pct_with_pfam"], 1),
    ])

    hallucinated_nums = []
    for bn in biogpt_nums:
        if bn <= 1.0:
            continue  # skip small decimals like 0.5 threshold
        grounded = any(
            abs(bn - vn) / max(abs(vn), 1e-9) < 0.10
            for vn in valid_nums if vn > 0
        )
        if not grounded:
            hallucinated_nums.append(bn)

    if hallucinated_nums:
        issues.append({
            "layer":    1,
            "type":     "NUMERIC",
            "severity": "HIGH" if len(hallucinated_nums) > 2 else "MEDIUM",
            "detail":   f"Ungrounded numbers in BioGPT output: {hallucinated_nums[:4]}. "
                        f"Valid DisProt numbers include: {sorted(list(valid_nums))[:6]}",
            "fix":      f"Replace ungrounded numbers with DisProt values."
        })

    return issues


# --- Layer 2: Hallucination Guard ----------------------------
def layer2_hallucination(biogpt_ans, gt):
    """
    Check ratio of BioGPT content not grounded in GT.
    High ungrounded ratio = hallucination risk.
    """
    issues      = []
    biogpt_toks = set(tokenize(biogpt_ans))
    gt_toks     = set(tokenize(gt))

    ungrounded  = biogpt_toks - gt_toks
    halluc_ratio = len(ungrounded) / max(len(biogpt_toks), 1)

    if halluc_ratio > 0.60:
        sample = sorted(list(ungrounded))[:8]
        issues.append({
            "layer":    2,
            "type":     "HALLUCINATION",
            "severity": "HIGH",
            "detail":   f"{halluc_ratio*100:.1f}% of BioGPT content not in GT. "
                        f"Ungrounded terms: {', '.join(sample)}",
            "fix":      "Block and replace with ground truth answer."
        })
    elif halluc_ratio > 0.40:
        issues.append({
            "layer":    2,
            "type":     "HALLUCINATION",
            "severity": "MEDIUM",
            "detail":   f"{halluc_ratio*100:.1f}% of content ungrounded in GT.",
            "fix":      "Warn and annotate answer with grounding caveat."
        })

    return issues


# --- Layer 3: Contradiction Guard ----------------------------
def layer3_contradiction(biogpt_ans, gt):
    """
    Detect direct contradictions between BioGPT answer and GT.
    """
    issues = []
    bl, gl = biogpt_ans.lower(), gt.lower()

    for pos, neg in DIRECTIONAL_PAIRS:
        gt_pos   = pos in gl
        gt_neg   = neg in gl
        bio_pos  = pos in bl
        bio_neg  = neg in bl

        if gt_pos and bio_neg and not bio_pos:
            issues.append({
                "layer":    3,
                "type":     "CONTRADICTION",
                "severity": "HIGH",
                "detail":   f"GT uses '{pos}' but BioGPT uses '{neg}'.",
                "fix":      "Replace contradicting phrase with GT-aligned language."
            })
        elif gt_neg and bio_pos and not bio_neg:
            issues.append({
                "layer":    3,
                "type":     "CONTRADICTION",
                "severity": "HIGH",
                "detail":   f"GT uses '{neg}' but BioGPT uses '{pos}'.",
                "fix":      "Replace contradicting phrase with GT-aligned language."
            })

    return issues


# --- Layer 4: Domain Term Guard ------------------------------
def layer4_domain_terms(biogpt_ans, gt):
    """
    Check that BioGPT uses correct biomedical terminology from GT.
    """
    issues    = []
    gt_terms  = [t for t in DOMAIN_TERMS if t in gt.lower()]
    bio_terms = [t for t in DOMAIN_TERMS if t in biogpt_ans.lower()]
    missing   = [t for t in gt_terms if t not in bio_terms]

    if len(missing) > 4:
        issues.append({
            "layer":    4,
            "type":     "TERMINOLOGY",
            "severity": "MEDIUM",
            "detail":   f"Missing {len(missing)} domain terms from GT: {', '.join(missing[:6])}",
            "fix":      "Inject missing terminology into BioGPT answer."
        })
    elif len(missing) > 2:
        issues.append({
            "layer":    4,
            "type":     "TERMINOLOGY",
            "severity": "LOW",
            "detail":   f"Missing {len(missing)} domain terms: {', '.join(missing[:4])}",
            "fix":      "Consider adding missing terms for completeness."
        })

    return issues


# --- Layer 5: Confidence Guard -------------------------------
def layer5_confidence(biogpt_ans, question):
    """
    Check BioGPT answer length and specificity.
    Very short or vague answers indicate low confidence.
    """
    issues     = []
    word_count = len(biogpt_ans.split())
    sentences  = [s for s in biogpt_ans.split(".") if len(s.strip()) > 5]
    q_toks     = set(tokenize(question))
    ans_toks   = set(tokenize(biogpt_ans))
    relevance  = len(q_toks & ans_toks) / max(len(q_toks), 1)

    if word_count < 20:
        issues.append({
            "layer":    5,
            "type":     "LOW_CONFIDENCE",
            "severity": "HIGH",
            "detail":   f"Answer too short ({word_count} words). Likely low-confidence output.",
            "fix":      "Block and replace with GT fallback answer."
        })
    elif word_count < 40:
        issues.append({
            "layer":    5,
            "type":     "LOW_CONFIDENCE",
            "severity": "MEDIUM",
            "detail":   f"Answer is short ({word_count} words). May lack sufficient detail.",
            "fix":      "Warn and supplement with GT information."
        })

    if relevance < 0.2:
        issues.append({
            "layer":    5,
            "type":     "LOW_RELEVANCE",
            "severity": "MEDIUM",
            "detail":   f"Only {relevance*100:.0f}% of question keywords in answer. "
                        f"Answer may be off-topic.",
            "fix":      "Warn and flag for manual review."
        })

    return issues


# =============================================================
# 4. GUARDRAIL ENGINE
# =============================================================

def run_guardrail(question, biogpt_ans, gt, stats):
    """
    Run all 5 guardrail layers and compute overall verdict.

    Verdict logic:
      BLOCK -- any HIGH severity issue found
      WARN  -- any MEDIUM severity issue, no HIGH
      PASS  -- only LOW or no issues
    """
    all_issues = []
    all_issues.extend(layer1_numeric(biogpt_ans, gt, stats))
    all_issues.extend(layer2_hallucination(biogpt_ans, gt))
    all_issues.extend(layer3_contradiction(biogpt_ans, gt))
    all_issues.extend(layer4_domain_terms(biogpt_ans, gt))
    all_issues.extend(layer5_confidence(biogpt_ans, question))

    high_count   = sum(1 for i in all_issues if i["severity"] == "HIGH")
    medium_count = sum(1 for i in all_issues if i["severity"] == "MEDIUM")
    low_count    = sum(1 for i in all_issues if i["severity"] == "LOW")

    # Compute hallucination risk score (0.0 - 1.0)
    risk_score = round(min(1.0, (high_count * 0.3 + medium_count * 0.15 + low_count * 0.05)), 4)

    if high_count > 0:
        verdict         = "BLOCK"
        final_answer    = gt   # replace with ground truth
        verdict_reason  = f"{high_count} HIGH severity issues detected"
    elif medium_count > 0:
        verdict         = "WARN"
        final_answer    = biogpt_ans + f" [GUARDRAIL WARNING: {medium_count} issues flagged]"
        verdict_reason  = f"{medium_count} MEDIUM severity issues detected"
    else:
        verdict         = "PASS"
        final_answer    = biogpt_ans
        verdict_reason  = "All guardrail layers passed"

    return {
        "verdict":        verdict,
        "verdict_reason": verdict_reason,
        "risk_score":     risk_score,
        "issues":         all_issues,
        "high_count":     high_count,
        "medium_count":   medium_count,
        "low_count":      low_count,
        "original_answer": biogpt_ans,
        "final_answer":    final_answer,
        "was_replaced":    verdict == "BLOCK",
    }


# =============================================================
# 5. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt        = get_ground_truth(q, stats)
        biogpt_ans = get_biogpt_answer(q, stats)
        gr        = run_guardrail(q, biogpt_ans, gt, stats)

        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "biogpt_ans":   biogpt_ans,
            "guardrail":    gr,
        })

        replaced = " [REPLACED]" if gr["was_replaced"] else ""
        print(f"  Q{i:3d} | {gr['verdict']:<6} | Risk={gr['risk_score']:.4f} | "
              f"Issues={len(gr['issues'])} "
              f"(H={gr['high_count']},M={gr['medium_count']},L={gr['low_count']}) "
              f"| {gr['verdict_reason']}{replaced}")

    return results


# =============================================================
# 6. WRITE guardrail_results.txt
# =============================================================

def write_results(results, stats):
    verdicts   = [r["guardrail"]["verdict"]    for r in results]
    risks      = [r["guardrail"]["risk_score"] for r in results]
    replaced   = [r["guardrail"]["was_replaced"] for r in results]

    passes     = verdicts.count("PASS")
    warns      = verdicts.count("WARN")
    blocks     = verdicts.count("BLOCK")
    n_replaced = sum(replaced)
    mean_risk  = sum(risks) / len(risks)

    all_issues = [i for r in results for i in r["guardrail"]["issues"]]
    layer_counts = Counter(i["layer"] for i in all_issues)
    type_counts  = Counter(i["type"]  for i in all_issues)

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- BioGPT Guardrail System")
    lines.append("  Baseline : BioGPT (microsoft/biogpt)")
    lines.append("  Guard    : 5-Layer Hallucination Detection + GT Grounding")
    lines.append(f"  Dataset  : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS THE GUARDRAIL SYSTEM?")
    lines.append("-" * 70)
    lines.append("  A guardrail validates BioGPT answers against DisProt ground")
    lines.append("  truth BEFORE they enter the evaluation pipeline.")
    lines.append("  This reduces hallucination risk in the baseline answers.")
    lines.append("")
    lines.append("  5 GUARDRAIL LAYERS:")
    lines.append("  Layer 1 -- NUMERIC GUARD")
    lines.append("    Checks BioGPT numbers against known DisProt statistics.")
    lines.append("    Tolerance: 10%. Flags ungrounded numbers.")
    lines.append("")
    lines.append("  Layer 2 -- HALLUCINATION GUARD")
    lines.append("    Checks ratio of BioGPT content not found in GT.")
    lines.append("    >60% ungrounded = HIGH risk, >40% = MEDIUM risk.")
    lines.append("")
    lines.append("  Layer 3 -- CONTRADICTION GUARD")
    lines.append("    Detects opposite directional words vs GT.")
    lines.append("    e.g. GT says 'reliable' but BioGPT says 'unreliable'.")
    lines.append("")
    lines.append("  Layer 4 -- DOMAIN TERM GUARD")
    lines.append("    Checks for missing biomedical terms present in GT.")
    lines.append("    Flags answers missing IDR, pLDDT, Pfam, etc.")
    lines.append("")
    lines.append("  Layer 5 -- CONFIDENCE GUARD")
    lines.append("    Checks answer length and question relevance.")
    lines.append("    <20 words = HIGH risk, <40 words = MEDIUM risk.")
    lines.append("")
    lines.append("  VERDICTS:")
    lines.append("    PASS  -- no issues or LOW only -- use BioGPT answer as-is")
    lines.append("    WARN  -- MEDIUM issues -- use with warning annotation")
    lines.append("    BLOCK -- HIGH issues  -- replace with GT fallback answer")
    lines.append("")
    lines.append("OVERALL GUARDRAIL RESULTS")
    lines.append("-" * 70)
    lines.append(f"  PASS  : {passes:3d} questions ({passes/len(results)*100:.1f}%)")
    lines.append(f"  WARN  : {warns:3d} questions ({warns/len(results)*100:.1f}%)")
    lines.append(f"  BLOCK : {blocks:3d} questions ({blocks/len(results)*100:.1f}%)")
    lines.append(f"  Answers replaced with GT : {n_replaced}")
    lines.append(f"  Mean hallucination risk  : {mean_risk:.4f}")
    lines.append(f"  Total issues detected    : {len(all_issues)}")
    lines.append("")
    lines.append("  Issues by layer:")
    for layer, name in [(1,"Numeric"),(2,"Hallucination"),(3,"Contradiction"),
                         (4,"Domain Terms"),(5,"Confidence")]:
        count = layer_counts.get(layer, 0)
        bar   = "#" * count + "." * max(0, 10 - count)
        lines.append(f"    Layer {layer} {name:<15} [{bar}] {count} issues")
    lines.append("")
    lines.append("  Issues by type:")
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * c + "." * max(0, 10 - c)
        lines.append(f"    {t:<20} [{bar}] {c} issues")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION GUARDRAIL REPORT")
    lines.append("=" * 70)

    for r in results:
        g = r["guardrail"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Verdict    : {g['verdict']}  --  {g['verdict_reason']}")
        lines.append(f"  Risk score : {g['risk_score']:.4f}")
        lines.append(f"  Replaced   : {'YES -- GT fallback used' if g['was_replaced'] else 'NO -- BioGPT answer kept'}")
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65]
                      for i in range(0, len(r["ground_truth"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  BIOGPT ORIGINAL ANSWER:")
        for chunk in [r["biogpt_ans"][i:i+65]
                      for i in range(0, len(r["biogpt_ans"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  FINAL ANSWER (after guardrail):")
        for chunk in [g["final_answer"][i:i+65]
                      for i in range(0, len(g["final_answer"]), 65)]:
            lines.append(f"    {chunk}")
        lines.append("")

        if g["issues"]:
            lines.append("  GUARDRAIL ISSUES DETECTED:")
            for issue in g["issues"]:
                lines.append(f"    [Layer {issue['layer']} | {issue['type']} | {issue['severity']}]")
                lines.append(f"      Issue : {issue['detail']}")
                lines.append(f"      Fix   : {issue['fix']}")
        else:
            lines.append("  No guardrail issues detected.")
        lines.append("-" * 70)

    lines.append("")
    lines.append("  HOW TO INTEGRATE WITH BIOGPT PIPELINE")
    lines.append("-" * 70)
    lines.append("  In groundtruth_answers.py or qa_pipeline.py, wrap")
    lines.append("  BioGPT output with the guardrail before evaluation:")
    lines.append("")
    lines.append("  from biogpt_guardrail import run_guardrail, get_ground_truth")
    lines.append("")
    lines.append("  # After BioGPT generates answer:")
    lines.append("  gt     = get_ground_truth(question, stats)")
    lines.append("  result = run_guardrail(question, biogpt_answer, gt, stats)")
    lines.append("")
    lines.append("  # Use guardrailed answer instead of raw BioGPT output:")
    lines.append("  safe_answer = result['final_answer']")
    lines.append("  verdict     = result['verdict']  # PASS / WARN / BLOCK")
    lines.append("  risk        = result['risk_score']")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF GUARDRAIL REPORT")
    lines.append(f"  PASS: {passes} | WARN: {warns} | BLOCK: {blocks} | "
                 f"Replaced: {n_replaced} | Mean risk: {mean_risk:.4f}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "guardrail_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Guardrail results written to: {out_path}\n")


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
        description="BioGPT Guardrail: hallucination detection against DisProt GT"
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
    print("[INFO] Running BioGPT guardrail system...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()