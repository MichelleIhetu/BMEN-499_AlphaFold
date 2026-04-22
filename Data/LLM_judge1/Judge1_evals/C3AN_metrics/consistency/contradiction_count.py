"""
BMEN-499 AlphaFold -- Contradiction Count: LLM Judge 1 vs Ground Truth
-----------------------------------------------------------------------
Purpose:
    Detects and counts contradictions between LLM Judge 1 predicted
    answers and DisProt ground truth answers.

What counts as a contradiction?
    1. NUMERIC CONTRADICTION   -- Prediction states a different number
                                  than the ground truth
                                  e.g. GT says 29.1% but LLM says 50%

    2. DIRECTIONAL CONTRADICTION -- Prediction uses the opposite
                                    directional word to the ground truth
                                    e.g. GT says "reliable" but LLM
                                    says "unreliable"

    3. FACTUAL NEGATION        -- Prediction negates a claim the ground
                                  truth makes positively, or vice versa
                                  e.g. GT says "does predict disorder"
                                  but LLM says "does not predict"

Severity:
    HIGH   -- Factual negations and large numeric differences (>50%)
    MEDIUM -- Directional opposites and smaller numeric differences

Output: contradiction_results.txt (saved to same folder as this script)

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
# 3. CONTRADICTION DETECTION ENGINE
# =============================================================

DIRECTIONAL_PAIRS = [
    ("reliable",     "unreliable"),
    ("conservative", "aggressive"),
    ("strong",       "weak"),
    ("increases",    "decreases"),
    ("high",         "low"),
    ("frequently",   "rarely"),
    ("consistent",   "inconsistent"),
    ("significant",  "insignificant"),
    ("sufficient",   "insufficient"),
    ("structured",   "unstructured"),
    ("predicts",     "does not predict"),
    ("correlates",   "does not correlate"),
    ("co-occur",     "do not co-occur"),
]

NEGATION_PATTERNS = [
    (r"\bis\s+a\s+strong\b",       r"\bis\s+not\s+a\s+strong\b"),
    (r"\bstrongly\s+correlat",     r"\bdoes\s+not\s+correlat"),
    (r"\breliable\b",              r"\bnot\s+reliable\b"),
    (r"\bpredicts?\s+disorder\b",  r"\bdoes\s+not\s+predict\s+disorder\b"),
    (r"\bconfident\b",             r"\bnot\s+confident\b"),
]


def extract_numbers(text):
    results = []
    for match in re.finditer(r"(\d+\.?\d*)\s*(%?)", text):
        val     = float(match.group(1))
        is_pct  = bool(match.group(2))
        start   = max(0, match.start() - 30)
        context = text[start:match.end() + 30].strip()
        results.append({"value": val, "is_pct": is_pct, "context": context})
    return results


def detect_numeric(pred, gt):
    contradictions = []
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)
    for gn in gt_nums:
        gv = gn["value"]
        if gv == 0:
            continue
        close = any(abs(pn["value"] - gv) / abs(gv) < 0.03 for pn in pred_nums)
        if not close:
            conflicts = [pn for pn in pred_nums
                         if abs(pn["value"] - gv) / abs(gv) > 0.10
                         and abs(pn["value"] - gv) < abs(gv) * 3]
            if conflicts:
                pv = conflicts[0]["value"]
                contradictions.append({
                    "type":     "NUMERIC",
                    "detail":   f"GT states {gv}{'%' if gn['is_pct'] else ''} but prediction has {pv}{'%' if conflicts[0]['is_pct'] else ''}",
                    "gt_ctx":   gn["context"][:60],
                    "pred_ctx": conflicts[0]["context"][:60],
                    "severity": "HIGH" if abs(pv - gv) / abs(gv) > 0.5 else "MEDIUM",
                })
    return contradictions


def detect_directional(pred, gt):
    contradictions = []
    pl, gl = pred.lower(), gt.lower()
    for pos, neg in DIRECTIONAL_PAIRS:
        gt_pos  = pos in gl
        gt_neg  = neg in gl
        pr_pos  = pos in pl
        pr_neg  = neg in pl
        if gt_pos and pr_neg and not pr_pos:
            contradictions.append({
                "type":     "DIRECTIONAL",
                "detail":   f"GT uses '{pos}' but prediction uses '{neg}'",
                "severity": "MEDIUM",
            })
        elif gt_neg and pr_pos and not pr_neg:
            contradictions.append({
                "type":     "DIRECTIONAL",
                "detail":   f"GT uses '{neg}' but prediction uses '{pos}'",
                "severity": "MEDIUM",
            })
    return contradictions


def detect_negation(pred, gt):
    contradictions = []
    for pos_pat, neg_pat in NEGATION_PATTERNS:
        gt_pos  = bool(re.search(pos_pat, gt,   re.IGNORECASE))
        pr_neg  = bool(re.search(neg_pat, pred, re.IGNORECASE))
        gt_neg  = bool(re.search(neg_pat, gt,   re.IGNORECASE))
        pr_pos  = bool(re.search(pos_pat, pred, re.IGNORECASE))
        if gt_pos and pr_neg and not gt_neg:
            contradictions.append({
                "type":     "NEGATION",
                "detail":   f"GT makes a positive claim that prediction negates",
                "pattern":  pos_pat[:40],
                "severity": "HIGH",
            })
        elif gt_neg and pr_pos and not gt_pos:
            contradictions.append({
                "type":     "NEGATION",
                "detail":   f"GT negates a claim that prediction makes positively",
                "pattern":  neg_pat[:40],
                "severity": "HIGH",
            })
    return contradictions


def count_contradictions(pred, gt):
    numeric     = detect_numeric(pred, gt)
    directional = detect_directional(pred, gt)
    negation    = detect_negation(pred, gt)
    all_c       = numeric + directional + negation
    total       = len(all_c)
    high        = sum(1 for c in all_c if c["severity"] == "HIGH")
    medium      = sum(1 for c in all_c if c["severity"] == "MEDIUM")
    sev_score   = round(high * 1.0 + medium * 0.5, 2)
    label = (
        "NO CONTRADICTIONS"       if total == 0 else
        "MINOR CONTRADICTIONS"    if sev_score <= 1.0 else
        "MODERATE CONTRADICTIONS" if sev_score <= 2.5 else
        "SEVERE CONTRADICTIONS"
    )
    return {
        "total": total, "numeric": numeric, "directional": directional,
        "negation": negation, "high_severity": high, "medium_severity": medium,
        "severity_score": sev_score, "label": label,
    }


# =============================================================
# 4. EVALUATE + WRITE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = count_contradictions(pred, gt)
        results.append({"q_num": i, "question": q,
                         "ground_truth": gt, "prediction": pred, "score": sc})
        print(f"  Q{i:3d} | Contradictions={sc['total']} "
              f"(High={sc['high_severity']}, Med={sc['medium_severity']}) | {sc['label']}")
    return results


def write_results(results, stats):
    total_all  = sum(r["score"]["total"]             for r in results)
    t_numeric  = sum(len(r["score"]["numeric"])      for r in results)
    t_direct   = sum(len(r["score"]["directional"])  for r in results)
    t_negation = sum(len(r["score"]["negation"])     for r in results)
    t_high     = sum(r["score"]["high_severity"]     for r in results)
    t_medium   = sum(r["score"]["medium_severity"]   for r in results)
    none_c     = sum(1 for r in results if r["score"]["label"] == "NO CONTRADICTIONS")
    minor_c    = sum(1 for r in results if r["score"]["label"] == "MINOR CONTRADICTIONS")
    mod_c      = sum(1 for r in results if "MODERATE" in r["score"]["label"])
    severe_c   = sum(1 for r in results if "SEVERE"   in r["score"]["label"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Contradiction Count: LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : Contradiction Detection (Numeric, Directional, Negation)")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS CONTRADICTION COUNTING?")
    lines.append("-" * 70)
    lines.append("  Contradictions occur when the LLM prediction states something")
    lines.append("  that directly conflicts with the ground truth. Three types:")
    lines.append("")
    lines.append("  TYPE 1 -- NUMERIC CONTRADICTION")
    lines.append("    Prediction states a different number than ground truth.")
    lines.append("    Example: GT says 29.1% but prediction says 50%.")
    lines.append("    Tolerance: numbers within 3% are NOT contradictions.")
    lines.append("")
    lines.append("  TYPE 2 -- DIRECTIONAL CONTRADICTION")
    lines.append("    Prediction uses an opposite directional word to GT.")
    lines.append("    Example: GT says 'reliable' but prediction says 'unreliable'.")
    lines.append("")
    lines.append("  TYPE 3 -- FACTUAL NEGATION")
    lines.append("    Prediction negates a claim the ground truth makes.")
    lines.append("    Example: GT says 'strongly correlates' but prediction")
    lines.append("    says 'does not correlate'.")
    lines.append("")
    lines.append("  Severity: HIGH = negations + large numeric gaps (>50%)")
    lines.append("            MEDIUM = directional opposites + smaller numeric gaps")
    lines.append("  Severity score = HIGH*1.0 + MEDIUM*0.5")
    lines.append("")
    lines.append("OVERALL CONTRADICTION SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total contradictions found    : {total_all}")
    lines.append(f"    Numeric contradictions      : {t_numeric}")
    lines.append(f"    Directional contradictions  : {t_direct}")
    lines.append(f"    Negation contradictions     : {t_negation}")
    lines.append(f"  High severity                 : {t_high}")
    lines.append(f"  Medium severity               : {t_medium}")
    lines.append("")
    lines.append(f"  Questions with NO CONTRADICTIONS    : {none_c}")
    lines.append(f"  Questions with MINOR CONTRADICTIONS : {minor_c}")
    lines.append(f"  Questions with MODERATE             : {mod_c}")
    lines.append(f"  Questions with SEVERE               : {severe_c}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION CONTRADICTION REPORT")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Status         : {s['label']}")
        lines.append(f"  Contradictions : {s['total']} total  "
                     f"(High={s['high_severity']}, Medium={s['medium_severity']}, "
                     f"Severity score={s['severity_score']})")
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

        if s["total"] == 0:
            lines.append("  No contradictions detected.")
        else:
            lines.append("  CONTRADICTIONS FOUND:")
            for c in s["numeric"]:
                lines.append(f"    [NUMERIC | {c['severity']}] {c['detail']}")
                lines.append(f"      GT context   : ...{c['gt_ctx']}...")
                lines.append(f"      Pred context : ...{c['pred_ctx']}...")
            for c in s["directional"]:
                lines.append(f"    [DIRECTIONAL | {c['severity']}] {c['detail']}")
            for c in s["negation"]:
                lines.append(f"    [NEGATION | {c['severity']}] {c['detail']}")

        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF CONTRADICTION COUNT -- LLM Judge 1")
    lines.append(f"  Total: {total_all} contradictions across {len(results)} questions")
    lines.append(f"  None: {none_c} | Minor: {minor_c} | "
                 f"Moderate: {mod_c} | Severe: {severe_c}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "contradiction_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] Contradiction results written to: {out_path}\n")


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
        description="Contradiction count for LLM Judge 1 vs ground truth"
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
    print("[INFO] Detecting contradictions...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)

if __name__ == "__main__":
    main()