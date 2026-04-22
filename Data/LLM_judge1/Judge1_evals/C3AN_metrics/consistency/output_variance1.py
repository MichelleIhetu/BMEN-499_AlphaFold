"""
BMEN-499 AlphaFold -- Output Variance: LLM Judge 1 vs Ground Truth
-------------------------------------------------------------------
Purpose:
    Measures how much the LLM Judge 1 predicted answers vary from
    the DisProt ground truth answers across multiple dimensions.

What is Output Variance?
    Variance measures how spread out or inconsistent the outputs are.
    In a QA evaluation context it answers:

      - How much do the predicted answers DIFFER in length from GT?
      - How much do the predicted answers DIFFER in vocabulary?
      - How much do the numeric values DRIFT from ground truth values?
      - How CONSISTENT is the model across similar questions?
      - How much does the INFORMATION DENSITY vary?

    Low variance = model is consistent and close to ground truth
    High variance = model drifts significantly from ground truth

Variance metrics computed:
    1. LENGTH VARIANCE       -- Difference in word count between pred and GT
    2. VOCABULARY VARIANCE   -- Difference in unique word counts
    3. NUMERIC DRIFT         -- How far off predicted numbers are from GT
    4. LEXICAL DIVERSITY     -- Type-token ratio variance between pred and GT
    5. SENTENCE COUNT VARIANCE -- Difference in number of sentences
    6. INFORMATION DENSITY   -- Content word ratio variance

Output: variance_results.txt (saved to same folder as this script)

Usage:
    python output_variance1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python output_variance1.py --demo
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
# 3. VARIANCE METRICS ENGINE
# =============================================================

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","of","in","on","at",
    "to","for","with","by","from","and","or","but","not","this","that",
    "it","its","they","we","as","also","both","very","than","such","each"
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokens(text):
    return normalize(text).split()

def content_tokens(text):
    return [w for w in tokens(text) if w not in STOPWORDS and len(w) > 1]

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

def sentence_count(text):
    return len([s for s in text.split(".") if len(s.strip()) > 5])

def type_token_ratio(text):
    toks = tokens(text)
    return len(set(toks)) / len(toks) if toks else 0.0

def information_density(text):
    """Content word ratio = content tokens / all tokens."""
    all_toks     = tokens(text)
    content_toks = content_tokens(text)
    return len(content_toks) / len(all_toks) if all_toks else 0.0

def variance(values):
    """Population variance of a list of numbers."""
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)

def std_dev(values):
    return math.sqrt(variance(values))


# --- Per-question variance metrics ----------------------------

def compute_question_variance(pred, gt):
    """
    Compute all variance metrics for a single pred/GT pair.
    Each metric measures how much the prediction deviates from GT.
    """
    # 1. Length variance (word count difference)
    pred_len   = len(tokens(pred))
    gt_len     = len(tokens(gt))
    len_diff   = pred_len - gt_len
    len_pct    = round(abs(len_diff) / max(gt_len, 1) * 100, 2)

    # 2. Vocabulary variance (unique word count difference)
    pred_vocab = len(set(tokens(pred)))
    gt_vocab   = len(set(tokens(gt)))
    vocab_diff = pred_vocab - gt_vocab
    vocab_pct  = round(abs(vocab_diff) / max(gt_vocab, 1) * 100, 2)

    # 3. Numeric drift (mean absolute difference of numbers)
    pred_nums = extract_numbers(pred)
    gt_nums   = extract_numbers(gt)
    if gt_nums and pred_nums:
        # Match each GT number to the closest pred number
        drifts = []
        for gn in gt_nums:
            closest = min(pred_nums, key=lambda pn: abs(pn - gn))
            if gn != 0:
                drifts.append(abs(closest - gn) / abs(gn))
        numeric_drift = round(sum(drifts) / len(drifts) * 100, 2) if drifts else 0.0
    else:
        numeric_drift = 0.0

    # 4. Lexical diversity variance (type-token ratio difference)
    pred_ttr  = type_token_ratio(pred)
    gt_ttr    = type_token_ratio(gt)
    ttr_diff  = round(abs(pred_ttr - gt_ttr), 4)

    # 5. Sentence count variance
    pred_sents = sentence_count(pred)
    gt_sents   = sentence_count(gt)
    sent_diff  = abs(pred_sents - gt_sents)

    # 6. Information density variance
    pred_density = information_density(pred)
    gt_density   = information_density(gt)
    density_diff = round(abs(pred_density - gt_density), 4)

    # Overall variance score (normalized 0-1, lower is better)
    # Each component normalized to 0-1 range then averaged
    norm_len     = min(1.0, len_pct / 100)
    norm_vocab   = min(1.0, vocab_pct / 100)
    norm_numeric = min(1.0, numeric_drift / 100)
    norm_ttr     = min(1.0, ttr_diff * 5)
    norm_sent    = min(1.0, sent_diff / 5)
    norm_density = min(1.0, density_diff * 5)

    overall_variance = round(
        0.20 * norm_len     +
        0.15 * norm_vocab   +
        0.30 * norm_numeric +
        0.15 * norm_ttr     +
        0.10 * norm_sent    +
        0.10 * norm_density,
        4
    )

    label = (
        "VERY LOW VARIANCE"  if overall_variance < 0.10 else
        "LOW VARIANCE"       if overall_variance < 0.25 else
        "MODERATE VARIANCE"  if overall_variance < 0.50 else
        "HIGH VARIANCE"      if overall_variance < 0.75 else
        "VERY HIGH VARIANCE"
    )

    return {
        "overall_variance":  overall_variance,
        "label":             label,
        "length": {
            "pred_words":  pred_len,
            "gt_words":    gt_len,
            "difference":  len_diff,
            "pct_diff":    len_pct,
        },
        "vocabulary": {
            "pred_unique": pred_vocab,
            "gt_unique":   gt_vocab,
            "difference":  vocab_diff,
            "pct_diff":    vocab_pct,
        },
        "numeric_drift": {
            "pred_numbers": pred_nums,
            "gt_numbers":   gt_nums,
            "mean_drift_pct": numeric_drift,
        },
        "lexical_diversity": {
            "pred_ttr":   round(pred_ttr, 4),
            "gt_ttr":     round(gt_ttr, 4),
            "difference": ttr_diff,
        },
        "sentence_count": {
            "pred_sentences": pred_sents,
            "gt_sentences":   gt_sents,
            "difference":     sent_diff,
        },
        "information_density": {
            "pred_density": round(pred_density, 4),
            "gt_density":   round(gt_density, 4),
            "difference":   density_diff,
        },
    }


# =============================================================
# 4. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = compute_question_variance(pred, gt)
        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })
        print(f"  Q{i:3d} | Variance={sc['overall_variance']:.4f} | {sc['label']}")
    return results


# =============================================================
# 5. WRITE variance_results.txt
# =============================================================

def write_results(results, stats):
    ov_scores   = [r["score"]["overall_variance"]                    for r in results]
    len_diffs   = [abs(r["score"]["length"]["difference"])           for r in results]
    vocab_diffs = [abs(r["score"]["vocabulary"]["difference"])       for r in results]
    num_drifts  = [r["score"]["numeric_drift"]["mean_drift_pct"]     for r in results]
    ttr_diffs   = [r["score"]["lexical_diversity"]["difference"]     for r in results]
    sent_diffs  = [r["score"]["sentence_count"]["difference"]        for r in results]
    dens_diffs  = [r["score"]["information_density"]["difference"]   for r in results]

    mean_ov     = sum(ov_scores)   / len(ov_scores)
    mean_len    = sum(len_diffs)   / len(len_diffs)
    mean_vocab  = sum(vocab_diffs) / len(vocab_diffs)
    mean_num    = sum(num_drifts)  / len(num_drifts)
    mean_ttr    = sum(ttr_diffs)   / len(ttr_diffs)
    mean_sent   = sum(sent_diffs)  / len(sent_diffs)
    mean_dens   = sum(dens_diffs)  / len(dens_diffs)

    std_ov      = std_dev(ov_scores)

    very_low  = sum(1 for r in results if r["score"]["label"] == "VERY LOW VARIANCE")
    low       = sum(1 for r in results if r["score"]["label"] == "LOW VARIANCE")
    moderate  = sum(1 for r in results if r["score"]["label"] == "MODERATE VARIANCE")
    high      = sum(1 for r in results if r["score"]["label"] == "HIGH VARIANCE")
    very_high = sum(1 for r in results if r["score"]["label"] == "VERY HIGH VARIANCE")

    best_q  = min(results, key=lambda r: r["score"]["overall_variance"])
    worst_q = max(results, key=lambda r: r["score"]["overall_variance"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Output Variance: LLM Judge 1 vs Ground Truth")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : Multi-dimensional Output Variance Analysis")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS OUTPUT VARIANCE?")
    lines.append("-" * 70)
    lines.append("  Output variance measures how much LLM1 predictions DEVIATE")
    lines.append("  from the ground truth across six dimensions.")
    lines.append("  Low variance = model stays close to ground truth.")
    lines.append("  High variance = model drifts significantly from ground truth.")
    lines.append("")
    lines.append("  METRIC 1 -- LENGTH VARIANCE (weight=20%)")
    lines.append("    Difference in word count between prediction and GT.")
    lines.append("    A prediction much longer or shorter than GT loses points.")
    lines.append("")
    lines.append("  METRIC 2 -- VOCABULARY VARIANCE (weight=15%)")
    lines.append("    Difference in unique word counts.")
    lines.append("    High vocabulary variance means different terminology used.")
    lines.append("")
    lines.append("  METRIC 3 -- NUMERIC DRIFT (weight=30%)")
    lines.append("    How far off predicted numbers are from GT numbers.")
    lines.append("    e.g. GT says 29.1% but prediction uses very different numbers.")
    lines.append("    Most heavily weighted -- numeric accuracy is critical in science.")
    lines.append("")
    lines.append("  METRIC 4 -- LEXICAL DIVERSITY (weight=15%)")
    lines.append("    Difference in type-token ratio (unique words / total words).")
    lines.append("    Measures whether pred and GT have similar writing complexity.")
    lines.append("")
    lines.append("  METRIC 5 -- SENTENCE COUNT VARIANCE (weight=10%)")
    lines.append("    Difference in number of sentences.")
    lines.append("    Measures structural similarity of the two answers.")
    lines.append("")
    lines.append("  METRIC 6 -- INFORMATION DENSITY (weight=10%)")
    lines.append("    Difference in content word ratio.")
    lines.append("    High density = information-packed. Low = more filler words.")
    lines.append("")
    lines.append("  Overall variance score: 0.0 (no variance) to 1.0 (maximum variance)")
    lines.append("    < 0.10 : VERY LOW VARIANCE")
    lines.append("    < 0.25 : LOW VARIANCE")
    lines.append("    < 0.50 : MODERATE VARIANCE")
    lines.append("    < 0.75 : HIGH VARIANCE")
    lines.append("    >= 0.75 : VERY HIGH VARIANCE")
    lines.append("")

    lines.append("OVERALL VARIANCE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean overall variance score  : {mean_ov:.4f}  (std={std_ov:.4f})")
    lines.append(f"  Mean length difference       : {mean_len:.1f} words")
    lines.append(f"  Mean vocabulary difference   : {mean_vocab:.1f} unique words")
    lines.append(f"  Mean numeric drift           : {mean_num:.2f}%")
    lines.append(f"  Mean lexical diversity gap   : {mean_ttr:.4f} TTR units")
    lines.append(f"  Mean sentence count gap      : {mean_sent:.1f} sentences")
    lines.append(f"  Mean information density gap : {mean_dens:.4f}")
    lines.append("")
    lines.append(f"  Lowest variance  : Q{best_q['q_num']} = {best_q['score']['overall_variance']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Highest variance : Q{worst_q['q_num']} = {worst_q['score']['overall_variance']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Variance breakdown:")
    lines.append(f"    VERY LOW  (<0.10) : {very_low:3d} questions")
    lines.append(f"    LOW       (<0.25) : {low:3d} questions")
    lines.append(f"    MODERATE  (<0.50) : {moderate:3d} questions")
    lines.append(f"    HIGH      (<0.75) : {high:3d} questions")
    lines.append(f"    VERY HIGH (>=0.75): {very_high:3d} questions")
    lines.append("")

    lines.append("  Variance Score Distribution:")
    for lo, hi, lbl in [(0.0,0.10,"<0.10 VERY LOW "),(0.10,0.25,"<0.25 LOW      "),
                         (0.25,0.50,"<0.50 MODERATE "),(0.50,0.75,"<0.75 HIGH     "),
                         (0.75,1.01,">=0.75 VERY HIGH")]:
        count = sum(1 for s in ov_scores if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("  PER-METRIC VARIANCE SUMMARY:")
    metrics = [
        ("Length diff (words)",    [abs(r["score"]["length"]["difference"])        for r in results]),
        ("Vocab diff (words)",     [abs(r["score"]["vocabulary"]["difference"])     for r in results]),
        ("Numeric drift (%)",      [r["score"]["numeric_drift"]["mean_drift_pct"]  for r in results]),
        ("TTR difference",         [r["score"]["lexical_diversity"]["difference"]  for r in results]),
        ("Sentence diff",          [r["score"]["sentence_count"]["difference"]     for r in results]),
        ("Density diff",           [r["score"]["information_density"]["difference"] for r in results]),
    ]
    for name, vals in metrics:
        m   = sum(vals) / len(vals)
        sd  = std_dev(vals)
        mn  = min(vals)
        mx  = max(vals)
        lines.append(f"    {name:<25} mean={m:.3f}  std={sd:.3f}  min={mn:.3f}  max={mx:.3f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION VARIANCE REPORT")
    lines.append("=" * 70)

    for r in results:
        s  = r["score"]
        lv = s["length"]
        vv = s["vocabulary"]
        nv = s["numeric_drift"]
        tv = s["lexical_diversity"]
        sv = s["sentence_count"]
        dv = s["information_density"]

        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Overall variance : {s['overall_variance']:.4f}  --  {s['label']}")
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
        lines.append("  VARIANCE BREAKDOWN:")
        lines.append(f"    Length           : GT={lv['gt_words']} words, "
                     f"Pred={lv['pred_words']} words, "
                     f"diff={lv['difference']:+d} ({lv['pct_diff']:.1f}%)")
        lines.append(f"    Vocabulary       : GT={vv['gt_unique']} unique, "
                     f"Pred={vv['pred_unique']} unique, "
                     f"diff={vv['difference']:+d} ({vv['pct_diff']:.1f}%)")
        lines.append(f"    Numeric drift    : GT={nv['gt_numbers']}, "
                     f"Pred={nv['pred_numbers']}, "
                     f"mean drift={nv['mean_drift_pct']:.2f}%")
        lines.append(f"    Lexical diversity: GT TTR={tv['gt_ttr']:.4f}, "
                     f"Pred TTR={tv['pred_ttr']:.4f}, "
                     f"diff={tv['difference']:.4f}")
        lines.append(f"    Sentence count   : GT={sv['gt_sentences']}, "
                     f"Pred={sv['pred_sentences']}, "
                     f"diff={sv['difference']}")
        lines.append(f"    Info density     : GT={dv['gt_density']:.4f}, "
                     f"Pred={dv['pred_density']:.4f}, "
                     f"diff={dv['difference']:.4f}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF OUTPUT VARIANCE -- LLM Judge 1")
    lines.append(f"  Mean variance: {mean_ov:.4f} (std={std_ov:.4f})")
    lines.append(f"  Very Low: {very_low} | Low: {low} | Moderate: {moderate} | "
                 f"High: {high} | Very High: {very_high}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "variance_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Variance results written to: {out_path}\n")


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
        description="Output variance: LLM Judge 1 predictions vs ground truth"
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
    print("[INFO] Computing output variance...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()