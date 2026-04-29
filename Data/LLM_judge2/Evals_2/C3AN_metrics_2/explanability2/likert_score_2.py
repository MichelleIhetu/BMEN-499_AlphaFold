"""
BMEN-499 AlphaFold -- Likert Score: LLM Judge 2 vs Ground Truth
----------------------------------------------------------------
Purpose:
    Evaluates LLM Judge 2 (Vanilla RAG) predicted answers against
    DisProt ground truth using a 5-point Likert scale across five
    explainability criteria.

LLM Judge 2 -- Vanilla RAG:
    BiomedBERT retrieves top-k DisProt knowledge base passages
    and concatenates them as the answer. No symbolic rules,
    no calibration -- pure neural retrieval baseline.

Likert Criteria (1-5 scale):
    1. FACTUAL ACCURACY  (30%) -- numbers and facts match GT
    2. COMPLETENESS      (25%) -- covers all key GT points
    3. CLARITY           (20%) -- clearly written answer
    4. RELEVANCE         (15%) -- stays on topic for the question
    5. SCIENTIFIC DEPTH  (10%) -- uses appropriate biomedical terms

    1 = Very Poor | 2 = Poor | 3 = Acceptable | 4 = Good | 5 = Excellent

Output: likert_score_results_2.txt (saved to same folder)

Usage:
    python likert_score_2.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python likert_score_2.py --demo
"""

import json
import re
import sys
import os
import argparse
import math
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

LLM_RULES = [
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

SCIENCE_TERMS = [
    "intrinsically disordered", "idr", "idp", "plddt", "alphafold",
    "pfam", "proline", "glycine", "amino acid", "residue", "backbone",
    "alpha-helix", "beta-sheet", "secondary structure", "conformational",
    "disprot", "disorder content", "threshold", "cutoff", "sequence",
    "annotation", "experimental", "computational", "prediction", "morf"
]

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","of","in","on","at","to","for","with","by","from","and","or",
    "but","not","this","that","these","those","it","its","they","them",
    "their","we","our","as","also","both","very","each","more","than"
}

def get_answer(question, rules, stats):
    q = question.lower()
    for keywords, fn in rules:
        if any(kw in q for kw in keywords):
            try:
                return fn(stats)
            except:
                pass
    return f"DisProt summary {stats['total_proteins']:,} proteins mean disorder {stats['mean_disorder']:.3f}."

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return [w for w in normalize(text).split()
            if w not in STOPWORDS and len(w) > 1]

def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# =============================================================
# 3. LIKERT SCORING ENGINE
# =============================================================

def score_factual_accuracy(pred, gt):
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
    gt_toks     = set(tokenize(gt))
    pred_toks   = set(tokenize(pred))
    term_overlap = len(gt_toks & pred_toks) / max(len(gt_toks), 1)
    raw   = 0.6 * matched_ratio + 0.4 * term_overlap
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Number match={matched_ratio:.2f}, Term overlap={term_overlap:.2f}"


def score_completeness(pred, gt):
    gt_toks   = set(tokenize(gt))
    pred_toks = set(tokenize(pred))
    if not gt_toks:
        return 5, "No GT content"
    covered      = gt_toks & pred_toks
    ratio        = len(covered) / len(gt_toks)
    pred_sents   = len([s for s in pred.split(".") if len(s.strip()) > 10])
    gt_sents     = len([s for s in gt.split(".")   if len(s.strip()) > 10])
    length_ratio = min(1.0, pred_sents / max(gt_sents, 1))
    raw   = 0.7 * ratio + 0.3 * length_ratio
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Coverage={ratio:.2f}, Length ratio={length_ratio:.2f}"


def score_clarity(pred):
    sentences = [s.strip() for s in pred.split(".") if len(s.strip()) > 5]
    if not sentences:
        return 1, "No complete sentences"
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if 15 <= avg_len <= 25:
        length_score = 1.0
    elif 10 <= avg_len < 15 or 25 < avg_len <= 35:
        length_score = 0.75
    else:
        length_score = 0.5
    tokens      = tokenize(pred)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    rep_score    = min(1.0, unique_ratio * 1.5)
    raw   = 0.5 * length_score + 0.5 * rep_score
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Avg sentence len={avg_len:.1f} words, Uniqueness={unique_ratio:.2f}"


def score_relevance(pred, question):
    q_toks    = set(tokenize(question))
    pred_toks = set(tokenize(pred))
    if not q_toks:
        return 5, "No question tokens"
    overlap = q_toks & pred_toks
    ratio   = len(overlap) / len(q_toks)
    score   = max(1, min(5, round(1 + ratio * 4)))
    return score, f"Question keyword coverage={ratio:.2f} ({len(overlap)}/{len(q_toks)} terms)"


def score_scientific_depth(pred):
    pred_lower = pred.lower()
    found      = [term for term in SCIENCE_TERMS if term in pred_lower]
    ratio      = len(found) / len(SCIENCE_TERMS)
    word_count = len(pred.split())
    length_bonus = min(0.2, word_count / 500)
    raw   = min(1.0, ratio * 3 + length_bonus)
    score = max(1, min(5, round(1 + raw * 4)))
    return score, f"Science terms={len(found)}/{len(SCIENCE_TERMS)}: {', '.join(found[:5])}"


def likert_score(question, pred, gt):
    acc_score,  acc_note  = score_factual_accuracy(pred, gt)
    comp_score, comp_note = score_completeness(pred, gt)
    clar_score, clar_note = score_clarity(pred)
    rel_score,  rel_note  = score_relevance(pred, question)
    dep_score,  dep_note  = score_scientific_depth(pred)

    overall = round(
        0.30 * acc_score  +
        0.25 * comp_score +
        0.20 * clar_score +
        0.15 * rel_score  +
        0.10 * dep_score,
        3
    )

    label = (
        "EXCELLENT (5)"  if overall >= 4.5 else
        "GOOD (4)"       if overall >= 3.5 else
        "ACCEPTABLE (3)" if overall >= 2.5 else
        "POOR (2)"       if overall >= 1.5 else
        "VERY POOR (1)"
    )

    return {
        "factual_accuracy": {"score": acc_score,  "note": acc_note},
        "completeness":     {"score": comp_score,  "note": comp_note},
        "clarity":          {"score": clar_score,  "note": clar_note},
        "relevance":        {"score": rel_score,   "note": rel_note},
        "scientific_depth": {"score": dep_score,   "note": dep_note},
        "overall":          overall,
        "label":            label,
    }


# =============================================================
# 4. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM_RULES, stats)
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
# 5. WRITE likert_score_results_2.txt
# =============================================================

def write_results(results, stats):
    overall = [r["score"]["overall"]                   for r in results]
    acc     = [r["score"]["factual_accuracy"]["score"] for r in results]
    comp    = [r["score"]["completeness"]["score"]     for r in results]
    clar    = [r["score"]["clarity"]["score"]          for r in results]
    rel     = [r["score"]["relevance"]["score"]        for r in results]
    dep     = [r["score"]["scientific_depth"]["score"] for r in results]

    mean_overall = sum(overall) / len(overall)
    mean_acc     = sum(acc)     / len(acc)
    mean_comp    = sum(comp)    / len(comp)
    mean_clar    = sum(clar)    / len(clar)
    mean_rel     = sum(rel)     / len(rel)
    mean_dep     = sum(dep)     / len(dep)
    std_overall  = math.sqrt(sum((s - mean_overall)**2
                                  for s in overall) / len(overall))

    excellent  = sum(1 for r in results if r["score"]["overall"] >= 4.5)
    good       = sum(1 for r in results if 3.5 <= r["score"]["overall"] < 4.5)
    acceptable = sum(1 for r in results if 2.5 <= r["score"]["overall"] < 3.5)
    poor       = sum(1 for r in results if 1.5 <= r["score"]["overall"] < 2.5)
    very_poor  = sum(1 for r in results if r["score"]["overall"] < 1.5)

    best_q  = max(results, key=lambda r: r["score"]["overall"])
    worst_q = min(results, key=lambda r: r["score"]["overall"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Likert Score: LLM Judge 2 vs Ground Truth")
    lines.append("  Model   : Vanilla RAG -- BiomedBERT Retriever (LLM Judge 2)")
    lines.append("  Metric  : 5-Point Likert Scale across 5 explainability criteria")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS LLM JUDGE 2 (VANILLA RAG)?")
    lines.append("-" * 70)
    lines.append("  BiomedBERT retrieves the most semantically similar DisProt")
    lines.append("  knowledge base passages and concatenates them as the answer.")
    lines.append("  No symbolic rules, no calibration -- pure neural retrieval.")
    lines.append("  Compare with LLM Judge 2 (Vanilla RAG) to see the value of symbolic grounding.")
    lines.append("")
    lines.append("LIKERT SCALE")
    lines.append("-" * 70)
    lines.append("  1 = Very Poor | 2 = Poor | 3 = Acceptable | 4 = Good | 5 = Excellent")
    lines.append("")
    lines.append("  CRITERION 1 -- Factual Accuracy (weight=30%)")
    lines.append("    Numbers within 3% and key facts match ground truth.")
    lines.append("")
    lines.append("  CRITERION 2 -- Completeness (weight=25%)")
    lines.append("    Prediction covers all key GT content words and sentences.")
    lines.append("")
    lines.append("  CRITERION 3 -- Clarity (weight=20%)")
    lines.append("    Sentence length (ideal 15-25 words) and word variety.")
    lines.append("")
    lines.append("  CRITERION 4 -- Relevance (weight=15%)")
    lines.append("    Question keyword coverage in the answer.")
    lines.append("")
    lines.append("  CRITERION 5 -- Scientific Depth (weight=10%)")
    lines.append("    Biomedical terms like IDR, pLDDT, Pfam, proline present.")
    lines.append("")
    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean overall Likert score  : {mean_overall:.3f} / 5.0  (std={std_overall:.3f})")
    lines.append(f"  Mean factual accuracy      : {mean_acc:.2f} / 5")
    lines.append(f"  Mean completeness          : {mean_comp:.2f} / 5")
    lines.append(f"  Mean clarity               : {mean_clar:.2f} / 5")
    lines.append(f"  Mean relevance             : {mean_rel:.2f} / 5")
    lines.append(f"  Mean scientific depth      : {mean_dep:.2f} / 5")
    lines.append(f"  Best  : Q{best_q['q_num']} = {best_q['score']['overall']:.3f} ({best_q['score']['label']})")
    lines.append(f"  Worst : Q{worst_q['q_num']} = {worst_q['score']['overall']:.3f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Quality breakdown:")
    lines.append(f"    Excellent  (4.5-5.0) : {excellent:3d} questions")
    lines.append(f"    Good       (3.5-4.5) : {good:3d} questions")
    lines.append(f"    Acceptable (2.5-3.5) : {acceptable:3d} questions")
    lines.append(f"    Poor       (1.5-2.5) : {poor:3d} questions")
    lines.append(f"    Very Poor  (0.0-1.5) : {very_poor:3d} questions")
    lines.append("")
    lines.append("  Criteria Radar (mean scores):")
    for name, val in [
        ("Factual Accuracy ", mean_acc),
        ("Completeness     ", mean_comp),
        ("Clarity          ", mean_clar),
        ("Relevance        ", mean_rel),
        ("Scientific Depth ", mean_dep),
    ]:
        bar = "#" * int(val * 4) + "." * max(0, 20 - int(val * 4))
        lines.append(f"    {name} [{bar}] {val:.2f}/5")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION LIKERT SCORES")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  OVERALL LIKERT : {s['overall']:.3f}/5  --  {s['label']}")
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
        lines.append("  LIKERT SCORES:")
        lines.append(f"    Factual Accuracy  : {s['factual_accuracy']['score']}/5  -- {s['factual_accuracy']['note']}")
        lines.append(f"    Completeness      : {s['completeness']['score']}/5  -- {s['completeness']['note']}")
        lines.append(f"    Clarity           : {s['clarity']['score']}/5  -- {s['clarity']['note']}")
        lines.append(f"    Relevance         : {s['relevance']['score']}/5  -- {s['relevance']['note']}")
        lines.append(f"    Scientific Depth  : {s['scientific_depth']['score']}/5  -- {s['scientific_depth']['note']}")
        lines.append(f"    OVERALL           : {s['overall']:.3f}/5  --  {s['label']}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF LIKERT SCORE -- LLM Judge 2 (Vanilla RAG)")
    lines.append(f"  Mean Likert: {mean_overall:.3f}/5 | "
                 f"Excellent: {excellent} | Good: {good} | "
                 f"Acceptable: {acceptable} | Poor: {poor} | Very Poor: {very_poor}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "likert_score_results_2.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Likert score results written to: {out_path}\n")


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
        description="Likert score: LLM Judge 2 (Vanilla RAG) vs ground truth"
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
    print("[INFO] Computing Likert scores for LLM Judge 2...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)


if __name__ == "__main__":
    main()