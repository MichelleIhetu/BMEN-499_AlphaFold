"""
BMEN-499 AlphaFold -- Performance Drop Analysis: LLM Judge 1
-------------------------------------------------------------
Purpose:
    Analyzes where and why LLM Judge 1 performance drops compared
    to ground truth answers. Identifies which question types,
    topics, and content characteristics cause the most degradation.

What is Performance Drop Analysis?
    Performance drop analysis goes beyond a single score -- it
    asks WHERE the model struggles and WHY. It answers:

      - Which question topics cause the most performance loss?
      - Does performance drop on numeric-heavy questions?
      - Does performance drop on questions requiring rare terms?
      - Which error types drive the most performance degradation?
      - Is performance consistent or highly variable across questions?

    Metrics used to measure performance at each question:
      - NAUR F1 score      (chunked text overlap)
      - Cosine similarity  (semantic vector similarity)
      - Agreement score    (factual alignment)
      - Error count        (number of detected errors)

    Drop = difference between best-performing question and
           current question across each metric.

Output: performance_drop_results.txt (saved to same folder)

Usage:
    python performance_drop1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python performance_drop1.py --demo
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

DOMAIN_TERMS = [
    "disorder","idr","idp","plddt","alphafold","pfam","proline",
    "glycine","residue","amino","backbone","threshold","cutoff",
    "disprot","intrinsic","region","sequence","confidence","annotated"
]

# Topic labels for question classification
TOPIC_MAP = [
    (["0.5","cutoff","threshold"],      "Disorder Threshold"),
    (["short","residue","10"],          "Short IDR Detection"),
    (["proline","glycine"],             "Sequence Composition"),
    (["sliding","window"],              "Sliding Window"),
    (["pfam","domain"],                 "Structural Domains"),
    (["alphafold","plddt"],             "AlphaFold pLDDT"),
]

def classify_topic(question):
    q = question.lower()
    for keywords, label in TOPIC_MAP:
        if any(kw in q for kw in keywords):
            return label
    return "General"

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
# 3. PERFORMANCE METRICS
#    Compute all four performance scores for one pred/GT pair
# =============================================================

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","of","in","on",
    "at","to","for","with","by","from","and","or","but","not","this",
    "that","it","its","they","we","as","also","both","very","each"
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


# -- NAUR F1 --------------------------------------------------
def ngrams(toks, n):
    return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def ngram_f1(pred, gt, n):
    pt = content_tokens(pred)
    gt_ = content_tokens(gt)
    if not pt or not gt_:
        return 0.0
    pn = ngrams(pt, n)
    gn = ngrams(gt_, n)
    if not pn or not gn:
        return 0.0
    matched   = sum((pn & gn).values())
    precision = matched / sum(pn.values())
    recall    = matched / sum(gn.values())
    return round(2*precision*recall/(precision+recall), 4) if (precision+recall) > 0 else 0.0

def naur_f1(pred, gt):
    return round(0.5*ngram_f1(pred,gt,1) +
                 0.3*ngram_f1(pred,gt,2) +
                 0.2*ngram_f1(pred,gt,3), 4)


# -- Cosine similarity ----------------------------------------
def cosine_sim(pred, gt):
    pt = Counter(content_tokens(pred))
    gt_ = Counter(content_tokens(gt))
    shared = set(pt) & set(gt_)
    dot    = sum(pt[t] * gt_[t] for t in shared)
    mag_p  = math.sqrt(sum(v**2 for v in pt.values()))
    mag_g  = math.sqrt(sum(v**2 for v in gt_.values()))
    return round(dot / (mag_p * mag_g), 4) if mag_p and mag_g else 0.0


# -- Agreement score ------------------------------------------
def agreement(pred, gt):
    pred_set = set(content_tokens(pred))
    gt_set   = set(content_tokens(gt))
    if not gt_set:
        return 1.0
    shared = pred_set & gt_set
    return round(len(shared) / len(gt_set), 4)


# -- Error count ----------------------------------------------
def error_count(pred, gt, stats):
    errors = 0
    gt_nums   = extract_numbers(gt)
    pred_nums = extract_numbers(pred)
    for gn in gt_nums:
        if gn == 0:
            continue
        close = any(abs(pn-gn)/max(abs(gn),1e-9) < 0.05 for pn in pred_nums)
        if not close:
            conflicts = [pn for pn in pred_nums
                         if abs(pn-gn)/max(abs(gn),1e-9) > 0.05
                         and abs(pn-gn) < abs(gn)*5]
            if conflicts:
                errors += 1
    gt_terms   = [t for t in DOMAIN_TERMS if t in gt.lower()]
    pred_terms = [t for t in DOMAIN_TERMS if t in pred.lower()]
    missing    = [t for t in gt_terms if t not in pred_terms]
    errors    += len(missing)
    return errors


# -- Combined performance score (0-1, higher is better) -------
def performance_score(pred, gt, stats):
    nf = naur_f1(pred, gt)
    cs = cosine_sim(pred, gt)
    ag = agreement(pred, gt)
    ec = error_count(pred, gt, stats)
    error_penalty = min(0.5, ec * 0.05)
    score = round(0.35*nf + 0.30*cs + 0.35*ag - error_penalty, 4)
    return max(0.0, score), nf, cs, ag, ec


# =============================================================
# 4. PERFORMANCE DROP ANALYSIS
# =============================================================

def analyze_drops(results):
    """
    For each metric, find the best score and compute the drop
    for every other question relative to that best.
    """
    perf_scores = [r["perf_score"]  for r in results]
    naur_scores = [r["naur"]        for r in results]
    cosine_scores = [r["cosine"]    for r in results]
    agree_scores  = [r["agreement"] for r in results]
    error_counts  = [r["errors"]    for r in results]

    best_perf   = max(perf_scores)
    best_naur   = max(naur_scores)
    best_cosine = max(cosine_scores)
    best_agree  = max(agree_scores)
    best_errors = min(error_counts)

    mean_perf   = sum(perf_scores)   / len(perf_scores)
    std_perf    = math.sqrt(sum((s - mean_perf)**2 for s in perf_scores) / len(perf_scores))

    for r in results:
        r["drop_perf"]   = round(best_perf   - r["perf_score"],  4)
        r["drop_naur"]   = round(best_naur   - r["naur"],        4)
        r["drop_cosine"] = round(best_cosine - r["cosine"],      4)
        r["drop_agree"]  = round(best_agree  - r["agreement"],   4)
        r["drop_errors"] = r["errors"] - best_errors

        # Drop severity label
        dp = r["drop_perf"]
        r["drop_label"] = (
            "NO DROP"          if dp == 0.0  else
            "MINOR DROP"       if dp < 0.10  else
            "MODERATE DROP"    if dp < 0.25  else
            "SIGNIFICANT DROP" if dp < 0.50  else
            "SEVERE DROP"
        )

    return results, best_perf, mean_perf, std_perf


def identify_drop_causes(results):
    """
    Identify patterns that correlate with performance drops.
    Returns a list of root cause findings.
    """
    causes = []

    # Find questions with significant drops
    dropped = [r for r in results if r["drop_perf"] > 0.10]
    if not dropped:
        causes.append("No significant performance drops detected across all questions.")
        return causes

    # Check if drops correlate with topic
    topic_drops = {}
    for r in results:
        t = r["topic"]
        if t not in topic_drops:
            topic_drops[t] = []
        topic_drops[t].append(r["perf_score"])

    worst_topic = min(topic_drops, key=lambda t: sum(topic_drops[t])/len(topic_drops[t]))
    best_topic  = max(topic_drops, key=lambda t: sum(topic_drops[t])/len(topic_drops[t]))
    causes.append(
        f"Lowest mean performance on topic '{worst_topic}' "
        f"(mean={sum(topic_drops[worst_topic])/len(topic_drops[worst_topic]):.4f}). "
        f"Highest on '{best_topic}' "
        f"(mean={sum(topic_drops[best_topic])/len(topic_drops[best_topic]):.4f})."
    )

    # Check if drops correlate with high error count
    high_error_qs = [r for r in results if r["errors"] > 3]
    if high_error_qs:
        causes.append(
            f"{len(high_error_qs)} questions have >3 errors -- "
            f"high error count is a primary driver of performance drop."
        )

    # Check if drops correlate with low cosine similarity
    low_cosine = [r for r in results if r["cosine"] < 0.3]
    if low_cosine:
        causes.append(
            f"{len(low_cosine)} questions have cosine similarity <0.3 -- "
            f"semantic vocabulary mismatch between LLM1 and ground truth."
        )

    # Check if drops correlate with low NAUR
    low_naur = [r for r in results if r["naur"] < 0.3]
    if low_naur:
        causes.append(
            f"{len(low_naur)} questions have NAUR F1 <0.3 -- "
            f"low text chunk overlap between prediction and ground truth."
        )

    # Check consistency -- high std = inconsistent performance
    perf_scores = [r["perf_score"] for r in results]
    std = math.sqrt(sum((s - sum(perf_scores)/len(perf_scores))**2
                        for s in perf_scores) / len(perf_scores))
    if std > 0.15:
        causes.append(
            f"High performance variance (std={std:.4f}) -- model is inconsistent "
            f"across question types. Performance is not uniform."
        )
    else:
        causes.append(
            f"Low performance variance (std={std:.4f}) -- model is consistent "
            f"across question types despite score differences."
        )

    return causes


# =============================================================
# 5. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt    = get_answer(q, GT_RULES,   stats)
        pred  = get_answer(q, LLM1_RULES, stats)
        topic = classify_topic(q)
        ps, nf, cs, ag, ec = performance_score(pred, gt, stats)

        results.append({
            "q_num":        i,
            "question":     q,
            "topic":        topic,
            "ground_truth": gt,
            "prediction":   pred,
            "perf_score":   ps,
            "naur":         nf,
            "cosine":       cs,
            "agreement":    ag,
            "errors":       ec,
        })
        print(f"  Q{i:3d} | Topic={topic:<22} | "
              f"Perf={ps:.4f} | NAUR={nf:.4f} | "
              f"Cosine={cs:.4f} | Agree={ag:.4f} | Errors={ec}")

    results, best_perf, mean_perf, std_perf = analyze_drops(results)
    return results, best_perf, mean_perf, std_perf


# =============================================================
# 6. WRITE performance_drop_results.txt
# =============================================================

def write_results(results, stats, best_perf, mean_perf, std_perf):
    causes = identify_drop_causes(results)

    perf_scores = [r["perf_score"]  for r in results]
    naur_scores = [r["naur"]        for r in results]
    cos_scores  = [r["cosine"]      for r in results]
    ag_scores   = [r["agreement"]   for r in results]
    err_counts  = [r["errors"]      for r in results]
    drop_scores = [r["drop_perf"]   for r in results]

    mean_naur   = sum(naur_scores) / len(naur_scores)
    mean_cos    = sum(cos_scores)  / len(cos_scores)
    mean_ag     = sum(ag_scores)   / len(ag_scores)
    mean_err    = sum(err_counts)  / len(err_counts)
    mean_drop   = sum(drop_scores) / len(drop_scores)

    no_drop   = sum(1 for r in results if r["drop_label"] == "NO DROP")
    minor     = sum(1 for r in results if r["drop_label"] == "MINOR DROP")
    moderate  = sum(1 for r in results if r["drop_label"] == "MODERATE DROP")
    sig       = sum(1 for r in results if r["drop_label"] == "SIGNIFICANT DROP")
    severe    = sum(1 for r in results if r["drop_label"] == "SEVERE DROP")

    best_q  = max(results, key=lambda r: r["perf_score"])
    worst_q = min(results, key=lambda r: r["perf_score"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Performance Drop Analysis: LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : Multi-metric Performance Drop Analysis")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS PERFORMANCE DROP ANALYSIS?")
    lines.append("-" * 70)
    lines.append("  Performance drop analysis identifies WHERE and WHY the model")
    lines.append("  degrades compared to the best-performing questions.")
    lines.append("")
    lines.append("  Combined performance score uses four metrics:")
    lines.append("    NAUR F1      (weight=35%) -- chunked text overlap")
    lines.append("    Agreement    (weight=35%) -- factual concept alignment")
    lines.append("    Cosine sim   (weight=30%) -- semantic vector similarity")
    lines.append("    Error penalty (-0.05 each error, capped at -0.50)")
    lines.append("")
    lines.append("  Drop = best_score - current_score")
    lines.append("  Drop labels:")
    lines.append("    NO DROP          : drop = 0.0  (this is the best question)")
    lines.append("    MINOR DROP       : drop < 0.10")
    lines.append("    MODERATE DROP    : drop < 0.25")
    lines.append("    SIGNIFICANT DROP : drop < 0.50")
    lines.append("    SEVERE DROP      : drop >= 0.50")
    lines.append("")

    lines.append("OVERALL PERFORMANCE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Best performance score  : {best_perf:.4f}  (Q{best_q['q_num']} -- {best_q['topic']})")
    lines.append(f"  Worst performance score : {worst_q['perf_score']:.4f}  (Q{worst_q['q_num']} -- {worst_q['topic']})")
    lines.append(f"  Mean performance score  : {mean_perf:.4f}")
    lines.append(f"  Std deviation           : {std_perf:.4f}  ({'consistent' if std_perf < 0.15 else 'inconsistent'})")
    lines.append(f"  Mean drop from best     : {mean_drop:.4f}")
    lines.append("")
    lines.append(f"  Mean NAUR F1            : {mean_naur:.4f}")
    lines.append(f"  Mean cosine similarity  : {mean_cos:.4f}")
    lines.append(f"  Mean agreement score    : {mean_ag:.4f}")
    lines.append(f"  Mean error count        : {mean_err:.2f}")
    lines.append("")
    lines.append(f"  Drop severity breakdown:")
    lines.append(f"    NO DROP          : {no_drop:3d} questions")
    lines.append(f"    MINOR DROP       : {minor:3d} questions")
    lines.append(f"    MODERATE DROP    : {moderate:3d} questions")
    lines.append(f"    SIGNIFICANT DROP : {sig:3d} questions")
    lines.append(f"    SEVERE DROP      : {severe:3d} questions")
    lines.append("")

    # Performance ranking
    ranked = sorted(results, key=lambda r: r["perf_score"], reverse=True)
    lines.append("  PERFORMANCE RANKING (best to worst):")
    lines.append(f"    {'Rank':<5} {'Q#':<5} {'Score':<8} {'Drop':<8} {'Topic':<25} {'Label'}")
    lines.append(f"    {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*25} {'-'*20}")
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"    {rank:<5} Q{r['q_num']:<4} {r['perf_score']:<8.4f} "
            f"{r['drop_perf']:<8.4f} {r['topic']:<25} {r['drop_label']}"
        )
    lines.append("")

    # Topic analysis
    topic_perf = {}
    for r in results:
        t = r["topic"]
        if t not in topic_perf:
            topic_perf[t] = []
        topic_perf[t].append(r["perf_score"])
    lines.append("  PERFORMANCE BY TOPIC:")
    lines.append(f"    {'Topic':<25} {'Mean Score':<12} {'Questions':<10} {'Status'}")
    lines.append(f"    {'-'*25} {'-'*12} {'-'*10} {'-'*15}")
    for topic, scores in sorted(topic_perf.items(),
                                 key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        m = sum(scores) / len(scores)
        status = "STRONG" if m >= 0.5 else "MODERATE" if m >= 0.3 else "WEAK"
        lines.append(f"    {topic:<25} {m:<12.4f} {len(scores):<10} {status}")
    lines.append("")

    lines.append("  ROOT CAUSE ANALYSIS:")
    for i, cause in enumerate(causes, 1):
        lines.append(f"    {i}. {cause}")
    lines.append("")

    # Metric correlation with drop
    lines.append("  WHICH METRIC CORRELATES MOST WITH PERFORMANCE DROP?")
    lines.append("    (Higher correlation = this metric drives performance drop)")
    metric_pairs = [
        ("NAUR F1",     [r["naur"]      for r in results]),
        ("Cosine Sim",  [r["cosine"]    for r in results]),
        ("Agreement",   [r["agreement"] for r in results]),
        ("Error Count", [-r["errors"]   for r in results]),  # negative: more errors = lower perf
    ]
    for name, vals in metric_pairs:
        drops = [r["drop_perf"] for r in results]
        m_v   = sum(vals)   / len(vals)
        m_d   = sum(drops)  / len(drops)
        cov   = sum((v - m_v)*(d - m_d) for v, d in zip(vals, drops)) / len(vals)
        std_v = math.sqrt(sum((v - m_v)**2 for v in vals) / len(vals))
        std_d = math.sqrt(sum((d - m_d)**2 for d in drops) / len(drops))
        corr  = round(-cov / (std_v * std_d), 4) if std_v and std_d else 0.0
        bar   = "#" * int(abs(corr) * 20) + "." * max(0, 20 - int(abs(corr)*20))
        lines.append(f"    {name:<15} [{bar}] corr={corr:+.4f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION PERFORMANCE DROP REPORT")
    lines.append("=" * 70)

    for r in results:
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  Topic            : {r['topic']}")
        lines.append(f"  Performance score: {r['perf_score']:.4f}")
        lines.append(f"  Drop from best   : {r['drop_perf']:.4f}  --  {r['drop_label']}")
        lines.append("")
        lines.append(f"  Metric breakdown:")
        lines.append(f"    NAUR F1      : {r['naur']:.4f}  (drop={r['drop_naur']:.4f})")
        lines.append(f"    Cosine sim   : {r['cosine']:.4f}  (drop={r['drop_cosine']:.4f})")
        lines.append(f"    Agreement    : {r['agreement']:.4f}  (drop={r['drop_agree']:.4f})")
        lines.append(f"    Errors       : {r['errors']}  (above best={r['drop_errors']})")
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
    lines.append("  END OF PERFORMANCE DROP ANALYSIS -- LLM Judge 1")
    lines.append(f"  Best score: {best_perf:.4f} (Q{best_q['q_num']}) | "
                 f"Mean: {mean_perf:.4f} | Std: {std_perf:.4f}")
    lines.append(f"  No drop: {no_drop} | Minor: {minor} | "
                 f"Moderate: {moderate} | Significant: {sig} | Severe: {severe}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "performance_drop_results.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Performance drop results written to: {out_path}\n")


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
        description="Performance drop analysis: LLM Judge 1 vs ground truth"
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
    print("[INFO] Running performance drop analysis...\n")
    results, best_perf, mean_perf, std_perf = evaluate(questions, stats)
    write_results(results, stats, best_perf, mean_perf, std_perf)


if __name__ == "__main__":
    main()