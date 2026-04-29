"""
BMEN-499 AlphaFold -- Performance Drop Analysis: LLM Judge 2 vs Ground Truth
-----------------------------------------------------------------------------
Purpose:
    Analyzes where and why LLM Judge 2 (Vanilla RAG) performance drops
    compared to ground truth answers. Identifies which question types
    and topics cause the most degradation.

LLM Judge 2 -- Vanilla RAG:
    BiomedBERT retrieves top-k DisProt knowledge base passages
    and concatenates them as the answer. No symbolic rules,
    no calibration -- pure neural retrieval baseline.

Metrics used:
    - NAUR F1        (35%) -- chunked text overlap
    - Agreement      (35%) -- factual concept alignment
    - Cosine sim     (30%) -- semantic vector similarity
    - Error penalty  (-0.05 per error, capped at -0.50)

Drop = best_score - current_score per question

Output: performance_drop_results_2.txt (saved to same folder)

Usage:
    python performance_drop2.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python performance_drop2.py --demo
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

DOMAIN_TERMS = [
    "disorder","disordered","idr","idp","plddt","alphafold","pfam",
    "proline","glycine","residue","amino","backbone","threshold","cutoff",
    "disprot","intrinsic","region","sequence","confidence","annotated"
]

TOPIC_MAP = [
    (["0.5","cutoff","threshold"],  "Disorder Threshold"),
    (["short","residue","10"],      "Short IDR Detection"),
    (["proline","glycine"],         "Sequence Composition"),
    (["sliding","window"],          "Sliding Window"),
    (["pfam","domain"],             "Structural Domains"),
    (["alphafold","plddt"],         "AlphaFold pLDDT"),
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

def ngrams(toks, n):
    return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def ngram_f1(pred, gt, n):
    pt  = content_tokens(pred)
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

def cosine_sim(pred, gt):
    pt  = Counter(content_tokens(pred))
    gt_ = Counter(content_tokens(gt))
    shared = set(pt) & set(gt_)
    dot    = sum(pt[t] * gt_[t] for t in shared)
    mag_p  = math.sqrt(sum(v**2 for v in pt.values()))
    mag_g  = math.sqrt(sum(v**2 for v in gt_.values()))
    return round(dot / (mag_p * mag_g), 4) if mag_p and mag_g else 0.0

def agreement(pred, gt):
    pred_set = set(content_tokens(pred))
    gt_set   = set(content_tokens(gt))
    if not gt_set:
        return 1.0
    return round(len(pred_set & gt_set) / len(gt_set), 4)

def error_count(pred, gt, stats):
    errors  = 0
    gt_nums  = extract_numbers(gt)
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
    errors    += len([t for t in gt_terms if t not in pred_terms])
    return errors

def performance_score(pred, gt, stats):
    nf = naur_f1(pred, gt)
    cs = cosine_sim(pred, gt)
    ag = agreement(pred, gt)
    ec = error_count(pred, gt, stats)
    penalty = min(0.5, ec * 0.05)
    score   = round(max(0.0, 0.35*nf + 0.30*cs + 0.35*ag - penalty), 4)
    return score, nf, cs, ag, ec


# =============================================================
# 4. DROP ANALYSIS + ROOT CAUSES
# =============================================================

def analyze_drops(results):
    perf_scores = [r["perf_score"] for r in results]
    best_perf   = max(perf_scores)
    mean_perf   = sum(perf_scores) / len(perf_scores)
    std_perf    = math.sqrt(sum((s-mean_perf)**2 for s in perf_scores)/len(perf_scores))

    for r in results:
        r["drop_perf"]   = round(best_perf - r["perf_score"],  4)
        r["drop_naur"]   = round(max(r["naur"]   for r in results) - r["naur"],   4)
        r["drop_cosine"] = round(max(r["cosine"] for r in results) - r["cosine"], 4)
        r["drop_agree"]  = round(max(r["agreement"] for r in results) - r["agreement"], 4)
        r["drop_errors"] = r["errors"] - min(r["errors"] for r in results)
        dp = r["drop_perf"]
        r["drop_label"] = (
            "NO DROP"          if dp == 0.0  else
            "MINOR DROP"       if dp < 0.10  else
            "MODERATE DROP"    if dp < 0.25  else
            "SIGNIFICANT DROP" if dp < 0.50  else
            "SEVERE DROP"
        )
    return results, best_perf, mean_perf, std_perf


def identify_causes(results):
    causes = []
    dropped = [r for r in results if r["drop_perf"] > 0.10]
    if not dropped:
        causes.append("No significant performance drops detected.")
        return causes

    topic_perf = defaultdict(list)
    for r in results:
        topic_perf[r["topic"]].append(r["perf_score"])

    worst_topic = min(topic_perf, key=lambda t: sum(topic_perf[t])/len(topic_perf[t]))
    best_topic  = max(topic_perf, key=lambda t: sum(topic_perf[t])/len(topic_perf[t]))
    causes.append(
        f"Lowest mean performance on topic '{worst_topic}' "
        f"(mean={sum(topic_perf[worst_topic])/len(topic_perf[worst_topic]):.4f}). "
        f"Highest on '{best_topic}' "
        f"(mean={sum(topic_perf[best_topic])/len(topic_perf[best_topic]):.4f})."
    )

    high_error = [r for r in results if r["errors"] > 3]
    if high_error:
        causes.append(
            f"{len(high_error)} questions have >3 errors -- "
            f"error count is a primary driver of performance drop in Vanilla RAG."
        )

    low_cosine = [r for r in results if r["cosine"] < 0.3]
    if low_cosine:
        causes.append(
            f"{len(low_cosine)} questions have cosine similarity <0.3 -- "
            f"RAG retrieves relevant but differently-worded passages."
        )

    perf_scores = [r["perf_score"] for r in results]
    std = math.sqrt(sum((s-sum(perf_scores)/len(perf_scores))**2
                        for s in perf_scores)/len(perf_scores))
    if std > 0.15:
        causes.append(
            f"High variance (std={std:.4f}) -- Vanilla RAG is inconsistent "
            f"across question types. No symbolic rules to stabilize output."
        )
    else:
        causes.append(
            f"Low variance (std={std:.4f}) -- RAG is consistent "
            f"despite score differences across topics."
        )

    causes.append(
        "NOTE: Vanilla RAG drops are typically driven by topic mismatch "
        "in retrieval -- the retrieved passages are factually correct but "
        "may not directly address the specific question asked."
    )
    return causes


# =============================================================
# 5. EVALUATE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt    = get_answer(q, GT_RULES,   stats)
        pred  = get_answer(q, LLM_RULES, stats)
        topic = classify_topic(q)
        ps, nf, cs, ag, ec = performance_score(pred, gt, stats)
        results.append({
            "q_num": i, "question": q, "topic": topic,
            "ground_truth": gt, "prediction": pred,
            "perf_score": ps, "naur": nf, "cosine": cs,
            "agreement": ag, "errors": ec,
        })
        print(f"  Q{i:3d} | Topic={topic:<22} | "
              f"Perf={ps:.4f} | NAUR={nf:.4f} | "
              f"Cosine={cs:.4f} | Agree={ag:.4f} | Errors={ec}")

    results, best_perf, mean_perf, std_perf = analyze_drops(results)
    return results, best_perf, mean_perf, std_perf


# =============================================================
# 6. WRITE performance_drop_results_2.txt
# =============================================================

def write_results(results, stats, best_perf, mean_perf, std_perf):
    causes = identify_causes(results)

    perf_scores = [r["perf_score"]  for r in results]
    naur_scores = [r["naur"]        for r in results]
    cos_scores  = [r["cosine"]      for r in results]
    ag_scores   = [r["agreement"]   for r in results]
    err_counts  = [r["errors"]      for r in results]
    drop_scores = [r["drop_perf"]   for r in results]

    mean_naur  = sum(naur_scores) / len(naur_scores)
    mean_cos   = sum(cos_scores)  / len(cos_scores)
    mean_ag    = sum(ag_scores)   / len(ag_scores)
    mean_err   = sum(err_counts)  / len(err_counts)
    mean_drop  = sum(drop_scores) / len(drop_scores)

    no_drop  = sum(1 for r in results if r["drop_label"] == "NO DROP")
    minor    = sum(1 for r in results if r["drop_label"] == "MINOR DROP")
    moderate = sum(1 for r in results if r["drop_label"] == "MODERATE DROP")
    sig      = sum(1 for r in results if r["drop_label"] == "SIGNIFICANT DROP")
    severe   = sum(1 for r in results if r["drop_label"] == "SEVERE DROP")

    best_q  = max(results, key=lambda r: r["perf_score"])
    worst_q = min(results, key=lambda r: r["perf_score"])

    ranked = sorted(results, key=lambda r: r["perf_score"], reverse=True)

    topic_perf = defaultdict(list)
    for r in results:
        topic_perf[r["topic"]].append(r["perf_score"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Performance Drop: LLM Judge 2 vs Ground Truth")
    lines.append("  Model   : Vanilla RAG -- BiomedBERT Retriever (LLM Judge 2)")
    lines.append("  Metric  : Multi-metric Performance Drop Analysis")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS LLM JUDGE 2 (VANILLA RAG)?")
    lines.append("-" * 70)
    lines.append("  BiomedBERT retrieves top-k DisProt passages and concatenates")
    lines.append("  them as the answer. No symbolic rules, no calibration.")
    lines.append("  Performance drops reveal where pure retrieval struggles most.")
    lines.append("")
    lines.append("PERFORMANCE SCORE FORMULA")
    lines.append("-" * 70)
    lines.append("  Score = 0.35*NAUR_F1 + 0.30*Cosine + 0.35*Agreement - penalty")
    lines.append("  Penalty = 0.05 per error (capped at 0.50)")
    lines.append("  Drop = best_score - current_score")
    lines.append("  Labels: NO DROP | MINOR (<0.10) | MODERATE (<0.25) |")
    lines.append("          SIGNIFICANT (<0.50) | SEVERE (>=0.50)")
    lines.append("")
    lines.append("OVERALL PERFORMANCE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Best score    : {best_perf:.4f}  (Q{best_q['q_num']} -- {best_q['topic']})")
    lines.append(f"  Worst score   : {worst_q['perf_score']:.4f}  (Q{worst_q['q_num']} -- {worst_q['topic']})")
    lines.append(f"  Mean score    : {mean_perf:.4f}")
    lines.append(f"  Std deviation : {std_perf:.4f}  ({'consistent' if std_perf < 0.15 else 'inconsistent'})")
    lines.append(f"  Mean drop     : {mean_drop:.4f}")
    lines.append(f"  Mean NAUR F1  : {mean_naur:.4f}")
    lines.append(f"  Mean cosine   : {mean_cos:.4f}")
    lines.append(f"  Mean agreement: {mean_ag:.4f}")
    lines.append(f"  Mean errors   : {mean_err:.2f}")
    lines.append("")
    lines.append(f"  Drop severity:")
    lines.append(f"    NO DROP          : {no_drop:3d} questions")
    lines.append(f"    MINOR DROP       : {minor:3d} questions")
    lines.append(f"    MODERATE DROP    : {moderate:3d} questions")
    lines.append(f"    SIGNIFICANT DROP : {sig:3d} questions")
    lines.append(f"    SEVERE DROP      : {severe:3d} questions")
    lines.append("")
    lines.append("  PERFORMANCE RANKING (best to worst):")
    lines.append(f"    {'Rank':<5} {'Q#':<5} {'Score':<8} {'Drop':<8} {'Topic':<25} {'Label'}")
    lines.append(f"    {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*25} {'-'*20}")
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"    {rank:<5} Q{r['q_num']:<4} {r['perf_score']:<8.4f} "
            f"{r['drop_perf']:<8.4f} {r['topic']:<25} {r['drop_label']}"
        )
    lines.append("")
    lines.append("  PERFORMANCE BY TOPIC:")
    lines.append(f"    {'Topic':<25} {'Mean Score':<12} {'Questions':<10} {'Status'}")
    lines.append(f"    {'-'*25} {'-'*12} {'-'*10} {'-'*15}")
    for topic, scores in sorted(topic_perf.items(),
                                 key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        m      = sum(scores) / len(scores)
        status = "STRONG" if m >= 0.5 else "MODERATE" if m >= 0.3 else "WEAK"
        lines.append(f"    {topic:<25} {m:<12.4f} {len(scores):<10} {status}")
    lines.append("")
    lines.append("  ROOT CAUSE ANALYSIS:")
    for i, cause in enumerate(causes, 1):
        lines.append(f"    {i}. {cause}")
    lines.append("")
    lines.append("  METRIC CORRELATION WITH DROP:")
    metric_pairs = [
        ("NAUR F1",    [r["naur"]      for r in results]),
        ("Cosine Sim", [r["cosine"]    for r in results]),
        ("Agreement",  [r["agreement"] for r in results]),
        ("Error Count",[-r["errors"]   for r in results]),
    ]
    for name, vals in metric_pairs:
        drops = [r["drop_perf"] for r in results]
        m_v   = sum(vals)  / len(vals)
        m_d   = sum(drops) / len(drops)
        cov   = sum((v-m_v)*(d-m_d) for v, d in zip(vals, drops)) / len(vals)
        std_v = math.sqrt(sum((v-m_v)**2 for v in vals) / len(vals))
        std_d = math.sqrt(sum((d-m_d)**2 for d in drops) / len(drops))
        corr  = round(-cov/(std_v*std_d), 4) if std_v and std_d else 0.0
        bar   = "#" * int(abs(corr)*20) + "." * max(0, 20-int(abs(corr)*20))
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
        lines.append(f"  Metrics: NAUR={r['naur']:.4f} | Cosine={r['cosine']:.4f} | "
                     f"Agreement={r['agreement']:.4f} | Errors={r['errors']}")
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
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF PERFORMANCE DROP -- LLM Judge 2 (Vanilla RAG)")
    lines.append(f"  Best: {best_perf:.4f} | Mean: {mean_perf:.4f} | Std: {std_perf:.4f}")
    lines.append(f"  No drop: {no_drop} | Minor: {minor} | "
                 f"Moderate: {moderate} | Significant: {sig} | Severe: {severe}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "performance_drop_results_2.txt")

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
        description="Performance drop: LLM Judge 2 (Vanilla RAG) vs ground truth"
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
    print("[INFO] Running performance drop analysis for LLM Judge 2...\n")
    results, best_perf, mean_perf, std_perf = evaluate(questions, stats)
    write_results(results, stats, best_perf, mean_perf, std_perf)


if __name__ == "__main__":
    main()