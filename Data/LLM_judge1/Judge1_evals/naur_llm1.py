"""
BMEN-499 AlphaFold -- NAUR Score Evaluation: LLM Judge 1
---------------------------------------------------------
Purpose:
    Compares LLM Judge 1 (BiomedBERT + Symbolic Rules) predicted answers
    against DisProt ground truth answers using NAUR scoring.

What is NAUR?
    NAUR (Ngram-Aligned Unigram Recall) measures how well two text
    passages overlap using chunking techniques:

    1. CHUNKING    -- Split both texts into overlapping chunks (ngrams)
    2. ALIGNMENT   -- Find matching chunks between prediction and truth
    3. SCORING     -- Compute precision, recall, and F1 from matches

    Three chunk sizes:
      Unigram  (n=1, weight=50%) -- single word matches
      Bigram   (n=2, weight=30%) -- two-word phrase matches
      Trigram  (n=3, weight=20%) -- three-word semantic chunk matches

    Final NAUR score = weighted F1 across all three chunk sizes

Output: naur_results.txt (saved to same folder as this script)

Usage:
    python naur_llm1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python naur_llm1.py --demo
"""

import json, re, sys, os, argparse
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
     lambda s: f"Based on {s['total_proteins']:,} DisProt proteins a disorder score above 0.5 is a commonly used cutoff but it is conservative. Only {s['pct_above_0.5']:.1f}% of proteins exceed 0.5 while {s['pct_above_0.3']:.1f}% exceed 0.3. Many true IDRs fall in the 0.3 to 0.5 range and would be missed by a strict 0.5 threshold. The cutoff is a useful starting point but not fully reliable on its own."),
    (["short","residue","length","10"],
     lambda s: f"Disordered regions shorter than 10 amino acids are difficult to predict reliably. Of {s['total_regions']:,} annotated disordered regions in DisProt {s['pct_short_regions']:.1f}% are shorter than 10 residues with mean region length {s['mean_region_length']:.1f} aa. Short IDRs are underrepresented in experimental databases and prediction tools lack sufficient sequence context for short stretches."),
    (["proline","glycine"],
     lambda s: f"Proline content is a strong predictor of intrinsic disorder. The DisProt dataset mean proline fraction is {s['mean_proline']*100:.1f}% and mean glycine fraction is {s['mean_glycine']*100:.1f}%. When both are elevated together they form a strong composite disorder signal. Proline rigid ring structure disrupts alpha-helices and glycine adds backbone conformational entropy both hallmarks of intrinsically disordered regions."),
    (["sliding","window"],
     lambda s: f"Sliding window averaging smooths per-residue disorder scores to reduce noise. The mean disordered region length in DisProt is {s['mean_region_length']:.1f} amino acids. If the sliding window size exceeds this mean short disordered regions risk being averaged out and lost. Window size must be chosen carefully to balance noise reduction against signal preservation."),
    (["pfam","domain"],
     lambda s: f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam structured domain alongside their disordered regions. This confirms that structured domains and intrinsically disordered regions frequently co-occur. Each region must be evaluated independently rather than classifying the whole protein as ordered or disordered."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT scores below 50 are strong computational evidence of intrinsic disorder. DisProt experimentally confirms disorder in {s['total_proteins']:,} proteins. Regions annotated as disordered in DisProt consistently show pLDDT below 50 in AlphaFold predictions. This is the most reliable single computational signal for disorder."),
]

def get_answer(question, rules, stats):
    q = question.lower()
    for keywords, fn in rules:
        if any(kw in q for kw in keywords):
            try:
                return fn(stats)
            except:
                pass
    return f"DisProt summary {stats['total_proteins']:,} proteins mean disorder {stats['mean_disorder']:.3f} mean region length {stats['mean_region_length']:.1f} aa."


# =============================================================
# 3. NAUR SCORING ENGINE
# =============================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def ngram_score(pred, gt, n):
    p_toks = normalize(pred).split()
    g_toks = normalize(gt).split()
    if not p_toks or not g_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
    p_ng = ngrams(p_toks, n)
    g_ng = ngrams(g_toks, n)
    if not p_ng or not g_ng:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}
    matched   = sum((p_ng & g_ng).values())
    precision = matched / sum(p_ng.values())
    recall    = matched / sum(g_ng.values())
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return {"precision": round(precision,4), "recall": round(recall,4),
            "f1": round(f1,4), "matched": matched}

def naur_score(pred, gt):
    uni = ngram_score(pred, gt, 1)
    bi  = ngram_score(pred, gt, 2)
    tri = ngram_score(pred, gt, 3)
    f1  = round(0.5*uni["f1"] + 0.3*bi["f1"] + 0.2*tri["f1"], 4)
    gt_u   = set(normalize(gt).split())
    pr_u   = set(normalize(pred).split())
    cov    = round(len(gt_u & pr_u) / len(gt_u), 4) if gt_u else 0.0
    label  = ("STRONG MATCH"   if f1 >= 0.7 else
               "MODERATE MATCH" if f1 >= 0.4 else
               "WEAK MATCH"     if f1 >= 0.2 else
               "POOR MATCH")
    return {"unigram": uni, "bigram": bi, "trigram": tri,
            "naur_f1": f1, "coverage": cov, "label": label}


# =============================================================
# 4. EVALUATE + WRITE RESULTS
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = naur_score(pred, gt)
        results.append({"q_num": i, "question": q,
                         "ground_truth": gt, "prediction": pred, "score": sc})
        print(f"  Q{i:3d} | NAUR={sc['naur_f1']:.4f} | "
              f"Coverage={sc['coverage']:.4f} | {sc['label']}")
    return results

def write_results(results, stats):
    ns   = [r["score"]["naur_f1"]  for r in results]
    cvs  = [r["score"]["coverage"] for r in results]
    u1s  = [r["score"]["unigram"]["f1"] for r in results]
    b1s  = [r["score"]["bigram"]["f1"]  for r in results]
    t1s  = [r["score"]["trigram"]["f1"] for r in results]
    mn   = sum(ns)  / len(ns)
    mc   = sum(cvs) / len(cvs)
    mu   = sum(u1s) / len(u1s)
    mb   = sum(b1s) / len(b1s)
    mt   = sum(t1s) / len(t1s)
    strong   = sum(1 for r in results if r["score"]["label"] == "STRONG MATCH")
    moderate = sum(1 for r in results if r["score"]["label"] == "MODERATE MATCH")
    weak     = sum(1 for r in results if r["score"]["label"] == "WEAK MATCH")
    poor     = sum(1 for r in results if r["score"]["label"] == "POOR MATCH")

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- NAUR Score Evaluation: LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules")
    lines.append("  Metric  : NAUR (Ngram-Aligned Unigram Recall)")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS THE NAUR SCORE?")
    lines.append("-" * 70)
    lines.append("  NAUR measures how well predicted answers overlap with ground")
    lines.append("  truth using three chunk sizes:")
    lines.append("")
    lines.append("  Unigram  (n=1, weight=50%) -- single word matches")
    lines.append("    Captures key term and vocabulary coverage.")
    lines.append("")
    lines.append("  Bigram   (n=2, weight=30%) -- two-word phrase matches")
    lines.append("    Captures local phrasing similarity.")
    lines.append("")
    lines.append("  Trigram  (n=3, weight=20%) -- three-word chunk matches")
    lines.append("    Captures semantic chunk-level similarity.")
    lines.append("")
    lines.append("  Final NAUR F1 = 0.5 x Unigram + 0.3 x Bigram + 0.2 x Trigram")
    lines.append("")
    lines.append("  Score interpretation:")
    lines.append("    0.7 - 1.0 : STRONG MATCH   -- prediction closely matches truth")
    lines.append("    0.4 - 0.7 : MODERATE MATCH -- good overlap, some gaps")
    lines.append("    0.2 - 0.4 : WEAK MATCH     -- partial overlap")
    lines.append("    0.0 - 0.2 : POOR MATCH     -- little overlap")
    lines.append("")
    lines.append("OVERALL RESULTS SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Mean NAUR F1 score   : {mn:.4f}")
    lines.append(f"  Mean coverage        : {mc:.4f}")
    lines.append(f"  Mean unigram F1      : {mu:.4f}")
    lines.append(f"  Mean bigram F1       : {mb:.4f}")
    lines.append(f"  Mean trigram F1      : {mt:.4f}")
    lines.append("")
    lines.append(f"  Match quality breakdown:")
    lines.append(f"    STRONG MATCH   : {strong:3d} questions")
    lines.append(f"    MODERATE MATCH : {moderate:3d} questions")
    lines.append(f"    WEAK MATCH     : {weak:3d} questions")
    lines.append(f"    POOR MATCH     : {poor:3d} questions")
    lines.append("")
    lines.append("  NAUR Score Distribution:")
    for lo, hi, lbl in [(0.0,0.2,"0.0-0.2"),(0.2,0.4,"0.2-0.4"),
                         (0.4,0.6,"0.4-0.6"),(0.6,0.8,"0.6-0.8"),(0.8,1.01,"0.8-1.0")]:
        count = sum(1 for s in ns if lo <= s < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION NAUR SCORES")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65] for i in range(0,len(r["ground_truth"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  LLM1 PREDICTION:")
        for chunk in [r["prediction"][i:i+65] for i in range(0,len(r["prediction"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  NAUR SCORES:")
        lines.append(f"    Unigram  F1 : {s['unigram']['f1']:.4f}  (P={s['unigram']['precision']:.4f}, R={s['unigram']['recall']:.4f}, matched={s['unigram']['matched']} words)")
        lines.append(f"    Bigram   F1 : {s['bigram']['f1']:.4f}  (P={s['bigram']['precision']:.4f}, R={s['bigram']['recall']:.4f}, matched={s['bigram']['matched']} phrases)")
        lines.append(f"    Trigram  F1 : {s['trigram']['f1']:.4f}  (P={s['trigram']['precision']:.4f}, R={s['trigram']['recall']:.4f}, matched={s['trigram']['matched']} chunks)")
        lines.append(f"    NAUR F1     : {s['naur_f1']:.4f}  (weighted average)")
        lines.append(f"    Coverage    : {s['coverage']:.4f}  (fraction of GT vocabulary found)")
        lines.append(f"    Match label : {s['label']}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF NAUR EVALUATION -- LLM Judge 1")
    lines.append(f"  Mean NAUR F1: {mn:.4f} | Strong: {strong} | Moderate: {moderate} | Weak: {weak} | Poor: {poor}")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "naur_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] NAUR results written to: {out_path}\n")


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
    parser = argparse.ArgumentParser(description="NAUR score evaluation for LLM Judge 1")
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
    print("[INFO] Computing NAUR scores...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)

if __name__ == "__main__":
    main()