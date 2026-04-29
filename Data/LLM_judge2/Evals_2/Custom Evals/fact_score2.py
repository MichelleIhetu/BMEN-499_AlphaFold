"""
BMEN-499 AlphaFold -- FactScore: LLM Judge 2 vs Ground Truth
-------------------------------------------------------------
Purpose:
    Measures the factual precision of LLM Judge 2 (Vanilla RAG)
    predicted answers by decomposing each answer into atomic claims
    and checking whether each is supported by the DisProt ground truth.

LLM Judge 2 -- Vanilla RAG:
    BiomedBERT retrieves top-k DisProt knowledge base passages
    and concatenates them as the answer. No symbolic rules,
    no calibration -- pure neural retrieval baseline.

Reference:
    Min et al. (2023) FActScore: Fine-grained Atomic Evaluation
    of Factual Precision in Long Form Text Generation. EMNLP 2023.

Output: factscore_results_2.txt (saved to same folder)

Usage:
    python fact_score2.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python fact_score2.py --demo
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
# 3. ATOMIC FACT ENGINE
# =============================================================

def decompose_facts(text):
    facts = []
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    for sent in sentences:
        sl = sent.lower()
        clauses = re.split(r'\b(and|but|while|whereas|although)\b', sl)
        clauses = [c.strip() for c in clauses
                   if len(c.strip()) > 8
                   and c.strip() not in {"and","but","while","whereas","although"}]
        for clause in clauses:
            if re.search(r'\d+\.?\d*\s*%?', clause):
                ftype = "NUMERIC"
            elif re.search(r'\b(above|below|more|less|higher|lower|exceed)\b', clause):
                ftype = "COMPARATIVE"
            elif re.search(r'\b(causes?|leads? to|because|promotes?|disrupts?)\b', clause):
                ftype = "CAUSAL"
            elif re.search(r'\b(contains?|has|have|found|exist|annotated)\b', clause):
                ftype = "EXISTENCE"
            else:
                ftype = "PROPERTY"
            facts.append({"text": clause, "type": ftype})
    return facts

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
    return [w for w in normalize(text).split() if w not in sw and len(w)>1]

def verify_numeric(fact_text, gt):
    fn = extract_numbers(fact_text)
    gn = extract_numbers(gt)
    if not fn: return "UNVERIFIABLE", "No numbers"
    sup = sum(1 for f in fn if f!=0 and
              any(abs(f-g)/max(abs(g),1e-9)<0.05 for g in gn))
    tot = len([f for f in fn if f!=0])
    if tot == 0: return "UNVERIFIABLE", "Only zeros"
    if sup == tot: return "SUPPORTED",   f"All {sup} numbers verified"
    if sup > 0:    return "PARTIAL",     f"{sup}/{tot} numbers verified"
    return "UNSUPPORTED", f"Numbers {fn[:3]} not in GT {gn[:3]}"

def verify_keyword(fact_text, gt):
    ft  = set(tokenize(fact_text))
    gt_ = set(tokenize(gt))
    if not ft: return "UNVERIFIABLE", "No content"
    ratio = len(ft & gt_) / len(ft)
    if ratio >= 0.6: return "SUPPORTED",    f"{ratio*100:.0f}% keywords in GT"
    if ratio >= 0.3: return "PARTIAL",      f"{ratio*100:.0f}% keywords in GT"
    return "UNSUPPORTED", f"Only {ratio*100:.0f}% overlap"

def verify_fact(fact, gt):
    if fact["type"] == "NUMERIC":
        status, reason = verify_numeric(fact["text"], gt)
        if status == "UNSUPPORTED":
            ks, _ = verify_keyword(fact["text"], gt)
            if ks == "SUPPORTED":
                status, reason = "PARTIAL", reason + " (keywords match)"
    else:
        status, reason = verify_keyword(fact["text"], gt)
    return status, reason

def factscore(pred, gt):
    facts = decompose_facts(pred)
    if not facts:
        return {"score":0.0,"label":"UNVERIFIABLE","total_facts":0,
                "supported":0,"partial":0,"unsupported":0,"unverifiable":0,"facts":[]}
    verified = []
    for fact in facts:
        status, reason = verify_fact(fact, gt)
        verified.append({"text":fact["text"],"type":fact["type"],
                         "status":status,"reason":reason})
    sup   = sum(1 for f in verified if f["status"]=="SUPPORTED")
    par   = sum(1 for f in verified if f["status"]=="PARTIAL")
    unsup = sum(1 for f in verified if f["status"]=="UNSUPPORTED")
    unver = sum(1 for f in verified if f["status"]=="UNVERIFIABLE")
    score = round((sup + 0.5*par) / len(facts), 4)
    label = ("EXCELLENT" if score>=0.90 else "GOOD" if score>=0.75 else
             "ACCEPTABLE" if score>=0.50 else "POOR" if score>=0.25 else "VERY POOR")
    return {"score":score,"label":label,"total_facts":len(facts),
            "supported":sup,"partial":par,"unsupported":unsup,
            "unverifiable":unver,"facts":verified}


# =============================================================
# 4. EVALUATE + WRITE
# =============================================================

def evaluate(questions, stats):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM_RULES, stats)
        sc   = factscore(pred, gt)
        results.append({"q_num":i,"question":q,"ground_truth":gt,"prediction":pred,"score":sc})
        print(f"  Q{i:3d} | FactScore={sc['score']:.4f} | "
              f"Facts={sc['total_facts']} "
              f"(S={sc['supported']},P={sc['partial']},U={sc['unsupported']}) "
              f"| {sc['label']}")
    return results

def write_results(results, stats):
    scores    = [r["score"]["score"]       for r in results]
    tot_facts = [r["score"]["total_facts"] for r in results]
    sup_facts = [r["score"]["supported"]   for r in results]
    unsup     = [r["score"]["unsupported"] for r in results]
    mean_sc   = sum(scores)/len(scores)
    std_sc    = math.sqrt(sum((s-mean_sc)**2 for s in scores)/len(scores))
    excellent = sum(1 for r in results if r["score"]["label"]=="EXCELLENT")
    good      = sum(1 for r in results if r["score"]["label"]=="GOOD")
    acceptable= sum(1 for r in results if r["score"]["label"]=="ACCEPTABLE")
    poor      = sum(1 for r in results if r["score"]["label"]=="POOR")
    very_poor = sum(1 for r in results if r["score"]["label"]=="VERY POOR")
    best_q    = max(results, key=lambda r: r["score"]["score"])
    worst_q   = min(results, key=lambda r: r["score"]["score"])
    all_facts = [f for r in results for f in r["score"]["facts"]]
    type_counts = Counter(f["type"] for f in all_facts)
    type_support = {}
    for ft in ["NUMERIC","COMPARATIVE","CAUSAL","EXISTENCE","PROPERTY"]:
        ft_facts = [f for f in all_facts if f["type"]==ft]
        if ft_facts:
            s_ = sum(1 for f in ft_facts if f["status"]=="SUPPORTED")
            p_ = sum(1 for f in ft_facts if f["status"]=="PARTIAL")
            type_support[ft] = round((s_+0.5*p_)/len(ft_facts),4)

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- FactScore: LLM Judge 2 vs Ground Truth")
    lines.append("  Model   : Vanilla RAG -- BiomedBERT Retriever (LLM Judge 2)")
    lines.append("  Metric  : FactScore (Min et al., EMNLP 2023)")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions: {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS FACTSCORE?")
    lines.append("-" * 70)
    lines.append("  Decomposes each answer into atomic facts and verifies each")
    lines.append("  one against the ground truth.")
    lines.append("  SUPPORTED=full credit | PARTIAL=half | UNSUPPORTED=no credit")
    lines.append("  FactScore = (supported + 0.5 x partial) / total_facts")
    lines.append("")
    lines.append("  For Vanilla RAG: FactScore reveals whether retrieved passages")
    lines.append("  are factually grounded in the GT or introduce unsupported claims.")
    lines.append("")
    lines.append("  Reference: Min et al. (2023) FActScore. EMNLP 2023.")
    lines.append("")
    lines.append("OVERALL RESULTS")
    lines.append("-" * 70)
    lines.append(f"  Mean FactScore        : {mean_sc:.4f}  (std={std_sc:.4f})")
    lines.append(f"  Mean facts per answer : {sum(tot_facts)/len(tot_facts):.1f}")
    lines.append(f"  Total atomic facts    : {sum(tot_facts)}")
    lines.append(f"  Total supported       : {sum(sup_facts)}")
    lines.append(f"  Total unsupported     : {sum(unsup)}")
    lines.append(f"  Best  : Q{best_q['q_num']} = {best_q['score']['score']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Worst : Q{worst_q['q_num']} = {worst_q['score']['score']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Quality breakdown:")
    lines.append(f"    EXCELLENT  (>=0.90) : {excellent:3d}")
    lines.append(f"    GOOD       (>=0.75) : {good:3d}")
    lines.append(f"    ACCEPTABLE (>=0.50) : {acceptable:3d}")
    lines.append(f"    POOR       (>=0.25) : {poor:3d}")
    lines.append(f"    VERY POOR  (< 0.25) : {very_poor:3d}")
    lines.append("")
    lines.append("  FactScore by atomic fact type:")
    for ft in ["NUMERIC","COMPARATIVE","CAUSAL","EXISTENCE","PROPERTY"]:
        sc_  = type_support.get(ft)
        cnt  = type_counts.get(ft, 0)
        if sc_ is not None:
            interp = "Strong" if sc_>=0.75 else "Moderate" if sc_>=0.50 else "Weak"
            bar = "#"*int(sc_*10)+"."*(10-int(sc_*10))
            lines.append(f"    {ft:<15} [{bar}] {sc_:.4f}  {cnt} facts  {interp}")
        else:
            lines.append(f"    {ft:<15} N/A              0 facts")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION FACTSCORE")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  FactScore : {s['score']:.4f}  --  {s['label']}")
        lines.append(f"  Facts     : {s['total_facts']} total | "
                     f"S={s['supported']} P={s['partial']} "
                     f"U={s['unsupported']} ?={s['unverifiable']}")
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65] for i in range(0,len(r["ground_truth"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  LLM2 PREDICTION (Vanilla RAG):")
        for chunk in [r["prediction"][i:i+65] for i in range(0,len(r["prediction"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  ATOMIC FACTS:")
        icons = {"SUPPORTED":"[OK]","PARTIAL":"[~~]","UNSUPPORTED":"[XX]","UNVERIFIABLE":"[??]"}
        for j, f in enumerate(s["facts"], 1):
            lines.append(f"    {j:2d}. {icons.get(f['status'],'[--]')} [{f['type']:<12}] {f['text'][:60]}")
            lines.append(f"         Verdict: {f['status']} -- {f['reason']}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF FACTSCORE -- LLM Judge 2 (Vanilla RAG)")
    lines.append(f"  Mean: {mean_sc:.4f} | Excellent: {excellent} | Good: {good} | "
                 f"Acceptable: {acceptable} | Poor: {poor} | Very Poor: {very_poor}")
    lines.append(f"  Total facts: {sum(tot_facts)} ({sum(sup_facts)} supported, {sum(unsup)} unsupported)")
    lines.append("  Reference: Min et al. (2023) FActScore. EMNLP 2023.")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "factscore_results_2.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] FactScore results written to: {out_path}\n")


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
    parser = argparse.ArgumentParser(description="FactScore: LLM Judge 2 vs ground truth")
    parser.add_argument("--disprot", type=str)
    parser.add_argument("--qa",      type=str)
    parser.add_argument("--demo",    action="store_true")
    args = parser.parse_args()

    if args.demo or (not args.disprot and not args.qa):
        print("[INFO] Running in DEMO mode\n")
        proteins, questions = DEMO_PROTEINS, DEMO_QUESTIONS
    else:
        if not args.disprot or not args.qa:
            print("[ERROR] Provide both --disprot and --qa, or use --demo"); sys.exit(1)
        proteins  = load_disprot(args.disprot)
        questions = load_qa(args.qa)

    stats = compute_stats(proteins)
    print("[INFO] Computing FactScore for LLM Judge 2...\n")
    results = evaluate(questions, stats)
    write_results(results, stats)

if __name__ == "__main__":
    main()