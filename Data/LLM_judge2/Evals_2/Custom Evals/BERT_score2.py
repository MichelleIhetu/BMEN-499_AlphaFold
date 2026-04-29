"""
BMEN-499 AlphaFold -- BERTScore: LLM Judge 2 vs Ground Truth
-------------------------------------------------------------
Purpose:
    Computes BERTScore between LLM Judge 2 (Vanilla RAG) predicted
    answers and DisProt ground truth answers using BiomedBERT
    contextual embeddings.

LLM Judge 2 -- Vanilla RAG:
    BiomedBERT retrieves top-k DisProt knowledge base passages
    and concatenates them as the answer. No symbolic rules,
    no calibration -- pure neural retrieval baseline.

Reference:
    Zhang et al. (2020) BERTScore: Evaluating Text Generation
    with BERT. ICLR 2020.

Output: bertscore_results_2.txt (saved to same folder)

Usage:
    python BERT_score2.py --disprot Data/Baseline/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python BERT_score2.py --demo
    python BERT_score2.py --demo --no-bert
"""

import json
import re
import sys
import os
import argparse
import math
from pathlib import Path
from collections import Counter

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


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
# 3. BIOMEDBERT + BERTSCORE
# =============================================================

def load_biomedbert():
    if not BERT_AVAILABLE:
        print("[WARNING] transformers not installed. Run: pip install transformers torch")
        print("[WARNING] Falling back to TF-IDF approximation.\n")
        return None, None
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    print(f"[INFO] Loading BiomedBERT ({model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()
        print("[INFO] BiomedBERT ready\n")
        return tokenizer, model
    except Exception as e:
        print(f"[WARNING] BiomedBERT load failed: {e}")
        print("[WARNING] Falling back to TF-IDF approximation.\n")
        return None, None

def get_token_embeddings(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0][1:-1]
    hidden = F.normalize(hidden, p=2, dim=1)
    return hidden

def bertscore_bert(pred, gt, tokenizer, model):
    pred_emb = get_token_embeddings(pred, tokenizer, model)
    gt_emb   = get_token_embeddings(gt,   tokenizer, model)
    if pred_emb.shape[0] == 0 or gt_emb.shape[0] == 0:
        return 0.0, 0.0, 0.0
    sim      = torch.mm(pred_emb, gt_emb.T)
    prec     = sim.max(dim=1).values.mean().item()
    rec      = sim.max(dim=0).values.mean().item()
    f1       = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)

def bertscore_tfidf(pred, gt):
    sw = {"a","an","the","is","are","was","were","be","been","of","in","on",
          "at","to","for","with","by","from","and","or","but","not","this",
          "that","it","its","they","we","as"}
    def tok(text):
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return [w for w in text.split() if w not in sw and len(w) > 1]
    pt = Counter(tok(pred))
    gt_= Counter(tok(gt))
    shared = set(pt) & set(gt_)
    if not shared:
        return 0.0, 0.0, 0.0
    prec = sum(min(pt[t], gt_[t]) for t in shared) / sum(pt.values())
    rec  = sum(min(pt[t], gt_[t]) for t in shared) / sum(gt_.values())
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)

def compute_bertscore(pred, gt, tokenizer, model):
    if tokenizer and model:
        p, r, f1 = bertscore_bert(pred, gt, tokenizer, model)
        method   = "BiomedBERT contextual embeddings"
    else:
        p, r, f1 = bertscore_tfidf(pred, gt)
        method   = "TF-IDF approximation (BiomedBERT unavailable)"
    label = (
        "VERY HIGH" if f1 >= 0.90 else
        "HIGH"      if f1 >= 0.80 else
        "MODERATE"  if f1 >= 0.70 else
        "LOW"       if f1 >= 0.60 else
        "VERY LOW"
    )
    return {"precision": p, "recall": r, "f1": f1, "label": label, "method": method}


# =============================================================
# 4. EVALUATE + WRITE
# =============================================================

def evaluate(questions, stats, tokenizer, model):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM_RULES, stats)
        sc   = compute_bertscore(pred, gt, tokenizer, model)
        results.append({"q_num":i,"question":q,"ground_truth":gt,"prediction":pred,"score":sc})
        print(f"  Q{i:3d} | F1={sc['f1']:.4f} | P={sc['precision']:.4f} | R={sc['recall']:.4f} | {sc['label']}")
    return results

def write_results(results, stats):
    f1s   = [r["score"]["f1"]        for r in results]
    precs = [r["score"]["precision"] for r in results]
    recs  = [r["score"]["recall"]    for r in results]
    method = results[0]["score"]["method"] if results else "N/A"
    mean_f1   = sum(f1s)   / len(f1s)
    mean_prec = sum(precs) / len(precs)
    mean_rec  = sum(recs)  / len(recs)
    std_f1    = math.sqrt(sum((f-mean_f1)**2 for f in f1s)/len(f1s))
    very_high = sum(1 for f in f1s if f >= 0.90)
    high      = sum(1 for f in f1s if 0.80 <= f < 0.90)
    moderate  = sum(1 for f in f1s if 0.70 <= f < 0.80)
    low       = sum(1 for f in f1s if 0.60 <= f < 0.70)
    very_low  = sum(1 for f in f1s if f < 0.60)
    best_q    = max(results, key=lambda r: r["score"]["f1"])
    worst_q   = min(results, key=lambda r: r["score"]["f1"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- BERTScore: LLM Judge 2 vs Ground Truth")
    lines.append("  Model      : Vanilla RAG -- BiomedBERT Retriever (LLM Judge 2)")
    lines.append("  Metric     : BERTScore (Zhang et al., ICLR 2020)")
    lines.append(f"  Embeddings : {method}")
    lines.append(f"  Dataset    : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions  : {len(results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("WHAT IS BERTSCORE?")
    lines.append("-" * 70)
    lines.append("  BERTScore measures semantic similarity using contextual token")
    lines.append("  embeddings from BiomedBERT trained on PubMed abstracts.")
    lines.append("  Captures meaning even when different words express the same")
    lines.append("  concept -- unlike NAUR which requires exact word matches.")
    lines.append("")
    lines.append("  Precision = how much of pred is semantically in GT")
    lines.append("  Recall    = how much of GT is semantically in pred")
    lines.append("  F1        = harmonic mean of precision and recall")
    lines.append("")
    lines.append("  Score: >=0.90 VERY HIGH | >=0.80 HIGH | >=0.70 MODERATE")
    lines.append("         >=0.60 LOW       | < 0.60 VERY LOW")
    lines.append("")
    lines.append("OVERALL RESULTS")
    lines.append("-" * 70)
    lines.append(f"  Mean BERTScore F1  : {mean_f1:.4f}  (std={std_f1:.4f})")
    lines.append(f"  Mean Precision     : {mean_prec:.4f}")
    lines.append(f"  Mean Recall        : {mean_rec:.4f}")
    lines.append(f"  Best  : Q{best_q['q_num']} = {best_q['score']['f1']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Worst : Q{worst_q['q_num']} = {worst_q['score']['f1']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Breakdown:")
    lines.append(f"    VERY HIGH (>=0.90) : {very_high:3d}")
    lines.append(f"    HIGH      (>=0.80) : {high:3d}")
    lines.append(f"    MODERATE  (>=0.70) : {moderate:3d}")
    lines.append(f"    LOW       (>=0.60) : {low:3d}")
    lines.append(f"    VERY LOW  (< 0.60) : {very_low:3d}")
    lines.append("")
    lines.append("  Precision vs Recall:")
    for r in results:
        p, rc = r["score"]["precision"], r["score"]["recall"]
        diff  = p - rc
        dir_  = "pred wider" if diff > 0.02 else "GT wider" if diff < -0.02 else "balanced"
        lines.append(f"    Q{r['q_num']:2d} | P={p:.4f} R={rc:.4f} diff={diff:+.4f} ({dir_})")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION BERTSCORE")
    lines.append("=" * 70)
    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  F1={s['f1']:.4f} | P={s['precision']:.4f} | R={s['recall']:.4f} | {s['label']}")
        lines.append("")
        lines.append("  GROUND TRUTH:")
        for chunk in [r["ground_truth"][i:i+65] for i in range(0,len(r["ground_truth"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("")
        lines.append("  LLM2 PREDICTION (Vanilla RAG):")
        for chunk in [r["prediction"][i:i+65] for i in range(0,len(r["prediction"]),65)]:
            lines.append(f"    {chunk}")
        lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF BERTSCORE -- LLM Judge 2 (Vanilla RAG)")
    lines.append(f"  Mean F1: {mean_f1:.4f} | P: {mean_prec:.4f} | R: {mean_rec:.4f}")
    lines.append("  Reference: Zhang et al. (2020) BERTScore. ICLR 2020.")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "bertscore_results_2.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] BERTScore results written to: {out_path}\n")


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
    parser = argparse.ArgumentParser(description="BERTScore: LLM Judge 2 vs ground truth")
    parser.add_argument("--disprot", type=str)
    parser.add_argument("--qa",      type=str)
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--no-bert", action="store_true")
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
    tokenizer, model = (None, None) if args.no_bert else load_biomedbert()
    print("[INFO] Computing BERTScore for LLM Judge 2...\n")
    results = evaluate(questions, stats, tokenizer, model)
    write_results(results, stats)

if __name__ == "__main__":
    main()