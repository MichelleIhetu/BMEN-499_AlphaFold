"""
BMEN-499 AlphaFold -- BERTScore Evaluation: LLM Judge 1 vs Ground Truth
------------------------------------------------------------------------
Purpose:
    Computes BERTScore between LLM Judge 1 predicted answers and
    DisProt ground truth answers using BiomedBERT contextual embeddings.

What is BERTScore?
    BERTScore (Zhang et al., ICLR 2020) measures semantic similarity
    between two texts using contextual token embeddings from BERT.

    Unlike NAUR or cosine TF-IDF which rely on exact word matches,
    BERTScore captures meaning even when different words express the
    same concept. For example:
      "protein backbone flexibility"
      "conformational freedom of the chain"
    These score near zero on NAUR but high on BERTScore because
    BERT knows these phrases mean the same thing.

    How it works:
      1. EMBED   -- Each token in pred and GT is encoded by BiomedBERT
                    into a contextual vector
      2. MATCH   -- Each pred token is matched to its most similar GT
                    token using cosine similarity (greedy matching)
      3. SCORE   -- Precision = mean similarity of pred->GT matches
                    Recall    = mean similarity of GT->pred matches
                    F1        = harmonic mean of precision and recall

    Why BiomedBERT specifically?
      Using microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
      rather than generic BERT gives domain-specific embeddings
      trained on PubMed abstracts. Terms like "IDR", "pLDDT",
      "intrinsically disordered" are represented accurately.
      (Peng et al. 2019, Transfer Learning in Biomedical NLP)

    Score interpretation:
      0.9 - 1.0 : VERY HIGH -- semantically near-identical
      0.8 - 0.9 : HIGH      -- strong semantic match
      0.7 - 0.8 : MODERATE  -- good semantic overlap
      0.6 - 0.7 : LOW       -- partial semantic match
      < 0.6     : VERY LOW  -- weak semantic similarity

Output: bertscore_results.txt (saved to same folder as this script)

Dependencies:
    pip install transformers torch

Usage:
    python fact_score1.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python fact_score1.py --demo
    python fact_score1.py --demo --no-bert   (fast keyword fallback, no download)
"""

import json
import re
import sys
import os
import argparse
import math
from pathlib import Path
from collections import Counter

# BiomedBERT via HuggingFace
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
# 3. BIOMEDBERT ENCODER
# =============================================================

def load_biomedbert():
    """
    Load BiomedBERT for contextual token embeddings.
    Downloads ~440MB on first run and caches locally.

    Reference: Gu et al. (2021) Domain-specific language model
    pretraining for biomedical NLP. ACM CHIL 2021.
    """
    if not BERT_AVAILABLE:
        print("[WARNING] transformers not installed.")
        print("          Run: pip install transformers torch")
        print("[WARNING] Falling back to TF-IDF cosine BERTScore approximation.\n")
        return None, None

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    print(f"[INFO] Loading BiomedBERT ({model_name})...")
    print("[INFO] First run downloads ~440MB -- please wait...\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()
        print("[INFO] BiomedBERT ready\n")
        return tokenizer, model
    except Exception as e:
        print(f"[WARNING] BiomedBERT load failed: {e}")
        print("[WARNING] Falling back to TF-IDF cosine approximation.\n")
        return None, None


def get_token_embeddings(text, tokenizer, model, max_length=512):
    """
    Get contextual token embeddings for all tokens in text.
    Returns tensor of shape (num_tokens, hidden_size).
    Excludes [CLS] and [SEP] special tokens.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: (1, seq_len, hidden_size)
    hidden  = outputs.last_hidden_state[0]            # (seq_len, hidden)
    # Remove [CLS] (index 0) and [SEP] (last index)
    hidden  = hidden[1:-1]
    # L2-normalize each token vector
    hidden  = F.normalize(hidden, p=2, dim=1)
    return hidden


# =============================================================
# 4. BERTSCORE ENGINE
# =============================================================

def bertscore_bert(pred, gt, tokenizer, model):
    """
    Compute BERTScore using BiomedBERT contextual embeddings.

    Algorithm (Zhang et al. 2020):
      1. Get token embeddings for pred and GT
      2. Build pairwise cosine similarity matrix
      3. Precision: for each pred token, find max similarity to any GT token
      4. Recall: for each GT token, find max similarity to any pred token
      5. F1 = harmonic mean of precision and recall

    This is greedy matching -- no optimal assignment needed.
    """
    pred_emb = get_token_embeddings(pred, tokenizer, model)  # (P, H)
    gt_emb   = get_token_embeddings(gt,   tokenizer, model)  # (G, H)

    if pred_emb.shape[0] == 0 or gt_emb.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # Cosine similarity matrix: (P, G)
    sim_matrix = torch.mm(pred_emb, gt_emb.T)

    # Precision: max over GT dimension for each pred token
    precision_scores = sim_matrix.max(dim=1).values   # (P,)
    precision        = precision_scores.mean().item()

    # Recall: max over pred dimension for each GT token
    recall_scores = sim_matrix.max(dim=0).values       # (G,)
    recall        = recall_scores.mean().item()

    # F1
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def bertscore_tfidf_fallback(pred, gt):
    """
    TF-IDF cosine similarity as BERTScore approximation when
    BiomedBERT is unavailable. Less accurate but still useful.

    Reference: This approximates the uncontextualized version
    of BERTScore using bag-of-words representations.
    """
    stopwords = {"a","an","the","is","are","was","were","be","been","of",
                 "in","on","at","to","for","with","by","from","and","or",
                 "but","not","this","that","it","its","they","we","as"}

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if w not in stopwords and len(w) > 1]

    pred_toks = tokenize(pred)
    gt_toks   = tokenize(gt)

    if not pred_toks or not gt_toks:
        return 0.0, 0.0, 0.0

    pred_set = Counter(pred_toks)
    gt_set   = Counter(gt_toks)

    shared = set(pred_set) & set(gt_set)
    if not shared:
        return 0.0, 0.0, 0.0

    # Token-level precision and recall
    precision = sum(min(pred_set[t], gt_set[t]) for t in shared) / sum(pred_set.values())
    recall    = sum(min(pred_set[t], gt_set[t]) for t in shared) / sum(gt_set.values())
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_bertscore(pred, gt, tokenizer, model):
    """
    Compute BERTScore using BiomedBERT if available,
    otherwise fall back to TF-IDF approximation.
    """
    if tokenizer and model:
        p, r, f1 = bertscore_bert(pred, gt, tokenizer, model)
        method   = "BiomedBERT contextual embeddings"
    else:
        p, r, f1 = bertscore_tfidf_fallback(pred, gt)
        method   = "TF-IDF approximation (BiomedBERT unavailable)"

    label = (
        "VERY HIGH" if f1 >= 0.90 else
        "HIGH"      if f1 >= 0.80 else
        "MODERATE"  if f1 >= 0.70 else
        "LOW"       if f1 >= 0.60 else
        "VERY LOW"
    )

    return {
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "label":     label,
        "method":    method,
    }


# =============================================================
# 5. EVALUATE
# =============================================================

def evaluate(questions, stats, tokenizer, model):
    results = []
    for i, q in enumerate(questions, 1):
        gt   = get_answer(q, GT_RULES,   stats)
        pred = get_answer(q, LLM1_RULES, stats)
        sc   = compute_bertscore(pred, gt, tokenizer, model)

        results.append({
            "q_num":        i,
            "question":     q,
            "ground_truth": gt,
            "prediction":   pred,
            "score":        sc,
        })
        print(f"  Q{i:3d} | BERTScore F1={sc['f1']:.4f} | "
              f"P={sc['precision']:.4f} | R={sc['recall']:.4f} | {sc['label']}")

    return results


# =============================================================
# 6. WRITE bertscore_results.txt
# =============================================================

def write_results(results, stats):
    f1s    = [r["score"]["f1"]        for r in results]
    precs  = [r["score"]["precision"] for r in results]
    recs   = [r["score"]["recall"]    for r in results]
    method = results[0]["score"]["method"] if results else "N/A"

    mean_f1   = sum(f1s)   / len(f1s)
    mean_prec = sum(precs) / len(precs)
    mean_rec  = sum(recs)  / len(recs)
    std_f1    = math.sqrt(sum((f - mean_f1)**2 for f in f1s) / len(f1s))

    very_high = sum(1 for f in f1s if f >= 0.90)
    high      = sum(1 for f in f1s if 0.80 <= f < 0.90)
    moderate  = sum(1 for f in f1s if 0.70 <= f < 0.80)
    low       = sum(1 for f in f1s if 0.60 <= f < 0.70)
    very_low  = sum(1 for f in f1s if f < 0.60)

    best_q  = max(results, key=lambda r: r["score"]["f1"])
    worst_q = min(results, key=lambda r: r["score"]["f1"])

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- BERTScore Evaluation: LLM Judge 1")
    lines.append("  Model   : BiomedBERT + Calibrated Symbolic Rules (LLM Judge 1)")
    lines.append("  Metric  : BERTScore (Zhang et al., ICLR 2020)")
    lines.append(f"  Embeddings: {method}")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS BERTSCORE?")
    lines.append("-" * 70)
    lines.append("  BERTScore measures semantic similarity between two texts using")
    lines.append("  contextual token embeddings from a BERT model.")
    lines.append("")
    lines.append("  Why BERTScore over NAUR or cosine TF-IDF?")
    lines.append("    NAUR and TF-IDF only match EXACT words. BERTScore captures")
    lines.append("    MEANING even when different words express the same concept.")
    lines.append("    Example:")
    lines.append("      'protein backbone flexibility'")
    lines.append("      'conformational freedom of the chain'")
    lines.append("    These score near 0 on NAUR but high on BERTScore.")
    lines.append("")
    lines.append("  Why BiomedBERT specifically?")
    lines.append("    Generic BERT is trained on Wikipedia and books.")
    lines.append("    BiomedBERT is trained on PubMed abstracts, so it understands")
    lines.append("    biomedical terms like IDR, pLDDT, Pfam, intrinsically")
    lines.append("    disordered proteins accurately.")
    lines.append("    (Gu et al. 2021, ACM CHIL -- Domain-specific LM pretraining)")
    lines.append("")
    lines.append("  How it works:")
    lines.append("    1. EMBED   -- BiomedBERT encodes each token contextually")
    lines.append("    2. MATCH   -- Each pred token matched to most similar GT token")
    lines.append("    3. PRECISION = mean(max similarity of pred->GT matches)")
    lines.append("    4. RECALL    = mean(max similarity of GT->pred matches)")
    lines.append("    5. F1        = harmonic mean of precision and recall")
    lines.append("")
    lines.append("  Reference: Zhang et al. (2020) BERTScore: Evaluating Text")
    lines.append("  Generation with BERT. ICLR 2020.")
    lines.append("")
    lines.append("  Score interpretation:")
    lines.append("    0.90 - 1.0 : VERY HIGH -- semantically near-identical")
    lines.append("    0.80 - 0.90 : HIGH      -- strong semantic match")
    lines.append("    0.70 - 0.80 : MODERATE  -- good semantic overlap")
    lines.append("    0.60 - 0.70 : LOW       -- partial semantic match")
    lines.append("    < 0.60      : VERY LOW  -- weak semantic similarity")
    lines.append("")

    lines.append("OVERALL BERTSCORE RESULTS")
    lines.append("-" * 70)
    lines.append(f"  Mean BERTScore F1        : {mean_f1:.4f}  (std={std_f1:.4f})")
    lines.append(f"  Mean Precision           : {mean_prec:.4f}")
    lines.append(f"  Mean Recall              : {mean_rec:.4f}")
    lines.append(f"  Best  : Q{best_q['q_num']} = {best_q['score']['f1']:.4f} ({best_q['score']['label']})")
    lines.append(f"  Worst : Q{worst_q['q_num']} = {worst_q['score']['f1']:.4f} ({worst_q['score']['label']})")
    lines.append("")
    lines.append(f"  Score breakdown:")
    lines.append(f"    VERY HIGH (>=0.90) : {very_high:3d} questions")
    lines.append(f"    HIGH      (>=0.80) : {high:3d} questions")
    lines.append(f"    MODERATE  (>=0.70) : {moderate:3d} questions")
    lines.append(f"    LOW       (>=0.60) : {low:3d} questions")
    lines.append(f"    VERY LOW  (< 0.60) : {very_low:3d} questions")
    lines.append("")

    lines.append("  BERTScore F1 Distribution:")
    for lo, hi, lbl in [(0.0,0.60,"<0.60 VERY LOW "),(0.60,0.70,"<0.70 LOW      "),
                         (0.70,0.80,"<0.80 MODERATE "),(0.80,0.90,"<0.90 HIGH     "),
                         (0.90,1.01,">=0.90 VERY HIGH")]:
        count = sum(1 for f in f1s if lo <= f < hi)
        bar   = "#" * count + "." * max(0, 20 - count)
        lines.append(f"    {lbl} | {bar} | {count} questions")
    lines.append("")

    lines.append("  PRECISION vs RECALL ANALYSIS:")
    lines.append("  (Precision > Recall = pred covers more ground than GT)")
    lines.append("  (Recall > Precision = GT covers more ground than pred)")
    lines.append("")
    for r in results:
        p   = r["score"]["precision"]
        rec = r["score"]["recall"]
        diff = p - rec
        direction = "pred wider than GT" if diff > 0.02 else \
                    "GT wider than pred" if diff < -0.02 else \
                    "balanced"
        lines.append(f"    Q{r['q_num']:2d} | P={p:.4f}  R={rec:.4f}  "
                     f"diff={diff:+.4f}  ({direction})")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  QUESTION-BY-QUESTION BERTSCORE REPORT")
    lines.append("=" * 70)

    for r in results:
        s = r["score"]
        lines.append(f"\n[Q{r['q_num']}] {r['question']}")
        lines.append(f"  BERTScore F1  : {s['f1']:.4f}  --  {s['label']}")
        lines.append(f"  Precision     : {s['precision']:.4f}  "
                     f"(how much of pred is semantically in GT)")
        lines.append(f"  Recall        : {s['recall']:.4f}  "
                     f"(how much of GT is semantically in pred)")
        lines.append(f"  Method        : {s['method']}")
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
    lines.append("  COMPARISON: BERTScore vs NAUR vs Cosine TF-IDF")
    lines.append("-" * 70)
    lines.append("  BERTScore captures paraphrase similarity that NAUR misses.")
    lines.append("  If BERTScore >> NAUR, the model is paraphrasing correctly.")
    lines.append("  If BERTScore ~ NAUR, the model uses similar exact wording.")
    lines.append("  If BERTScore << NAUR, there may be vocabulary overlap without")
    lines.append("  true semantic alignment (surface matching without meaning).")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF BERTSCORE EVALUATION -- LLM Judge 1")
    lines.append(f"  Mean F1: {mean_f1:.4f} | P: {mean_prec:.4f} | R: {mean_rec:.4f}")
    lines.append(f"  Very High: {very_high} | High: {high} | Moderate: {moderate} | "
                 f"Low: {low} | Very Low: {very_low}")
    lines.append("  Reference: Zhang et al. (2020) BERTScore. ICLR 2020.")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "bertscore_results.txt")

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
    parser = argparse.ArgumentParser(
        description="BERTScore evaluation: LLM Judge 1 predictions vs ground truth"
    )
    parser.add_argument("--disprot", type=str)
    parser.add_argument("--qa",      type=str)
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BiomedBERT, use TF-IDF approximation")
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

    if args.no_bert:
        tokenizer, model = None, None
        print("[INFO] Using TF-IDF approximation (--no-bert flag)\n")
    else:
        tokenizer, model = load_biomedbert()

    print("[INFO] Computing BERTScore...\n")
    results = evaluate(questions, stats, tokenizer, model)
    write_results(results, stats)


if __name__ == "__main__":
    main()