"""
BMEN-499 AlphaFold -- LLM Judge 2: Vanilla RAG Experiment
-----------------------------------------------------------
Purpose:
    Implements a vanilla RAG (Retrieval-Augmented Generation) pipeline
    using BiomedBERT as the retriever. Unlike LLM Judge 1 (which used
    symbolic rules + BiomedBERT), this is a pure neural RAG approach
    with NO symbolic rules or calibration -- a clean baseline comparison.

What is Vanilla RAG?
    1. INDEX    -- Build a knowledge base from DisProt protein facts
    2. RETRIEVE -- BiomedBERT encodes the question, finds the most
                   semantically similar facts from the knowledge base
    3. GENERATE -- Combine retrieved facts into a coherent answer

Why compare this to LLM Judge 1?
    LLM Judge 1 = BiomedBERT + symbolic rules + calibration
    LLM Judge 2 = BiomedBERT + vanilla RAG (no rules, no calibration)
    Comparing both shows whether symbolic rules actually improve answers
    over pure neural retrieval alone.

Output: LLM2_predictions.txt (saved to same folder as this script)

Usage:
    python LLM_judge2.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python LLM_judge2.py --demo
    python LLM_judge2.py --demo --no-bert   (keyword retrieval only, no model download)
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

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

def load_json(filepath: str, label: str):
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] {label} not found: {filepath}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {label}: {filepath}")
    return data


def load_disprot(filepath: str) -> list:
    raw = load_json(filepath, "DisProt dataset")
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    print(f"[INFO] {len(raw)} DisProt proteins loaded\n")
    return raw


def load_qa(filepath: str) -> list:
    raw = load_json(filepath, "QA dataset")
    if isinstance(raw, dict):
        raw = raw.get("questions", list(raw.values())[0])
    cleaned = [re.sub(r"^Q\d+[:\.\)]\s*", "", q.strip()) for q in raw]
    print(f"[INFO] {len(cleaned)} questions loaded\n")
    return cleaned


# =============================================================
# 2. COMPUTE STATS
# =============================================================

def compute_stats(proteins: list) -> dict:
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


# =============================================================
# 3. BUILD KNOWLEDGE BASE
#    The RAG knowledge base is a flat list of biomedical facts
#    derived directly from DisProt statistics.
#    Each fact is a short passage that BiomedBERT can retrieve.
# =============================================================

def build_knowledge_base(stats: dict) -> list:
    """
    Build a knowledge base of biomedical passages from DisProt stats.
    Each passage is a self-contained fact that can be retrieved
    independently and combined to answer a question.

    This is the RAG document corpus -- no rules, no calibration.
    Just raw factual passages indexed for retrieval.
    """
    kb = [
        {
            "id":      "KB-001",
            "topic":   "disorder score cutoff",
            "passage": (
                f"The 0.5 disorder score threshold is commonly used to classify protein "
                f"regions as intrinsically disordered. Analysis of {stats['total_proteins']:,} "
                f"DisProt proteins shows that {stats['pct_above_0.5']:.1f}% exceed this "
                f"threshold, with a dataset mean disorder score of {stats['mean_disorder']:.3f}. "
                f"However, {stats['pct_above_0.3']:.1f}% of proteins exceed 0.3, suggesting "
                f"many disordered regions fall below the 0.5 cutoff and would be missed."
            )
        },
        {
            "id":      "KB-002",
            "topic":   "gray zone ambiguous disorder",
            "passage": (
                f"Disorder scores between 0.3 and 0.5 define an ambiguous gray zone where "
                f"proteins cannot be confidently classified as ordered or disordered. "
                f"In the DisProt dataset of {stats['total_proteins']:,} proteins, a substantial "
                f"fraction falls in this mid-range. These regions require secondary validation "
                f"using sequence composition or experimental methods."
            )
        },
        {
            "id":      "KB-003",
            "topic":   "high confidence disorder strong signal",
            "passage": (
                f"Disorder scores above 0.7 represent high confidence intrinsic disorder. "
                f"These regions consistently correspond to experimentally validated IDRs "
                f"in DisProt and typically show AlphaFold pLDDT scores below 50, "
                f"indicating very low structural confidence from the AlphaFold model."
            )
        },
        {
            "id":      "KB-004",
            "topic":   "proline amino acid disorder prediction",
            "passage": (
                f"Proline is a strong predictor of intrinsic disorder. The mean proline "
                f"content across {stats['total_proteins']:,} DisProt proteins is "
                f"{stats['mean_proline']*100:.1f}%. Proline's rigid pyrrolidine ring "
                f"disrupts alpha-helices and beta-sheets, preventing the formation of "
                f"regular secondary structure. Regions with elevated proline are "
                f"consistently associated with disordered behavior."
            )
        },
        {
            "id":      "KB-005",
            "topic":   "glycine amino acid flexible backbone",
            "passage": (
                f"Glycine is the smallest amino acid and adds conformational freedom to "
                f"protein backbones. The mean glycine content in DisProt is "
                f"{stats['mean_glycine']*100:.1f}%. While elevated glycine can contribute "
                f"to disorder, it is a weaker independent predictor than proline and "
                f"should be evaluated in combination with other disorder signals."
            )
        },
        {
            "id":      "KB-006",
            "topic":   "proline glycine combined disorder signal composition",
            "passage": (
                f"When both proline ({stats['mean_proline']*100:.1f}% mean) and glycine "
                f"({stats['mean_glycine']*100:.1f}% mean) are elevated together, they "
                f"form a strong composite disorder signal. Proline introduces backbone "
                f"kinks while glycine adds excess conformational freedom -- together "
                f"they strongly predict intrinsically disordered regions."
            )
        },
        {
            "id":      "KB-007",
            "topic":   "short IDR region length prediction confidence",
            "passage": (
                f"Disordered regions shorter than 10 amino acids are difficult to predict "
                f"reliably. Of {stats['total_regions']:,} annotated disordered regions "
                f"in DisProt, {stats['pct_short_regions']:.1f}% are shorter than 10 "
                f"residues, with a mean region length of {stats['mean_region_length']:.1f} aa. "
                f"Short IDRs are underrepresented in experimental databases because "
                f"prediction tools lack sufficient sequence context for short stretches."
            )
        },
        {
            "id":      "KB-008",
            "topic":   "sliding window smoothing IDR signal noise",
            "passage": (
                f"Sliding window averaging is applied to per-residue disorder scores to "
                f"reduce noise. The mean disordered region length in DisProt is "
                f"{stats['mean_region_length']:.1f} amino acids. If the sliding window "
                f"size exceeds this mean, short disordered regions risk being smoothed "
                f"out and lost. Window size must be chosen carefully to balance noise "
                f"reduction against signal preservation."
            )
        },
        {
            "id":      "KB-009",
            "topic":   "AlphaFold pLDDT low confidence disorder",
            "passage": (
                f"AlphaFold assigns each amino acid a pLDDT confidence score from 0 to 100. "
                f"Scores below 50 indicate very low structural confidence and strongly "
                f"correlate with intrinsic disorder. DisProt experimentally confirms "
                f"disorder in {stats['total_proteins']:,} proteins; regions annotated as "
                f"disordered in DisProt consistently show pLDDT below 50 in AlphaFold "
                f"predictions, making this the most reliable single computational signal."
            )
        },
        {
            "id":      "KB-010",
            "topic":   "AlphaFold pLDDT moderate conditional disorder MoRF",
            "passage": (
                f"AlphaFold pLDDT scores between 50 and 70 indicate low but not absent "
                f"structural confidence. These regions may be conditionally disordered -- "
                f"unstructured in isolation but folding upon binding to a partner molecule. "
                f"Such regions are called Molecular Recognition Features (MoRFs) and "
                f"require experimental validation to distinguish from stably structured regions."
            )
        },
        {
            "id":      "KB-011",
            "topic":   "AlphaFold pLDDT high confidence structured folded",
            "passage": (
                f"AlphaFold pLDDT scores of 70 or above indicate high confidence in the "
                f"predicted structure. Regions with these scores are likely ordered and "
                f"not intrinsically disordered. Where DisProt experimental annotations "
                f"exist for the same region, experimental data should take precedence "
                f"over computational predictions."
            )
        },
        {
            "id":      "KB-012",
            "topic":   "Pfam domain structured mixed protein disorder",
            "passage": (
                f"{stats['pct_with_pfam']:.1f}% of DisProt proteins contain at least one "
                f"Pfam structured domain alongside their disordered regions. This confirms "
                f"that structured domains and intrinsically disordered regions frequently "
                f"co-occur in the same protein. Each region must be evaluated independently "
                f"rather than classifying the whole protein as ordered or disordered."
            )
        },
        {
            "id":      "KB-013",
            "topic":   "fully disordered protein IDP no domain",
            "passage": (
                f"Proteins with no detectable Pfam domains and high overall disorder content "
                f"are classified as Intrinsically Disordered Proteins (IDPs) or Fully "
                f"Disordered Proteins (FDPs). If mean disorder content exceeds 0.5 and no "
                f"structured domains are found, the protein is likely an IDP. These are "
                f"common in signaling, transcription regulation, and hub proteins."
            )
        },
        {
            "id":      "KB-014",
            "topic":   "disorder content distribution dataset statistics",
            "passage": (
                f"The DisProt database contains {stats['total_proteins']:,} experimentally "
                f"validated disordered proteins. The mean disorder content is "
                f"{stats['mean_disorder']:.3f}. {stats['pct_above_0.5']:.1f}% of proteins "
                f"exceed a disorder score of 0.5, and {stats['pct_above_0.3']:.1f}% exceed "
                f"0.3. There are {stats['total_regions']:,} annotated disordered regions "
                f"with a mean length of {stats['mean_region_length']:.1f} amino acids."
            )
        },
    ]
    return kb


# =============================================================
# 4. BIOMEDBERT ENCODER
# =============================================================

def load_biomedbert():
    """
    Load BiomedBERT for semantic encoding.
    Downloads ~440MB on first run and caches locally.
    """
    if not BERT_AVAILABLE:
        print("[WARNING] transformers not installed.")
        print("          Run: pip install transformers torch")
        print("          Falling back to keyword retrieval.\n")
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
        print("[WARNING] Falling back to keyword retrieval.\n")
        return None, None


def mean_pooling(model_output, attention_mask):
    token_embeddings     = model_output[0]
    input_mask_expanded  = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode(text: str, tokenizer, model) -> list:
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        output = model(**inputs)
    embedding = mean_pooling(output, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].tolist()


def cosine_sim(v1: list, v2: list) -> float:
    if not v1 or not v2:
        return 0.0
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


# =============================================================
# 5. VANILLA RAG PIPELINE
#    Retrieve -> Read -> Generate
# =============================================================

def index_knowledge_base(kb: list, tokenizer, model) -> list:
    """
    Pre-encode all knowledge base passages into embeddings.
    This is the indexing step of RAG -- done once before queries.
    """
    if tokenizer is None or model is None:
        return kb   # no embeddings, fall back to keyword retrieval

    print("[INFO] Indexing knowledge base passages...")
    for i, doc in enumerate(kb):
        doc["embedding"] = encode(doc["passage"], tokenizer, model)
        print(f"  Indexed {i+1}/{len(kb)}: {doc['id']}", end="\r")
    print(f"\n[INFO] Knowledge base indexed ({len(kb)} passages)\n")
    return kb


def retrieve(question: str, kb: list, tokenizer, model,
             top_k: int = 3) -> list:
    """
    Retrieve the top_k most relevant passages for the question.

    If BiomedBERT is available: use cosine similarity on embeddings.
    Otherwise: fall back to keyword overlap scoring.
    """
    q_lower = question.lower()

    if tokenizer and model and "embedding" in kb[0]:
        # Neural retrieval -- encode question and compare to KB
        q_emb  = encode(question, tokenizer, model)
        scored = [
            {"doc": doc, "score": cosine_sim(q_emb, doc["embedding"])}
            for doc in kb
        ]
    else:
        # Keyword fallback retrieval
        scored = []
        q_words = set(q_lower.split())
        for doc in kb:
            doc_words  = set(doc["topic"].lower().split())
            overlap    = len(q_words & doc_words)
            score      = overlap / max(len(q_words), 1)
            scored.append({"doc": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_answer(question: str, retrieved_docs: list) -> str:
    """
    Generate an answer by combining the retrieved passages.

    Vanilla RAG generation: concatenate retrieved passages in order
    of relevance. No rules, no calibration -- pure retrieval output.
    """
    if not retrieved_docs:
        return "No relevant passages found in the knowledge base."

    # Build answer from top retrieved passages
    parts = []
    for i, result in enumerate(retrieved_docs, 1):
        doc = result["doc"]
        parts.append(f"[Retrieved {i}] {doc['passage']}")

    return " ".join(parts)


def vanilla_rag_predict(question: str, kb: list, tokenizer, model,
                        top_k: int = 3) -> dict:
    """
    Full vanilla RAG prediction for a single question.
    Returns the answer, retrieved docs, and retrieval scores.
    """
    retrieved = retrieve(question, kb, tokenizer, model, top_k=top_k)
    answer    = generate_answer(question, retrieved)

    retrieval_method = "BiomedBERT semantic similarity" if (
        tokenizer and model and retrieved and "embedding" in kb[0]
    ) else "keyword overlap"

    return {
        "question":          question,
        "predicted_answer":  answer,
        "retrieved_docs":    [
            {
                "id":    r["doc"]["id"],
                "topic": r["doc"]["topic"],
                "score": round(r["score"], 4),
            }
            for r in retrieved
        ],
        "top_doc_id":        retrieved[0]["doc"]["id"] if retrieved else "NONE",
        "top_score":         round(retrieved[0]["score"], 4) if retrieved else 0.0,
        "retrieval_method":  retrieval_method,
        "quality": (
            "HIGH"   if retrieved and retrieved[0]["score"] > 0.7 else
            "MEDIUM" if retrieved and retrieved[0]["score"] > 0.3 else
            "LOW"
        )
    }


# =============================================================
# 6. WRITE LLM2_predictions.txt
# =============================================================

def write_predictions(predictions: list, stats: dict):
    lines = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 2: Vanilla RAG Predictions")
    lines.append("  Model     : microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    lines.append("  Method    : Vanilla RAG (retrieve + read, NO symbolic rules)")
    lines.append(f"  Dataset   : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions : {len(predictions)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS VANILLA RAG?")
    lines.append("-" * 70)
    lines.append("  Vanilla RAG has three steps:")
    lines.append("  1. INDEX    -- DisProt facts are encoded into a knowledge base")
    lines.append("  2. RETRIEVE -- BiomedBERT finds the most relevant facts for")
    lines.append("                 each question using semantic similarity")
    lines.append("  3. GENERATE -- Retrieved facts are combined into an answer")
    lines.append("")
    lines.append("  NOTE: Unlike LLM Judge 1, this pipeline has NO symbolic rules")
    lines.append("  and NO calibration. It is a pure neural retrieval baseline.")
    lines.append("  Comparing Judge 1 vs Judge 2 shows the value of symbolic rules.")
    lines.append("")

    high   = sum(1 for p in predictions if p["quality"] == "HIGH")
    medium = sum(1 for p in predictions if p["quality"] == "MEDIUM")
    low    = sum(1 for p in predictions if p["quality"] == "LOW")
    lines.append(f"  Prediction quality summary:")
    lines.append(f"    HIGH   : {high}  (retrieval score > 0.7)")
    lines.append(f"    MEDIUM : {medium}  (retrieval score 0.3-0.7)")
    lines.append(f"    LOW    : {low}  (retrieval score < 0.3)")
    lines.append("")

    for i, pred in enumerate(predictions, 1):
        lines.append("=" * 70)
        lines.append(f"[Q{i}] {pred['question']}")
        lines.append("")
        lines.append("  PREDICTED ANSWER (from retrieved passages):")

        # Wrap answer text at 65 chars
        words = pred["predicted_answer"].split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 67:
                lines.append(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            lines.append(line)

        lines.append("")
        lines.append("  RETRIEVAL DETAILS:")
        for j, doc in enumerate(pred["retrieved_docs"], 1):
            lines.append(
                f"    [{j}] {doc['id']} -- {doc['topic']:<40} score={doc['score']:.4f}"
            )
        lines.append("")
        lines.append(f"  Top retrieved doc  : {pred['top_doc_id']}")
        lines.append(f"  Top retrieval score: {pred['top_score']:.4f}")
        lines.append(f"  Retrieval method   : {pred['retrieval_method']}")
        lines.append(f"  Prediction quality : {pred['quality']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  COMPARISON NOTE")
    lines.append("-" * 70)
    lines.append("  LLM Judge 1 (BiomedBERT + Symbolic Rules + Calibration)")
    lines.append("    -- Rules ground answers in hard biological constraints")
    lines.append("    -- Calibrated confidence scores reflect real accuracy")
    lines.append("    -- More interpretable, explainable predictions")
    lines.append("")
    lines.append("  LLM Judge 2 (Vanilla RAG -- this file)")
    lines.append("    -- Pure neural retrieval, no hard constraints")
    lines.append("    -- Answers are data-driven but unconstrained")
    lines.append("    -- Useful baseline to measure symbolic rule benefit")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF PREDICTIONS")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    # Save to same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "LLM2_predictions.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Predictions written to: {out_path}\n")


# =============================================================
# DEMO DATA
# =============================================================

DEMO_PROTEINS = [
    {
        "disprot_id": "DP00001", "sequence": "MDVFMKGPSK" * 14,
        "disorder_content_pure": 0.35,
        "regions": [{"start": 96, "end": 140, "term_name": "disorder"}],
        "features": {"pfam": []}
    },
    {
        "disprot_id": "DP00003", "sequence": "MSSRRGPGGK" * 36,
        "disorder_content_pure": 0.098,
        "regions": [{"start": 1, "end": 50, "term_name": "disorder"}],
        "features": {"pfam": [{"id": "PF02236", "name": "Viral DBP", "start": 184, "end": 262}]}
    },
    {
        "disprot_id": "DP00010", "sequence": "MEEPQSDPGP" * 39,
        "disorder_content_pure": 0.62,
        "regions": [{"start": 1, "end": 67, "term_name": "disorder"}],
        "features": {"pfam": [{"id": "PF00870", "name": "P53 DBD", "start": 94, "end": 292}]}
    },
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
        description="Vanilla RAG predictions -- LLM Judge 2"
    )
    parser.add_argument("--disprot", type=str, help="Path to DisProt JSON")
    parser.add_argument("--qa",      type=str, help="Path to QA questions JSON")
    parser.add_argument("--demo",    action="store_true", help="Run with built-in sample data")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BiomedBERT, use keyword retrieval only")
    parser.add_argument("--top-k",   type=int, default=3,
                        help="Number of passages to retrieve per question (default: 3)")
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
    kb    = build_knowledge_base(stats)

    # Load BiomedBERT
    if args.no_bert:
        tokenizer, model = None, None
        print("[INFO] Running in keyword retrieval mode (--no-bert flag)\n")
    else:
        tokenizer, model = load_biomedbert()

    # Index knowledge base
    kb = index_knowledge_base(kb, tokenizer, model)

    # Generate predictions
    print(f"[INFO] Generating RAG predictions for {len(questions)} questions...\n")
    predictions = []
    for i, question in enumerate(questions, 1):
        print(f"  Q{i}/{len(questions)}: {question[:60]}...")
        pred = vanilla_rag_predict(question, kb, tokenizer, model, top_k=args.top_k)
        predictions.append(pred)

    write_predictions(predictions, stats)


if __name__ == "__main__":
    main()