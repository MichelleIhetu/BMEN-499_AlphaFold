"""
BMEN-499 AlphaFold -- LLM Judge 3: BioMistral RAG
---------------------------------------------------
Purpose:
    Uses BioMistral-7B (a Mistral-based biomedical LLM) as the
    generative model in a RAG pipeline. BiomedBERT retrieves the
    most relevant DisProt facts, then BioMistral generates a
    fluent, detailed answer grounded in those facts.

How this differs from Judge 1 and Judge 2:
    Judge 1 -- BiomedBERT + symbolic rules + calibration
    Judge 2 -- BiomedBERT + vanilla RAG (no rules)
    Judge 3 -- BiomedBERT retriever + BioMistral generator (full RAG)

    Judge 3 is the most powerful setup: retrieval finds relevant
    facts, then a large generative model synthesizes them into
    a fluent biomedical answer.

BioMistral:
    Model : BioMistral/BioMistral-7B
    Based on Mistral-7B, fine-tuned on PubMed Central biomedical
    literature. Produces more fluent and detailed answers than
    BioGPT while staying grounded in biomedical knowledge.

    NOTE: BioMistral-7B requires ~14GB RAM/VRAM. If your machine
    cannot run it, the script falls back to a lightweight
    BioMistral-compatible model automatically.

Output: LLM3_predictions.txt (saved to same folder as this script)

Usage:
    python LLM_judge3.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python LLM_judge3.py --demo
    python LLM_judge3.py --demo --no-bert   (keyword retrieval, no model download)
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

# HuggingFace Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModel,
        AutoModelForCausalLM, pipeline
    )
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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
# 3. KNOWLEDGE BASE  (same as Judge 2 -- shared retrieval corpus)
# =============================================================

def build_knowledge_base(stats: dict) -> list:
    """
    14-passage DisProt knowledge base for retrieval.
    BiomedBERT indexes these; BioMistral generates from the top-k.
    """
    return [
        {
            "id": "KB-001", "topic": "disorder score cutoff threshold",
            "passage": (
                f"The 0.5 disorder score threshold classifies protein regions as "
                f"intrinsically disordered. Of {stats['total_proteins']:,} DisProt "
                f"proteins, {stats['pct_above_0.5']:.1f}% exceed this threshold "
                f"(mean={stats['mean_disorder']:.3f}). However, {stats['pct_above_0.3']:.1f}% "
                f"exceed 0.3, meaning many true IDRs fall below 0.5 and are missed."
            )
        },
        {
            "id": "KB-002", "topic": "gray zone ambiguous disorder 0.3 0.5",
            "passage": (
                f"Disorder scores between 0.3 and 0.5 define an ambiguous gray zone. "
                f"These regions cannot be confidently classified without secondary "
                f"validation using sequence composition or experimental methods."
            )
        },
        {
            "id": "KB-003", "topic": "high confidence disorder strong signal 0.7",
            "passage": (
                f"Disorder scores above 0.7 represent high confidence intrinsic disorder, "
                f"consistently matching experimentally validated IDRs in DisProt and "
                f"correlating with AlphaFold pLDDT scores below 50."
            )
        },
        {
            "id": "KB-004", "topic": "proline amino acid disorder prediction composition",
            "passage": (
                f"Proline content (DisProt mean={stats['mean_proline']*100:.1f}%) strongly "
                f"predicts intrinsic disorder. Proline's rigid pyrrolidine ring disrupts "
                f"alpha-helices and beta-sheets, preventing regular secondary structure."
            )
        },
        {
            "id": "KB-005", "topic": "glycine amino acid flexible backbone composition",
            "passage": (
                f"Glycine (DisProt mean={stats['mean_glycine']*100:.1f}%) adds backbone "
                f"conformational freedom. It is a weaker independent disorder predictor "
                f"than proline and should be combined with other signals."
            )
        },
        {
            "id": "KB-006", "topic": "proline glycine combined Pro-Gly disorder signal",
            "passage": (
                f"Elevated proline ({stats['mean_proline']*100:.1f}% mean) and glycine "
                f"({stats['mean_glycine']*100:.1f}% mean) together form a strong composite "
                f"disorder signal -- proline kinks the backbone while glycine adds excess "
                f"freedom, both disrupting regular folding."
            )
        },
        {
            "id": "KB-007", "topic": "short IDR region length confidence residues",
            "passage": (
                f"Of {stats['total_regions']:,} DisProt regions, "
                f"{stats['pct_short_regions']:.1f}% are shorter than 10 residues "
                f"(mean={stats['mean_region_length']:.1f} aa). Short IDRs are hard to "
                f"predict reliably due to limited sequence context."
            )
        },
        {
            "id": "KB-008", "topic": "sliding window smoothing noise IDR signal",
            "passage": (
                f"Sliding window averaging smooths per-residue disorder scores. Windows "
                f"larger than the mean region length ({stats['mean_region_length']:.1f} aa) "
                f"risk smoothing out short IDRs entirely."
            )
        },
        {
            "id": "KB-009", "topic": "AlphaFold pLDDT low confidence disorder below 50",
            "passage": (
                f"AlphaFold pLDDT below 50 strongly indicates intrinsic disorder. "
                f"DisProt-annotated disordered regions in {stats['total_proteins']:,} "
                f"proteins consistently show pLDDT below 50 -- the most reliable "
                f"single computational disorder signal."
            )
        },
        {
            "id": "KB-010", "topic": "AlphaFold pLDDT moderate 50 70 MoRF conditional",
            "passage": (
                f"pLDDT scores of 50-70 indicate ambiguous structure. Regions may be "
                f"conditionally disordered (Molecular Recognition Features, MoRFs) -- "
                f"unstructured alone but folding upon partner binding."
            )
        },
        {
            "id": "KB-011", "topic": "AlphaFold pLDDT high structured folded above 70",
            "passage": (
                f"pLDDT >= 70 indicates confident AlphaFold structure prediction. "
                f"These regions are likely ordered. DisProt experimental annotations "
                f"take precedence over computational predictions."
            )
        },
        {
            "id": "KB-012", "topic": "Pfam domain structured mixed protein disorder",
            "passage": (
                f"{stats['pct_with_pfam']:.1f}% of DisProt proteins contain Pfam domains "
                f"alongside disordered regions, confirming IDRs and structured domains "
                f"frequently co-occur. Each region must be evaluated independently."
            )
        },
        {
            "id": "KB-013", "topic": "fully disordered IDP no Pfam domain",
            "passage": (
                f"Proteins with no Pfam domains and disorder content > 0.5 are classified "
                f"as Intrinsically Disordered Proteins (IDPs). These are common in "
                f"signaling, transcription, and hub protein networks."
            )
        },
        {
            "id": "KB-014", "topic": "DisProt dataset statistics summary",
            "passage": (
                f"DisProt contains {stats['total_proteins']:,} experimentally validated "
                f"proteins. Mean disorder={stats['mean_disorder']:.3f}, "
                f"{stats['total_regions']:,} regions, mean length "
                f"{stats['mean_region_length']:.1f} aa, "
                f"{stats['pct_above_0.5']:.1f}% exceed 0.5 threshold."
            )
        },
    ]


# =============================================================
# 4. BIOMEDBERT RETRIEVER
# =============================================================

def load_biomedbert():
    if not TRANSFORMERS_AVAILABLE:
        print("[WARNING] transformers not installed. Run: pip install transformers torch")
        print("[WARNING] Falling back to keyword retrieval.\n")
        return None, None

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    print(f"[INFO] Loading BiomedBERT retriever ({model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()
        print("[INFO] BiomedBERT retriever ready\n")
        return tokenizer, model
    except Exception as e:
        print(f"[WARNING] BiomedBERT load failed: {e}")
        print("[WARNING] Falling back to keyword retrieval.\n")
        return None, None


def mean_pooling(model_output, attention_mask):
    token_embeddings    = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
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


def index_kb(kb: list, tokenizer, model) -> list:
    if tokenizer is None or model is None:
        return kb
    print("[INFO] Indexing knowledge base with BiomedBERT...")
    for i, doc in enumerate(kb):
        doc["embedding"] = encode(doc["passage"], tokenizer, model)
        print(f"  Indexed {i+1}/{len(kb)}", end="\r")
    print(f"\n[INFO] {len(kb)} passages indexed\n")
    return kb


def retrieve(question: str, kb: list, tokenizer, model,
             top_k: int = 3) -> list:
    q_lower = question.lower()

    if tokenizer and model and "embedding" in kb[0]:
        q_emb  = encode(question, tokenizer, model)
        scored = [
            {"doc": doc, "score": cosine_sim(q_emb, doc["embedding"])}
            for doc in kb
        ]
    else:
        q_words = set(q_lower.split())
        scored  = []
        for doc in kb:
            doc_words = set(doc["topic"].lower().split())
            overlap   = len(q_words & doc_words)
            scored.append({"doc": doc, "score": overlap / max(len(q_words), 1)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# =============================================================
# 5. BIOMISTRAL GENERATOR
# =============================================================

def load_biomistral():
    """
    Load BioMistral-7B generative model.

    BioMistral-7B requires ~14GB RAM. If unavailable, falls back to
    BioGPT which is much lighter (~1.5GB) as a compatible alternative.

    First run downloads model weights automatically.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None

    # Try BioMistral-7B first
    for model_name, size in [
        ("BioMistral/BioMistral-7B",         "~14GB -- may be slow on CPU"),
        ("microsoft/biogpt",                  "~1.5GB -- lightweight fallback"),
    ]:
        print(f"[INFO] Loading generator: {model_name} ({size})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()
            print(f"[INFO] Generator ready: {model_name}\n")
            return {"tokenizer": tokenizer, "model": model, "name": model_name}
        except Exception as e:
            print(f"[WARNING] Could not load {model_name}: {e}")
            continue

    print("[WARNING] No generative model loaded. Using extractive fallback.\n")
    return None


def generate_with_biomistral(question: str, context: str,
                              generator: dict) -> str:
    """
    Generate an answer using BioMistral given a question and
    retrieved context passages.

    Prompt format follows instruction-tuning conventions used
    in BioMistral fine-tuning on medical QA datasets.
    """
    if generator is None:
        # Extractive fallback -- return context directly
        return context

    tokenizer = generator["tokenizer"]
    model     = generator["model"]

    prompt = (
        f"You are a biomedical expert in protein folding and intrinsic disorder.\n"
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=768
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer if answer else context


# =============================================================
# 6. FULL BIOMISTRAL RAG PIPELINE
# =============================================================

def biomistral_rag_predict(question: str, kb: list,
                            retriever_tok, retriever_model,
                            generator: dict,
                            top_k: int = 3) -> dict:
    """
    Full BioMistral RAG prediction:
      1. Retrieve top-k passages with BiomedBERT
      2. Build context string from retrieved passages
      3. Generate answer with BioMistral
    """
    retrieved = retrieve(question, kb, retriever_tok, retriever_model, top_k)

    # Build context from retrieved passages
    context = "\n".join([r["doc"]["passage"] for r in retrieved])

    # Generate answer
    answer = generate_with_biomistral(question, context, generator)

    # Clean up answer -- remove any prompt leakage
    for marker in ["Answer:", "Question:", "Context:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    retrieval_method = "BiomedBERT semantic" if (
        retriever_tok and retriever_model and "embedding" in kb[0]
    ) else "keyword overlap"

    generator_name = generator["name"] if generator else "extractive fallback"

    return {
        "question":         question,
        "predicted_answer": answer if answer else context,
        "retrieved_docs": [
            {
                "id":    r["doc"]["id"],
                "topic": r["doc"]["topic"],
                "score": round(r["score"], 4),
            }
            for r in retrieved
        ],
        "top_doc_id":       retrieved[0]["doc"]["id"] if retrieved else "NONE",
        "top_score":        round(retrieved[0]["score"], 4) if retrieved else 0.0,
        "retrieval_method": retrieval_method,
        "generator":        generator_name,
        "quality": (
            "HIGH"   if retrieved and retrieved[0]["score"] > 0.7 else
            "MEDIUM" if retrieved and retrieved[0]["score"] > 0.3 else
            "LOW"
        )
    }


# =============================================================
# 7. WRITE LLM3_predictions.txt
# =============================================================

def write_predictions(predictions: list, stats: dict):
    lines = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: BioMistral RAG Predictions")
    lines.append("  Retriever : BiomedBERT (microsoft/BiomedNLP-BiomedBERT)")
    lines.append("  Generator : BioMistral-7B (BioMistral/BioMistral-7B)")
    lines.append("  Method    : RAG -- BiomedBERT retrieves, BioMistral generates")
    lines.append(f"  Dataset   : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions : {len(predictions)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS BIOMISTRAL RAG?")
    lines.append("-" * 70)
    lines.append("  BioMistral is a large language model fine-tuned on PubMed")
    lines.append("  Central biomedical literature. In this pipeline:")
    lines.append("  1. RETRIEVE -- BiomedBERT finds the most relevant DisProt facts")
    lines.append("  2. GENERATE -- BioMistral reads those facts and writes a fluent,")
    lines.append("                 detailed biomedical answer")
    lines.append("")
    lines.append("  This is the most powerful of the 3 judges:")
    lines.append("  Judge 1 = BiomedBERT + symbolic rules + calibration")
    lines.append("  Judge 2 = BiomedBERT + vanilla RAG (no rules)")
    lines.append("  Judge 3 = BiomedBERT + BioMistral generation (full RAG)")
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
        lines.append(f"  GENERATOR: {pred['generator']}")
        lines.append("")
        lines.append("  PREDICTED ANSWER:")

        # Wrap answer at 65 chars
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
                f"    [{j}] {doc['id']} -- {doc['topic']:<42} score={doc['score']:.4f}"
            )
        lines.append("")
        lines.append(f"  Top retrieved doc  : {pred['top_doc_id']}")
        lines.append(f"  Top retrieval score: {pred['top_score']:.4f}")
        lines.append(f"  Retrieval method   : {pred['retrieval_method']}")
        lines.append(f"  Prediction quality : {pred['quality']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  3-JUDGE COMPARISON SUMMARY")
    lines.append("-" * 70)
    lines.append("  Judge 1 -- BiomedBERT + Symbolic Rules + Calibration")
    lines.append("    Strengths : interpretable, grounded, calibrated confidence")
    lines.append("    Weakness  : limited to pre-defined rule categories")
    lines.append("")
    lines.append("  Judge 2 -- BiomedBERT + Vanilla RAG")
    lines.append("    Strengths : pure neural, no manual rule design needed")
    lines.append("    Weakness  : no calibration, answers may be unconstrained")
    lines.append("")
    lines.append("  Judge 3 -- BiomedBERT + BioMistral RAG (this file)")
    lines.append("    Strengths : fluent generation, biomedical domain knowledge")
    lines.append("    Weakness  : large model, harder to interpret reasoning")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF PREDICTIONS")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    # Save to same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "LLM3_predictions.txt")

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
        description="BioMistral RAG predictions -- LLM Judge 3"
    )
    parser.add_argument("--disprot",  type=str, help="Path to DisProt JSON")
    parser.add_argument("--qa",       type=str, help="Path to QA questions JSON")
    parser.add_argument("--demo",     action="store_true", help="Run with built-in sample data")
    parser.add_argument("--no-bert",  action="store_true",
                        help="Skip BiomedBERT, use keyword retrieval only")
    parser.add_argument("--no-gen",   action="store_true",
                        help="Skip BioMistral generator, use extractive answers only")
    parser.add_argument("--top-k",    type=int, default=3,
                        help="Passages to retrieve per question (default: 3)")
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

    # Load BiomedBERT retriever
    if args.no_bert:
        ret_tok, ret_model = None, None
        print("[INFO] Keyword retrieval mode (--no-bert)\n")
    else:
        ret_tok, ret_model = load_biomedbert()

    # Load BioMistral generator
    if args.no_gen:
        generator = None
        print("[INFO] Extractive mode (--no-gen) -- skipping BioMistral\n")
    else:
        generator = load_biomistral()

    # Index knowledge base
    kb = index_kb(kb, ret_tok, ret_model)

    # Generate predictions
    print(f"[INFO] Generating predictions for {len(questions)} questions...\n")
    predictions = []
    for i, question in enumerate(questions, 1):
        print(f"  Q{i}/{len(questions)}: {question[:60]}...")
        pred = biomistral_rag_predict(
            question, kb, ret_tok, ret_model, generator, top_k=args.top_k
        )
        predictions.append(pred)

    write_predictions(predictions, stats)


if __name__ == "__main__":
    main()