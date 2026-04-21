"""
BMEN-499 AlphaFold -- LLM Judge 1: BiomedBERT Predictions
-----------------------------------------------------------
Purpose:
    Uses BiomedBERT (Microsoft's biomedical BERT model) combined with
    symbolic rules and calibration to generate predicted answers for
    each QA question. Results are written to LLM1_predictions.txt.

Pipeline:
    1. Load DisProt JSON        -> compute ground truth statistics
    2. Load QA questions        -> parse question list
    3. Load calibrated rules    -> symbolic reasoning layer
    4. For each question:
         a. BiomedBERT encodes the question (neural layer)
         b. Symbolic rules fire on relevant protein context
         c. Calibrated confidence scores weight the final answer
         d. Combined prediction is written to LLM1_predictions.txt

Why BiomedBERT instead of BioGPT here:
    BiomedBERT is an encoder model -- it deeply understands biomedical
    text and produces rich semantic embeddings. Combined with symbolic
    rules, it scores how well each candidate answer matches the question
    semantically, then the rules ground the answer in hard data.

Usage:
    python Data/LLM_judge1/LLM1_predictions.py --disprot Data/DisProt_ProteinData.json --qa Data/QA_Dataset.json
    python Data/LLM_judge1/LLM1_predictions.py --demo
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
# 2. DISPROT STATISTICS
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
# 3. CALIBRATED SYMBOLIC RULES
#    Confidence values updated from calibration.py output
# =============================================================

def build_calibrated_rules(stats: dict) -> list:
    """
    Symbolic rules with calibrated confidence scores applied.
    Each rule produces a candidate answer when its condition fires.
    Confidence scores here reflect empirically adjusted values
    from calibration.py.
    """
    return [
        {
            "rule_id":    "DR-001",
            "category":   "Disorder Threshold",
            "keywords":   ["disorder", "cutoff", "0.5", "threshold", "score"],
            "confidence": 0.96,   # calibrated from 0.85
            "answer": (
                f"Based on {stats['total_proteins']:,} DisProt proteins, a disorder "
                f"score above 0.5 is a commonly used cutoff but it is conservative. "
                f"Only {stats['pct_above_0.5']:.1f}% of proteins exceed 0.5, while "
                f"{stats['pct_above_0.3']:.1f}% exceed 0.3. Many true IDRs fall in "
                f"the 0.3-0.5 range and would be missed by a strict 0.5 threshold. "
                f"The cutoff is a useful starting point but not fully reliable on its own."
            )
        },
        {
            "rule_id":    "DR-002",
            "category":   "Disorder Threshold",
            "keywords":   ["gray zone", "ambiguous", "borderline", "0.3", "0.4"],
            "confidence": 0.92,   # calibrated from 0.75
            "answer": (
                f"Disorder scores between 0.3 and 0.5 represent an ambiguous gray zone. "
                f"These regions cannot be confidently classified as ordered or disordered "
                f"from the score alone. Secondary signals such as elevated proline "
                f"(mean {stats['mean_proline']*100:.1f}%) or glycine "
                f"(mean {stats['mean_glycine']*100:.1f}%) content should be checked "
                f"before making a final classification."
            )
        },
        {
            "rule_id":    "DR-003",
            "category":   "Disorder Threshold",
            "keywords":   ["high disorder", "confident", "0.7", "strong signal"],
            "confidence": 0.95,   # not fired in calibration -- keep assigned
            "answer": (
                f"A disorder score above 0.7 is a high confidence signal of intrinsic "
                f"disorder. Regions scoring this high consistently correspond to "
                f"experimentally confirmed IDRs in DisProt and show AlphaFold pLDDT "
                f"scores below 50, indicating very low structural confidence."
            )
        },
        {
            "rule_id":    "SC-001",
            "category":   "Sequence Composition",
            "keywords":   ["proline", "pro-rich", "composition", "amino acid"],
            "confidence": 0.95,   # calibrated from 0.82
            "answer": (
                f"Proline content is a strong predictor of intrinsic disorder. "
                f"The DisProt dataset mean proline fraction is {stats['mean_proline']*100:.1f}%. "
                f"Regions with proline content above 1.5x this mean are very likely "
                f"disordered because proline's rigid ring structure disrupts alpha-helices "
                f"and beta-sheets, preventing regular folding."
            )
        },
        {
            "rule_id":    "SC-002",
            "category":   "Sequence Composition",
            "keywords":   ["glycine", "gly-rich", "flexible", "backbone"],
            "confidence": 0.24,   # calibrated from 0.80 -- heavily downweighted
            "answer": (
                f"Glycine content alone is not a reliable disorder predictor in this dataset. "
                f"The DisProt mean glycine fraction is {stats['mean_glycine']*100:.1f}%. "
                f"While glycine adds backbone flexibility, calibration shows this rule "
                f"has low empirical accuracy when used independently. It should be "
                f"combined with other signals before drawing conclusions."
            )
        },
        {
            "rule_id":    "SC-003",
            "category":   "Sequence Composition",
            "keywords":   ["proline", "glycine", "pro-gly", "both", "combined"],
            "confidence": 0.88,   # not fired in calibration -- keep assigned
            "answer": (
                f"When both proline ({stats['mean_proline']*100:.1f}% mean) and glycine "
                f"({stats['mean_glycine']*100:.1f}% mean) are elevated together, this is "
                f"a strong composite disorder signal. The combination of backbone kinking "
                f"(proline) and excess conformational freedom (glycine) consistently "
                f"predicts intrinsically disordered regions."
            )
        },
        {
            "rule_id":    "RL-001",
            "category":   "Region Length",
            "keywords":   ["short", "residue", "length", "10", "small region"],
            "confidence": 0.78,   # not fired in calibration -- keep assigned
            "answer": (
                f"Disordered regions shorter than 10 amino acids are difficult to predict "
                f"reliably. Of {stats['total_regions']:,} regions in DisProt, "
                f"{stats['pct_short_regions']:.1f}% are shorter than 10 residues. "
                f"Short IDRs are underrepresented in experimental databases and prediction "
                f"tools lack sufficient sequence context to make confident calls."
            )
        },
        {
            "rule_id":    "RL-002",
            "category":   "Region Length",
            "keywords":   ["region length", "typical", "average", "residues", "long"],
            "confidence": 0.96,   # calibrated from 0.85
            "answer": (
                f"The average disordered region in DisProt spans "
                f"{stats['mean_region_length']:.1f} amino acids across "
                f"{stats['total_regions']:,} annotated regions. Regions of 10 or more "
                f"residues provide sufficient sequence context for reliable disorder "
                f"prediction. Sliding window methods work best when the window size "
                f"stays well below this mean to avoid smoothing out true signal."
            )
        },
        {
            "rule_id":    "AF-001",
            "category":   "AlphaFold pLDDT",
            "keywords":   ["plddt", "alphafold", "confidence", "below 50", "low"],
            "confidence": 0.98,   # calibrated from 0.92
            "answer": (
                f"AlphaFold pLDDT scores below 50 are strong computational evidence of "
                f"intrinsic disorder. DisProt experimentally confirms disorder in "
                f"{stats['total_proteins']:,} proteins; regions annotated as disordered "
                f"in DisProt consistently show pLDDT below 50 in AlphaFold predictions. "
                f"This is the most reliable single computational signal for disorder."
            )
        },
        {
            "rule_id":    "AF-002",
            "category":   "AlphaFold pLDDT",
            "keywords":   ["plddt", "moderate", "50", "70", "morf", "conditional"],
            "confidence": 0.92,   # calibrated from 0.72
            "answer": (
                f"pLDDT scores between 50 and 70 indicate low but not absent structural "
                f"confidence. These regions are ambiguous -- they may be conditionally "
                f"disordered, meaning floppy in isolation but structured when bound to "
                f"a partner molecule. These are called Molecular Recognition Features "
                f"(MoRFs) and require experimental validation."
            )
        },
        {
            "rule_id":    "AF-003",
            "category":   "AlphaFold pLDDT",
            "keywords":   ["plddt", "structured", "high", "above 70", "folded"],
            "confidence": 0.86,   # calibrated from 0.90
            "answer": (
                f"pLDDT scores of 70 or above indicate AlphaFold is confident in the "
                f"predicted structure. These regions are likely ordered and not "
                f"intrinsically disordered. If DisProt annotations exist for the same "
                f"region, they should take precedence as experimental ground truth."
            )
        },
        {
            "rule_id":    "SD-001",
            "category":   "Structural Domain",
            "keywords":   ["pfam", "domain", "structured", "mixed", "both"],
            "confidence": 0.96,   # calibrated from 0.87
            "answer": (
                f"{stats['pct_with_pfam']:.1f}% of DisProt proteins contain at least "
                f"one Pfam structured domain alongside their disordered regions. "
                f"This confirms that structured domains and IDRs frequently co-occur "
                f"in the same protein. Each region must be evaluated independently -- "
                f"the presence of a Pfam domain does not mean the whole protein is ordered."
            )
        },
        {
            "rule_id":    "SD-002",
            "category":   "Structural Domain",
            "keywords":   ["no domain", "fully disordered", "idp", "no pfam", "entirely"],
            "confidence": 0.94,   # calibrated from 0.80
            "answer": (
                f"Proteins with no Pfam domains are candidates for fully disordered "
                f"protein (FDP) or intrinsically disordered protein (IDP) classification. "
                f"If the whole-sequence disorder content exceeds 0.5 and no structured "
                f"domains are detected, the protein is likely an IDP. These are common "
                f"in signaling and regulatory pathways."
            )
        },
    ]


# =============================================================
# 4. BIOMEDBERT -- neural encoder
# =============================================================

def load_biomedbert():
    """
    Load BiomedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract).
    Downloads ~440MB on first run and caches locally.

    BiomedBERT is an encoder -- it converts text into semantic vectors
    so we can measure how similar a question is to each rule's topic.
    """
    if not BERT_AVAILABLE:
        print("[WARNING] transformers not installed.")
        print("          Run: pip install transformers torch")
        print("          Falling back to keyword-only matching.\n")
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
        print("[WARNING] Falling back to keyword-only matching.\n")
        return None, None


def mean_pooling(model_output, attention_mask):
    """Average token embeddings weighted by attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_text(text: str, tokenizer, model) -> list:
    """Encode a text string into a semantic embedding vector."""
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        output = model(**inputs)
    embedding = mean_pooling(output, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].tolist()


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    dot   = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


# =============================================================
# 5. PREDICTION ENGINE
#    Combines BiomedBERT semantic similarity + symbolic rules
#    + calibrated confidence to produce a final prediction
# =============================================================

def predict(question: str, rules: list, tokenizer, model,
            stats: dict) -> dict:
    """
    Generate a prediction for a question by:
      1. Encoding the question with BiomedBERT (if available)
      2. Scoring each rule by semantic similarity to the question
      3. Falling back to keyword matching if BERT unavailable
      4. Weighting scores by calibrated confidence
      5. Selecting the best matching rule's answer
      6. Flagging low confidence predictions for review
    """
    q_lower = question.lower()
    scored  = []

    # Encode question if BiomedBERT is available
    q_embedding = None
    if tokenizer and model:
        q_embedding = encode_text(question, tokenizer, model)

    for rule in rules:
        # Neural score: cosine similarity between question and rule keywords
        if q_embedding:
            rule_text      = f"{rule['category']} {' '.join(rule['keywords'])}"
            rule_embedding = encode_text(rule_text, tokenizer, model)
            neural_score   = cosine_similarity(q_embedding, rule_embedding)
        else:
            neural_score = 0.0

        # Symbolic score: keyword overlap
        keyword_hits  = sum(1 for kw in rule["keywords"] if kw in q_lower)
        symbolic_score = keyword_hits / len(rule["keywords"])

        # Combined score: neural (60%) + symbolic (40%), weighted by calibrated confidence
        if q_embedding:
            combined = (0.6 * neural_score + 0.4 * symbolic_score) * rule["confidence"]
        else:
            combined = symbolic_score * rule["confidence"]

        scored.append({
            "rule_id":       rule["rule_id"],
            "category":      rule["category"],
            "answer":        rule["answer"],
            "confidence":    rule["confidence"],
            "neural_score":  neural_score,
            "symbolic_score": symbolic_score,
            "combined_score": combined,
        })

    # Sort by combined score
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    # Safety guard: if no rules scored, return fallback
    if not scored:
        return {
            "question":         question,
            "predicted_answer": f"No rule matched. DisProt summary: {stats['total_proteins']:,} proteins, mean disorder={stats['mean_disorder']:.3f}.",
            "rule_used":        "NONE",
            "category":         "Fallback",
            "confidence":       0.0,
            "neural_score":     0.0,
            "symbolic_score":   0.0,
            "combined_score":   0.0,
            "quality":          "LOW -- no rules matched",
            "top_3_rules":      [],
        }

    best = scored[0]

    # Determine prediction quality
    if best["combined_score"] > 0.5:
        quality = "HIGH"
    elif best["combined_score"] > 0.2:
        quality = "MEDIUM"
    else:
        quality = "LOW -- fallback to general stats used"
        best["answer"] = (
            f"No strong rule match found. General DisProt summary: "
            f"{stats['total_proteins']:,} proteins, mean disorder = "
            f"{stats['mean_disorder']:.3f}, mean region length = "
            f"{stats['mean_region_length']:.1f} aa, "
            f"{stats['pct_above_0.5']:.1f}% exceed 0.5 disorder threshold."
        )

    return {
        "question":       question,
        "predicted_answer": best["answer"],
        "rule_used":      best["rule_id"],
        "category":       best["category"],
        "confidence":     best["confidence"],
        "neural_score":   round(best["neural_score"], 4),
        "symbolic_score": round(best["symbolic_score"], 4),
        "combined_score": round(best["combined_score"], 4),
        "quality":        quality,
        "top_3_rules":    [(s["rule_id"], round(s["combined_score"], 4)) for s in scored[:3]],
    }


# =============================================================
# 6. WRITE PREDICTIONS TO TEXT FILE
# =============================================================

def write_predictions(predictions: list, stats: dict, output_path: str):
    lines = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 1: BiomedBERT Predictions")
    lines.append("  Model   : microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    lines.append("  Method  : BiomedBERT semantic similarity + calibrated symbolic rules")
    lines.append(f"  Dataset : {stats['total_proteins']:,} DisProt proteins")
    lines.append(f"  Questions evaluated: {len(predictions)}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("HOW PREDICTIONS ARE MADE")
    lines.append("-" * 70)
    lines.append("  Step 1 -- BiomedBERT encodes each question into a semantic vector")
    lines.append("  Step 2 -- Each symbolic rule's topic is also encoded")
    lines.append("  Step 3 -- Cosine similarity finds the most relevant rule")
    lines.append("  Step 4 -- Calibrated confidence scores weight the match")
    lines.append("  Step 5 -- The best matching rule's answer becomes the prediction")
    lines.append("  Step 6 -- Prediction quality is flagged (HIGH / MEDIUM / LOW)")
    lines.append("")

    high   = sum(1 for p in predictions if p["quality"] == "HIGH")
    medium = sum(1 for p in predictions if p["quality"] == "MEDIUM")
    low    = sum(1 for p in predictions if "LOW" in p["quality"])

    lines.append(f"  Prediction quality summary:")
    lines.append(f"    HIGH   : {high}  questions")
    lines.append(f"    MEDIUM : {medium}  questions")
    lines.append(f"    LOW    : {low}  questions (general fallback used)")
    lines.append("")

    for i, pred in enumerate(predictions, 1):
        lines.append("=" * 70)
        lines.append(f"[Q{i}] {pred['question']}")
        lines.append("")
        lines.append("  PREDICTED ANSWER:")
        # Wrap answer text at 65 chars for readability
        words   = pred["predicted_answer"].split()
        line    = "  "
        for word in words:
            if len(line) + len(word) + 1 > 67:
                lines.append(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            lines.append(line)
        lines.append("")
        lines.append(f"  Rule used        : [{pred['rule_used']}] {pred['category']}")
        lines.append(f"  Calibrated conf  : {pred['confidence']:.0%}")
        lines.append(f"  Neural score     : {pred['neural_score']:.4f}  (BiomedBERT similarity)")
        lines.append(f"  Symbolic score   : {pred['symbolic_score']:.4f}  (keyword match)")
        lines.append(f"  Combined score   : {pred['combined_score']:.4f}  (weighted total)")
        lines.append(f"  Prediction quality: {pred['quality']}")
        lines.append(f"  Top 3 rule matches: {pred['top_3_rules']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF PREDICTIONS")
    lines.append(f"  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    # Always save to same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "LLM1_predictions.txt")

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
        description="BiomedBERT + Symbolic Rules predictions -- LLM Judge 1"
    )
    parser.add_argument("--disprot", type=str, help="Path to DisProt JSON")
    parser.add_argument("--qa",      type=str, help="Path to QA questions JSON")
    parser.add_argument("--demo",    action="store_true", help="Run with built-in sample data")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BiomedBERT, use keyword matching only")
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
    rules = build_calibrated_rules(stats)

    # Load BiomedBERT
    if args.no_bert:
        tokenizer, model = None, None
        print("[INFO] Running in keyword-only mode (--no-bert flag set)\n")
    else:
        tokenizer, model = load_biomedbert()

    # Generate predictions
    print(f"[INFO] Generating predictions for {len(questions)} questions...\n")
    predictions = []
    for i, question in enumerate(questions, 1):
        print(f"  Processing Q{i}/{len(questions)}: {question[:60]}...")
        pred = predict(question, rules, tokenizer, model, stats)
        predictions.append(pred)

    write_predictions(predictions, stats, "LLM1_predictions.txt")


if __name__ == "__main__":
    main()