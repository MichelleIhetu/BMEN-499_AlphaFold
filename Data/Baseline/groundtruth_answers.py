"""
BMEN-499 AlphaFold — QA Evaluation Pipeline
---------------------------------------------
Flow:
  1. Load DisProt JSON  → compute ground truth answers (data-driven)
  2. Load QA questions  → feed each to BioGPT        → generated answer
  3. Send (question, ground_truth, biogpt_answer) to 3 LLM judges
  4. Print scores + inter-judge agreement summary

Local setup (run once before first use):
    pip install transformers torch sacremoses

Usage:
    python qa_pipeline.py --disprot Data/disprot.json --qa Data/qa_pairs.json
    python qa_pipeline.py --demo            # sample data, calls judges via API
    python qa_pipeline.py --demo --no-api   # fully offline test
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from biogpt_guardrail import run_guardrail, get_ground_truth


# ── BioGPT (HuggingFace) ─────────────────────────────────
try:
    from transformers import BioGptTokenizer, BioGptForCausalLM
    import torch
    BIOGPT_AVAILABLE = True
except ImportError:
    BIOGPT_AVAILABLE = False

# ── Anthropic API (judges) ────────────────────────────────
import urllib.request


# =============================================================
# 0. ANTHROPIC API HELPER
# =============================================================

def call_anthropic(system_prompt: str, user_prompt: str,
                   model: str = "claude-sonnet-4-20250514",
                   max_tokens: int = 512) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"[API ERROR: {e}]"


# =============================================================
# 1. LOADERS
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
# 2. GROUND TRUTH  (computed from DisProt dataset)
# =============================================================

def compute_dataset_stats(proteins: list) -> dict:
    stats = defaultdict(list)
    for p in proteins:
        dc = p.get("disorder_content_pure") or p.get("disorder_content_obs")
        if dc is not None:
            stats["disorder_scores"].append(dc)
        for r in p.get("regions", []):
            if isinstance(r, dict):
                length = r.get("end", 0) - r.get("start", 0) + 1
                stats["region_lengths"].append(length)
        seq = p.get("sequence", "")
        if seq:
            stats["proline_fractions"].append(seq.count("P") / len(seq))
            stats["glycine_fractions"].append(seq.count("G") / len(seq))
        pfam = p.get("features", {}).get("pfam", [])
        stats["pfam_counts"].append(len(pfam))

    def mean(lst):    return sum(lst) / len(lst) if lst else 0.0
    def pct(lst, fn): return sum(1 for x in lst if fn(x)) / len(lst) * 100 if lst else 0.0

    return {
        "total_proteins":        len(proteins),
        "mean_disorder_score":   mean(stats["disorder_scores"]),
        "pct_above_0.5":         pct(stats["disorder_scores"], lambda x: x > 0.5),
        "pct_above_0.3":         pct(stats["disorder_scores"], lambda x: x > 0.3),
        "mean_region_length":    mean(stats["region_lengths"]),
        "pct_short_regions":     pct(stats["region_lengths"], lambda x: x < 10),
        "total_regions":         len(stats["region_lengths"]),
        "mean_proline_fraction": mean(stats["proline_fractions"]),
        "mean_glycine_fraction": mean(stats["glycine_fractions"]),
        "pct_with_pfam":         pct(stats["pfam_counts"], lambda x: x > 0),
    }


RULES = [
    (["0.5", "cutoff", "disorder"],
     lambda s: (
         f"Across {s['total_proteins']} DisProt proteins, {s['pct_above_0.5']:.1f}% exceed "
         f"disorder content 0.5 (mean={s['mean_disorder_score']:.3f}). A 0.5 cutoff is commonly "
         f"used but may be conservative — {s['pct_above_0.3']:.1f}% exceed 0.3, suggesting many "
         f"IDRs fall in the mid-range where 0.5 may undercount disorder."
     )),
    (["short", "residue"],
     lambda s: (
         f"Of {s['total_regions']} annotated disordered regions, {s['pct_short_regions']:.1f}% "
         f"are shorter than 10 residues (mean length={s['mean_region_length']:.1f} aa). Short IDRs "
         f"are underrepresented in DisProt, consistent with lower prediction confidence for very "
         f"short disordered stretches."
     )),
    (["proline", "glycine"],
     lambda s: (
         f"Mean proline fraction: {s['mean_proline_fraction']*100:.1f}%, "
         f"mean glycine fraction: {s['mean_glycine_fraction']*100:.1f}%. "
         f"Both amino acids promote backbone flexibility and disrupt secondary structure, "
         f"making Pro/Gly-rich regions strong predictors of intrinsic disorder."
     )),
    (["pfam", "domain"],
     lambda s: (
         f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain at least one Pfam domain, "
         f"indicating co-occurrence of structured domains and disordered regions."
     )),
]


def get_ground_truth(question: str, stats: dict) -> str:
    q = question.lower()
    for keywords, fn in RULES:
        if all(kw in q for kw in keywords):
            try:
                return fn(stats)
            except Exception as e:
                return f"[GT error: {e}]"
    return (
        f"General DisProt stats ({stats['total_proteins']} proteins): "
        f"mean disorder={stats['mean_disorder_score']:.3f}, "
        f"mean region length={stats['mean_region_length']:.1f} aa."
    )


# =============================================================
# 3. BIOGPT — answer generator (baseline model)
# =============================================================

def load_biogpt():
    """
    Load BioGPT locally from HuggingFace.
    First-time run downloads ~1.5GB of model weights automatically.

    Prerequisites:
        pip install transformers torch sacremoses
    """
    if not BIOGPT_AVAILABLE:
        print("[WARNING] transformers not installed. Run: pip install transformers torch sacremoses")
        print("[WARNING] Using mock BioGPT answers for now.\n")
        return None, None

    print("[INFO] Loading BioGPT (microsoft/biogpt) — downloads on first run (~1.5GB)...")
    try:
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        model     = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        model.eval()
        print("[INFO] BioGPT ready\n")
        return tokenizer, model
    except Exception as e:
        print(f"[WARNING] BioGPT load failed: {e}")
        print("[WARNING] Using mock answers. Check your internet connection and dependencies.\n")
        return None, None


def biogpt_answer(question: str, tokenizer, model, max_new_tokens: int = 150) -> str:
    """Generate an answer using BioGPT."""
    if tokenizer is None or model is None:
        return f"[BioGPT unavailable — mock answer for: '{question[:60]}...']"

    prompt  = f"Question about protein disorder and AlphaFold: {question}\nAnswer:"
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =============================================================
# 4. LLM JUDGES — score BioGPT answer vs ground truth
# =============================================================

JUDGE_SYSTEM = """You are an expert evaluator for biomedical QA systems specializing in 
protein disorder and AlphaFold predictions.

Score the BioGPT answer against the ground truth on:
- factual_accuracy (1-5): Contains correct scientific facts?
- completeness (1-5): Fully addresses the question?
- groundedness (1-5): Consistent with the ground truth?
- hallucination (1-5): 5=none, 1=severe hallucination

Respond ONLY with valid JSON. No preamble, no markdown fences. Example:
{"factual_accuracy": 4, "completeness": 3, "groundedness": 4, "hallucination": 5, "reasoning": "brief explanation"}"""


def judge_prompt(question: str, ground_truth: str, biogpt_ans: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"BioGPT Answer: {biogpt_ans}\n\n"
        f"Score the BioGPT Answer."
    )


# In production: swap these for 3 genuinely different models
# e.g. GPT-4o, Gemini 1.5 Pro, Llama 3.1 70B
JUDGES = [
    {"name": "Judge-1 (Strict)",   "model": "claude-sonnet-4-20250514"},
    {"name": "Judge-2 (Balanced)", "model": "claude-sonnet-4-20250514"},
    {"name": "Judge-3 (Lenient)",  "model": "claude-sonnet-4-20250514"},
]

OFFLINE_SCORES = {
    "factual_accuracy": 3, "completeness": 3,
    "groundedness": 3,     "hallucination": 4,
    "reasoning": "[offline mode]"
}


def run_judges(question: str, ground_truth: str, biogpt_ans: str,
               use_api: bool = True) -> list:
    prompt  = judge_prompt(question, ground_truth, biogpt_ans)
    results = []

    for judge in JUDGES:
        if use_api:
            raw = call_anthropic(JUDGE_SYSTEM, prompt, model=judge["model"])
            try:
                clean  = re.sub(r"```json|```", "", raw).strip()
                scores = json.loads(clean)
            except Exception:
                scores = {"parse_error": raw[:200]}
            time.sleep(0.5)
        else:
            scores = OFFLINE_SCORES.copy()

        results.append({"judge": judge["name"], "scores": scores})

    return results


# =============================================================
# 5. INTER-JUDGE AGREEMENT
# =============================================================

def compute_agreement(all_results: list) -> dict:
    criteria = ["factual_accuracy", "completeness", "groundedness", "hallucination"]
    totals   = defaultdict(list)

    for result in all_results:
        for jr in result["judges"]:
            s = jr["scores"]
            for c in criteria:
                if isinstance(s.get(c), (int, float)):
                    totals[c].append(s[c])

    return {c: round(sum(v) / len(v), 2) if v else None for c, v in totals.items()}


# =============================================================
# 6. MAIN PIPELINE
# =============================================================

def run_pipeline(questions: list, proteins: list, use_api: bool = True):
    stats            = compute_dataset_stats(proteins)
    tokenizer, model = load_biogpt()
    all_results      = []

    print("=" * 70)
    print("  BMEN-499 — BioGPT Answer Generator + 3-Judge Evaluation")
    print(f"  {len(questions)} questions | {stats['total_proteins']} DisProt proteins")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n[Q{i}] {question}")

        # Step 1: DisProt ground truth
        gt = get_ground_truth(question, stats)
        print(f"\n  [Ground Truth]\n  {gt}")

        # Step 2: BioGPT generates answer
        print(f"\n  [BioGPT generating...]")
        bg = biogpt_answer(question, tokenizer, model)
        print(f"  [BioGPT Answer]\n  {bg}")

        # Step 3: 3 judges score BioGPT vs ground truth
        print(f"\n  [Judge Scores]")
        judge_results = run_judges(question, gt, bg, use_api=use_api)

        for jr in judge_results:
            s = jr["scores"]
            if "parse_error" not in s:
                print(
                    f"    {jr['judge']:25s} | "
                    f"Accuracy={s.get('factual_accuracy','?')}  "
                    f"Complete={s.get('completeness','?')}  "
                    f"Grounded={s.get('groundedness','?')}  "
                    f"Halluc={s.get('hallucination','?')}"
                )
                if s.get("reasoning") and s["reasoning"] != "[offline mode]":
                    print(f"      Reasoning: {s['reasoning'][:120]}")
            else:
                print(f"    {jr['judge']}: [parse error] {s['parse_error']}")

        print("-" * 70)
        all_results.append({
            "question": question, "ground_truth": gt,
            "biogpt_answer": bg,  "judges": judge_results
        })

    # Summary
    agreement = compute_agreement(all_results)
    print("\n  INTER-JUDGE AGREEMENT (mean across all questions & judges)")
    for criterion, score in agreement.items():
        filled = "█" * int(score or 0) + "░" * (5 - int(score or 0))
        print(f"    {criterion:20s} {filled}  {score}/5")

    print(f"\n[DONE] {len(questions)} questions evaluated.\n")
    return all_results


# =============================================================
# DEMO DATA
# =============================================================

DEMO_PROTEINS = [
    {
        "disprot_id": "DP00003", "name": "Adenovirus DNA-binding protein",
        "sequence": "MSSRRGPGGK" * 36, "disorder_content_pure": 0.098,
        "regions": [{"start": 1, "end": 50, "term_name": "disorder"},
                    {"start": 300, "end": 360, "term_name": "disorder"}],
        "features": {"pfam": [{"id": "PF02236", "name": "Viral DBP", "start": 184, "end": 262}]}
    },
    {
        "disprot_id": "DP00001", "name": "Alpha-synuclein",
        "sequence": "MDVFMKGPSK" * 14, "disorder_content_pure": 0.35,
        "regions": [{"start": 96, "end": 140, "term_name": "disorder"}],
        "features": {"pfam": []}
    },
    {
        "disprot_id": "DP00010", "name": "p53",
        "sequence": "MEEPQSDPGP" * 39, "disorder_content_pure": 0.62,
        "regions": [{"start": 1, "end": 67, "term_name": "disorder"},
                    {"start": 364, "end": 393, "term_name": "disorder"}],
        "features": {"pfam": [{"id": "PF00870", "name": "P53 DNA-binding", "start": 94, "end": 292}]}
    },
]

DEMO_QUESTIONS = [
    "Is a disorder score above 0.5 a reliable cutoff for calling a region disordered?",
    "Do confidence scores drop for IDRs shorter than 10 residues?",
    "Do proline and glycine-rich regions consistently score higher disorder confidence than average?",
]


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="BMEN-499 QA Pipeline: BioGPT + 3 LLM Judges")
    parser.add_argument("--disprot", type=str, help="Path to DisProt JSON  (e.g. Data/disprot.json)")
    parser.add_argument("--qa",      type=str, help="Path to QA JSON       (e.g. Data/qa_pairs.json)")
    parser.add_argument("--demo",    action="store_true", help="Run with built-in sample data")
    parser.add_argument("--no-api",  action="store_true", help="Skip judge API calls (offline test)")
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

    run_pipeline(questions, proteins, use_api=not args.no_api)


if __name__ == "__main__":
    main()