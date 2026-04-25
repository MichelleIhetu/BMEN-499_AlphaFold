"""
BMEN-499 AlphaFold -- LLM Judge 3: Contradiction Count Analysis
----------------------------------------------------------------
File    : contradiction_count3.py
Output  : contradiction_count3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Scans every predicted answer in LLM3_predictions.txt for internal
    contradictions -- cases where the same answer asserts two
    logically conflicting claims about protein disorder.

Contradiction categories checked:
    C1  -- Threshold conflict       : score > 0.5 reliable  vs  0.5 misses true IDRs
    C2  -- pLDDT polarity flip      : pLDDT < 50 = disorder  vs  pLDDT >= 70 = ordered
                                      (flagged only when BOTH appear as definitive claims)
    C3  -- Short IDR confidence     : short IDRs unreliable  vs  short IDRs validated
    C4  -- Proline signal strength  : proline = strong predictor  vs  proline = weak
    C5  -- Glycine signal strength  : glycine = strong predictor  vs  glycine = weak
    C6  -- Gray zone assertion      : region is ambiguous  vs  region is classified
    C7  -- Experimental vs model    : DisProt takes precedence  vs  model is most reliable
    C8  -- Pfam co-occurrence       : structured + disordered co-occur  vs  classify whole protein
    C9  -- Sliding window risk      : window smooths IDRs away  vs  window preserves signal
    C10 -- IDP classification       : no Pfam + high disorder = IDP  vs  Pfam domains present

Scoring:
    Each question-answer pair is inspected for each of the 10 categories.
    A contradiction is flagged (1) or not (0) per category per question.
    The output reports:
        - Per-question contradiction flags and total count
        - Per-category contradiction frequency across all 100 questions
        - Severity tiers: NONE / LOW (1) / MEDIUM (2-3) / HIGH (4+)
        - Overall contradiction rate

Usage:
    python contradiction_count3.py
    (Run from the folder containing LLM3_predictions.txt, or set
     LLM3_PATH below to the absolute path of that file.)
"""

import os
import re
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH    = os.path.join(SCRIPT_DIR, "contradiction_count3_output.txt")

# ── Contradiction rule definitions ────────────────────────────────────
# Each rule is a dict with:
#   id       : rule identifier
#   label    : short name
#   desc     : human-readable description
#   sig_a    : list of phrases that assert position A
#   sig_b    : list of phrases that assert position B (conflicts with A)
#   logic    : "both"  -- contradiction if BOTH A and B appear in same answer
#              "a_not_b" -- contradiction if A appears but B doesn't qualify it
CONTRADICTION_RULES = [
    {
        "id":    "C1",
        "label": "Threshold reliability conflict",
        "desc":  "Answer states 0.5 is the standard cutoff, but also states 0.5 misses true IDRs",
        "sig_a": [
            "0.5 disorder score threshold classifies",
            "commonly used to classify",
            "reliable cutoff",
        ],
        "sig_b": [
            "many true idrs fall below 0.5",
            "would be missed",
            "fall below the 0.5 cutoff",
        ],
        "logic": "both",
    },
    {
        "id":    "C2",
        "label": "pLDDT polarity conflict",
        "desc":  "Answer conflates pLDDT < 50 (disorder) with pLDDT >= 70 (order) as equally 'reliable'",
        "sig_a": [
            "most reliable single computational signal",
            "strongly correlate with intrinsic disorder",
            "strongly indicates intrinsic disorder",
        ],
        "sig_b": [
            "plddT >= 70 indicates confident",
            "plddt of 70 or above indicate high confidence",
            "likely ordered and not intrinsically disordered",
        ],
        "logic": "both",
    },
    {
        "id":    "C3",
        "label": "Short IDR reliability conflict",
        "desc":  "Answer calls short IDRs unreliable but also cites them as validated DisProt regions",
        "sig_a": [
            "short idrs are hard to predict reliably",
            "difficult to predict reliably",
            "lack sufficient sequence context",
        ],
        "sig_b": [
            "experimentally validated",
            "consistently show plddt below 50",
            "experimentally confirms disorder",
        ],
        "logic": "both",
    },
    {
        "id":    "C4",
        "label": "Proline signal strength conflict",
        "desc":  "Answer calls proline a 'strong' predictor but elsewhere qualifies it as needing combination",
        "sig_a": [
            "strongly predicts intrinsic disorder",
            "strong predictor of intrinsic disorder",
            "consistently associated with disordered",
        ],
        "sig_b": [
            "should be evaluated in combination",
            "should be combined with other signals",
            "weaker independent predictor",
        ],
        "logic": "both",
    },
    {
        "id":    "C5",
        "label": "Glycine signal strength conflict",
        "desc":  "Answer implies glycine contributes meaningfully to disorder but also calls it weak",
        "sig_a": [
            "adds backbone conformational freedom",
            "can contribute to disorder",
            "adds conformational freedom",
        ],
        "sig_b": [
            "weaker independent predictor than proline",
            "weaker independent disorder predictor",
        ],
        "logic": "both",
    },
    {
        "id":    "C6",
        "label": "Gray zone classification conflict",
        "desc":  "Answer calls 0.3-0.5 range ambiguous/unclassifiable but also assigns a label to it",
        "sig_a": [
            "cannot be confidently classified",
            "ambiguous gray zone",
            "require secondary validation",
        ],
        "sig_b": [
            "these regions require secondary validation",
            "disordered regions in disprot",
            "experimentally confirmed",
        ],
        "logic": "both",
    },
    {
        "id":    "C7",
        "label": "Experimental vs computational precedence conflict",
        "desc":  "Answer states DisProt experimental data takes precedence, but also calls pLDDT the most reliable signal",
        "sig_a": [
            "experimental data should take precedence",
            "experimental annotations take precedence",
            "disprot experimental annotations take precedence",
        ],
        "sig_b": [
            "most reliable single computational",
            "most reliable single computational disorder signal",
        ],
        "logic": "both",
    },
    {
        "id":    "C8",
        "label": "Pfam co-occurrence vs whole-protein classification conflict",
        "desc":  "Answer notes IDRs and domains co-occur but also classifies the whole protein",
        "sig_a": [
            "each region must be evaluated independently",
            "co-occur in the same protein",
            "frequently co-occur",
        ],
        "sig_b": [
            "classif",
            "classified as intrinsically disordered proteins",
            "classified as idps",
        ],
        "logic": "both",
    },
    {
        "id":    "C9",
        "label": "Sliding window signal loss conflict",
        "desc":  "Answer warns sliding window smooths out short IDRs, but uses window averaging as a solution",
        "sig_a": [
            "risk smoothing out short idrs",
            "short disordered regions risk being smoothed",
            "risk being smoothed out and lost",
        ],
        "sig_b": [
            "sliding window averaging smooths",
            "applied to per-residue disorder scores to reduce noise",
            "smooth out confidence scores",
        ],
        "logic": "both",
    },
    {
        "id":    "C10",
        "label": "IDP classification vs Pfam presence conflict",
        "desc":  "Answer classifies protein as IDP (no Pfam) while also citing Pfam domain co-occurrence",
        "sig_a": [
            "proteins with no pfam domains",
            "no detectable pfam domains",
            "no pfam domains and disorder content",
        ],
        "sig_b": [
            "contain pfam domains alongside",
            "contain at least one pfam",
            "pfam structured domain alongside",
        ],
        "logic": "both",
    },
]

# ── Parse LLM3_predictions.txt ────────────────────────────────────────

def parse_predictions(filepath: str) -> list:
    """
    Extract (question_number, question_text, predicted_answer) tuples
    from LLM3_predictions.txt format.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] LLM3_predictions.txt not found at: {filepath}")
        print("        Place this script in the same folder as LLM3_predictions.txt")
        print("        or update LLM3_PATH at the top of this script.")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    # Split on question blocks
    blocks = re.split(r"={6,}", text)
    predictions = []

    q_pattern      = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    answer_pattern = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )

    for block in blocks:
        q_match = q_pattern.search(block)
        a_match = answer_pattern.search(block)
        if q_match and a_match:
            q_num    = int(q_match.group(1))
            q_text   = q_match.group(2).strip()
            a_text   = a_match.group(1).strip()
            # Clean up whitespace artifacts
            a_clean  = re.sub(r"\s+", " ", a_text).strip()
            predictions.append({
                "q_num":    q_num,
                "question": q_text,
                "answer":   a_clean,
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions


# ── Check contradictions ──────────────────────────────────────────────

def check_contradictions(answer: str, rules: list) -> dict:
    """
    Check a single answer string against all contradiction rules.
    Returns dict mapping rule_id -> bool (True = contradiction found).
    """
    a_lower = answer.lower()
    results = {}

    for rule in rules:
        sig_a_hit = any(s.lower() in a_lower for s in rule["sig_a"])
        sig_b_hit = any(s.lower() in a_lower for s in rule["sig_b"])

        if rule["logic"] == "both":
            results[rule["id"]] = sig_a_hit and sig_b_hit
        elif rule["logic"] == "a_not_b":
            results[rule["id"]] = sig_a_hit and not sig_b_hit
        else:
            results[rule["id"]] = False

    return results


def severity(count: int) -> str:
    if count == 0:    return "NONE"
    elif count == 1:  return "LOW"
    elif count <= 3:  return "MEDIUM"
    else:             return "HIGH"


# ── Main analysis ─────────────────────────────────────────────────────

def run_analysis(predictions: list) -> dict:
    per_question = []
    rule_totals  = {r["id"]: 0 for r in CONTRADICTION_RULES}

    for pred in predictions:
        flags        = check_contradictions(pred["answer"], CONTRADICTION_RULES)
        total_flags  = sum(flags.values())
        sev          = severity(total_flags)

        for rid, hit in flags.items():
            if hit:
                rule_totals[rid] += 1

        per_question.append({
            "q_num":       pred["q_num"],
            "question":    pred["question"],
            "flags":       flags,
            "total_flags": total_flags,
            "severity":    sev,
        })

    n_questions = len(predictions)
    n_with_any  = sum(1 for p in per_question if p["total_flags"] > 0)
    total_flags = sum(p["total_flags"] for p in per_question)

    severity_counts = {
        "NONE":   sum(1 for p in per_question if p["severity"] == "NONE"),
        "LOW":    sum(1 for p in per_question if p["severity"] == "LOW"),
        "MEDIUM": sum(1 for p in per_question if p["severity"] == "MEDIUM"),
        "HIGH":   sum(1 for p in per_question if p["severity"] == "HIGH"),
    }

    return {
        "per_question":    per_question,
        "rule_totals":     rule_totals,
        "n_questions":     n_questions,
        "n_with_any":      n_with_any,
        "total_flags":     total_flags,
        "severity_counts": severity_counts,
        "contradiction_rate": round(n_with_any / n_questions * 100, 2) if n_questions else 0,
        "mean_per_q":      round(total_flags / n_questions, 4) if n_questions else 0,
    }


# ── Write output ──────────────────────────────────────────────────────

def write_output(analysis: dict, out_path: str):
    lines = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Contradiction Count")
    lines.append("  Script  : contradiction_count3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {analysis['n_questions']}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("CONTRADICTION RULE DEFINITIONS")
    lines.append("-" * 70)
    for rule in CONTRADICTION_RULES:
        lines.append(f"  {rule['id']:4}  {rule['label']}")
        lines.append(f"        {rule['desc']}")
    lines.append("")

    lines.append("PER-QUESTION CONTRADICTION FLAGS")
    lines.append("-" * 70)
    rule_ids = [r["id"] for r in CONTRADICTION_RULES]
    header   = f"  {'Q':>4}  " + "  ".join(f"{rid:>4}" for rid in rule_ids) + f"  {'Total':>5}  {'Severity':<8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for p in analysis["per_question"]:
        flag_str = "  ".join(
            f"{'  1' if p['flags'][rid] else '  0':>4}" for rid in rule_ids
        )
        sev_color = {
            "NONE":   "NONE",
            "LOW":    "LOW ",
            "MEDIUM": "MED ",
            "HIGH":   "HIGH",
        }[p["severity"]]
        lines.append(
            f"  {p['q_num']:>4}  {flag_str}  {p['total_flags']:>5}  {sev_color}"
        )

    lines.append("")
    lines.append("PER-RULE TOTALS (across all questions)")
    lines.append("-" * 70)
    lines.append(f"  {'Rule':<6} {'Label':<40} {'Count':>6}  {'Rate':>8}")
    lines.append("  " + "-" * 62)
    for rule in CONTRADICTION_RULES:
        rid   = rule["id"]
        count = analysis["rule_totals"][rid]
        rate  = count / analysis["n_questions"] * 100 if analysis["n_questions"] else 0
        lines.append(f"  {rid:<6} {rule['label']:<40} {count:>6}  {rate:>7.1f}%")

    lines.append("")
    lines.append("SEVERITY DISTRIBUTION")
    lines.append("-" * 70)
    sc = analysis["severity_counts"]
    n  = analysis["n_questions"]
    lines.append(f"  NONE   (0 contradictions)  : {sc['NONE']:>4}  ({sc['NONE']/n*100:>5.1f}%)")
    lines.append(f"  LOW    (1 contradiction)   : {sc['LOW']:>4}  ({sc['LOW']/n*100:>5.1f}%)")
    lines.append(f"  MEDIUM (2-3 contradictions): {sc['MEDIUM']:>4}  ({sc['MEDIUM']/n*100:>5.1f}%)")
    lines.append(f"  HIGH   (4+ contradictions) : {sc['HIGH']:>4}  ({sc['HIGH']/n*100:>5.1f}%)")

    lines.append("")
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  Total questions              : {analysis['n_questions']}")
    lines.append(f"  Questions with >= 1 flag     : {analysis['n_with_any']}")
    lines.append(f"  Contradiction rate           : {analysis['contradiction_rate']:.2f}%")
    lines.append(f"  Total contradiction flags    : {analysis['total_flags']}")
    lines.append(f"  Mean contradictions per Q    : {analysis['mean_per_q']:.4f}")

    # Most contradicted rule
    top_rule_id    = max(analysis["rule_totals"], key=lambda k: analysis["rule_totals"][k])
    top_rule_count = analysis["rule_totals"][top_rule_id]
    top_rule_label = next(r["label"] for r in CONTRADICTION_RULES if r["id"] == top_rule_id)
    lines.append(f"  Most frequent contradiction  : {top_rule_id} -- {top_rule_label} ({top_rule_count}x)")

    # Worst question
    worst = max(analysis["per_question"], key=lambda p: p["total_flags"])
    lines.append(f"  Most contradicted question   : Q{worst['q_num']} ({worst['total_flags']} flags)")
    lines.append(f"    \"{worst['question'][:60]}...\"")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  LLM Judge 3 (BioMistral RAG) uses an extractive fallback when")
    lines.append("  BioMistral-7B is not loaded. The extractive mode concatenates")
    lines.append("  retrieved passages directly, which can produce internal")
    lines.append("  contradictions when the top-3 retrieved documents assert")
    lines.append("  conflicting claims (e.g., 'pLDDT is the most reliable signal'")
    lines.append("  AND 'experimental data takes precedence').")
    lines.append("")
    lines.append("  C7 (Experimental vs computational precedence) is expected to be")
    lines.append("  the most frequent contradiction because KB-009 ('pLDDT is the")
    lines.append("  most reliable signal') and KB-011 ('experimental data takes")
    lines.append("  precedence') are frequently retrieved together.")
    lines.append("")
    lines.append("  C4/C5 (Proline/Glycine signal strength) contradictions arise")
    lines.append("  because KB-004 labels proline a 'strong predictor' while KB-006")
    lines.append("  qualifies glycine as needing combination -- but KB-005 also")
    lines.append("  notes glycine 'adds conformational freedom', creating tension.")
    lines.append("")
    lines.append("  TO REDUCE CONTRADICTIONS: Run with --no-gen flag removed and")
    lines.append("  BioMistral-7B loaded. The generative model synthesizes retrieved")
    lines.append("  passages into a coherent answer that resolves conflicts, rather")
    lines.append("  than concatenating them directly.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] {out_path}")

    return output


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Loading predictions from: {LLM3_PATH}")
    predictions = parse_predictions(LLM3_PATH)
    print(f"[INFO] Parsed {len(predictions)} questions\n")

    print("[INFO] Running contradiction analysis...")
    analysis = run_analysis(predictions)

    write_output(analysis, OUT_PATH)
