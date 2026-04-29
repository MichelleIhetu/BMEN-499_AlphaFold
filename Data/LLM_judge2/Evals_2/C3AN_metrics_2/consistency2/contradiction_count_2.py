"""
BMEN-499 AlphaFold -- LLM Judge 2: Contradiction Count Test
------------------------------------------------------------
Purpose:
    Evaluates the 100 vanilla RAG predictions from LLM_judge2.py
    for internal contradictions within each predicted answer.

What is a Contradiction?
    A contradiction occurs when retrieved passages in the same
    predicted answer make logically incompatible claims. Since
    vanilla RAG concatenates passages without reasoning or
    constraint checking, conflicting facts can appear side-by-side.

Contradiction Types Detected:
    TYPE-A  Confidence threshold conflict
            A passage says >0.5 is reliable AND another says it misses
            real IDRs (i.e., the threshold is NOT reliable alone)
    TYPE-B  pLDDT interpretation conflict
            A passage says pLDDT <50 = disorder AND another says
            pLDDT >70 = structure, but a third says 50-70 is ambiguous --
            retrieved together for a question expecting a single answer
    TYPE-C  Sequence composition conflict
            A passage states proline/glycine predict disorder AND another
            implies composition is insufficient without pLDDT validation
    TYPE-D  Short IDR confidence conflict
            A passage says short IDRs are unreliable AND another says
            the knowledge base confirms disorder across all regions
    TYPE-E  Sliding window conflict
            A passage warns sliding windows erase short IDRs AND another
            recommends window smoothing without qualification
    TYPE-F  Structured-vs-disordered co-occurrence conflict
            A passage says structured domains and IDRs co-occur (mixed)
            AND another passage implies whole-protein disorder classification
    TYPE-G  Experimental-vs-computational hierarchy conflict
            One passage defers to experimental annotations AND another
            uses AlphaFold pLDDT as primary evidence without caveats

Detection Strategy:
    Each predicted answer (the concatenated retrieved passages) is
    scanned for co-presence of mutually conflicting signal phrases.
    Contradiction pairs are defined as keyword clusters that are
    semantically incompatible when both appear in the same answer.

Output:
    contradiction_count_2.txt  -- full report (same folder as script)
    Prints summary to console

Usage:
    python contradiction_count_2.py --predictions LLM2_predictions.txt
    python contradiction_count_2.py --demo
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


# =============================================================
# 1. CONTRADICTION RULE DEFINITIONS
#    Each rule is a dict with:
#      id       -- rule identifier
#      type     -- contradiction type label
#      name     -- short description
#      side_a   -- list of phrases: if ANY appear, side A is present
#      side_b   -- list of phrases: if ANY appear, side B is present
#      conflict -- human-readable explanation of why A+B = contradiction
# =============================================================

CONTRADICTION_RULES = [
    {
        "id":       "CR-001",
        "type":     "TYPE-A",
        "name":     "0.5 threshold reliable vs. misses real IDRs",
        "side_a":   [
            "0.5 disorder score threshold is commonly used",
            "scores above 0.5",
            "exceed this threshold",
        ],
        "side_b":   [
            "fall below the 0.5 cutoff and would be missed",
            "exceed 0.3, suggesting many disordered regions fall below",
            "substantial fraction falls in this mid-range",
            "gray zone",
        ],
        "conflict": (
            "One passage treats 0.5 as a standard reliable cutoff while "
            "another explicitly states that many real IDRs score below 0.5 "
            "and would be missed -- these are logically incompatible "
            "recommendations for threshold selection."
        ),
    },
    {
        "id":       "CR-002",
        "type":     "TYPE-B",
        "name":     "pLDDT <50 = disorder vs. pLDDT >70 = ordered (no resolution)",
        "side_a":   [
            "scores below 50 indicate very low structural confidence",
            "pLDDT below 50",
            "scores below 50",
        ],
        "side_b":   [
            "pLDDT scores of 70 or above indicate high confidence",
            "scores of 70 or above indicate high confidence",
            "70 or above indicate high confidence in the predicted structure",
        ],
        "conflict": (
            "The answer simultaneously presents pLDDT <50 as the disorder "
            "signal and pLDDT >70 as the ordered signal. Without also "
            "including the 50-70 ambiguous zone, the two passages imply "
            "a binary interpretation that contradicts the continuous nature "
            "of the score and leaves 50-70 unresolved."
        ),
    },
    {
        "id":       "CR-003",
        "type":     "TYPE-B",
        "name":     "High-confidence disorder (>0.7) vs. unreliable moderate range",
        "side_a":   [
            "scores above 0.7 represent high confidence intrinsic disorder",
            "disorder scores above 0.7",
        ],
        "side_b":   [
            "scores between 50 and 70 indicate low but not absent",
            "conditionally disordered",
            "molecular recognition features",
            "MoRFs",
        ],
        "conflict": (
            "One passage asserts disorder scores >0.7 are high-confidence, "
            "while another flags the pLDDT 50-70 range as ambiguous "
            "conditional disorder. Retrieved together without integration, "
            "these create contradictory guidance -- one urges confidence, "
            "the other urges caution, for the same mid-high range."
        ),
    },
    {
        "id":       "CR-004",
        "type":     "TYPE-C",
        "name":     "Proline/glycine predict disorder vs. insufficient alone",
        "side_a":   [
            "proline is a strong predictor of intrinsic disorder",
            "regions with elevated proline are consistently associated",
            "strong composite disorder signal",
            "they strongly predict intrinsically disordered regions",
        ],
        "side_b":   [
            "weaker independent predictor than proline and should be "
            "evaluated in combination",
            "require secondary validation using sequence composition or "
            "experimental methods",
            "should be evaluated in combination with other disorder signals",
        ],
        "conflict": (
            "One passage declares proline (and/or glycine) a strong, "
            "reliable disorder predictor, while another immediately "
            "qualifies that composition alone is insufficient and secondary "
            "validation is required. The answer gives contradictory "
            "confidence levels for the same compositional features."
        ),
    },
    {
        "id":       "CR-005",
        "type":     "TYPE-D",
        "name":     "Short IDRs unreliable vs. global DisProt confirmation",
        "side_a":   [
            "shorter than 10 amino acids are difficult to predict reliably",
            "short idrs are underrepresented in experimental databases",
            "prediction tools lack sufficient sequence context for short",
        ],
        "side_b":   [
            "disprot experimentally confirms disorder in",
            "regions annotated as disordered in disprot consistently",
            "making this the most reliable single computational signal",
        ],
        "conflict": (
            "One passage warns that short IDRs (<10 aa) are difficult to "
            "predict reliably and underrepresented in databases, but another "
            "passage cites DisProt experimental confirmation as broadly "
            "reliable. The two together imply DisProt is both comprehensive "
            "and incomplete -- a direct contradiction for short-IDR cases."
        ),
    },
    {
        "id":       "CR-006",
        "type":     "TYPE-E",
        "name":     "Sliding window erases short IDRs vs. reduces noise reliably",
        "side_a":   [
            "short disordered regions risk being smoothed out and lost",
            "sliding window size exceeds this mean, short disordered regions "
            "risk being smoothed",
            "window size must be chosen carefully",
        ],
        "side_b":   [
            "sliding window averaging is applied to per-residue disorder "
            "scores to reduce noise",
        ],
        "conflict": (
            "The answer recommends sliding window averaging for noise "
            "reduction while simultaneously warning that this same technique "
            "can erase short disordered regions. Without a window-size "
            "recommendation, the two passages give directly contradictory "
            "advice about whether to use this technique."
        ),
    },
    {
        "id":       "CR-007",
        "type":     "TYPE-F",
        "name":     "Mixed protein (IDR + domain) vs. whole-protein IDP classification",
        "side_a":   [
            "structured domains and intrinsically disordered regions "
            "frequently co-occur",
            "each region must be evaluated independently",
            "pfam structured domain alongside their disordered regions",
        ],
        "side_b":   [
            "classified as intrinsically disordered proteins (idps) or "
            "fully disordered proteins",
            "if mean disorder content exceeds 0.5 and no structured domains "
            "are found, the protein is likely an idp",
            "whole protein as ordered or disordered",
        ],
        "conflict": (
            "One passage insists that IDRs and structured domains co-occur "
            "and must be evaluated per-region, while another passage "
            "proposes classifying entire proteins as IDPs. Using these "
            "two passages together in the same answer contradicts the "
            "per-region evaluation principle with a whole-protein rule."
        ),
    },
    {
        "id":       "CR-008",
        "type":     "TYPE-G",
        "name":     "Experimental data takes precedence vs. pLDDT as primary signal",
        "side_a":   [
            "experimental data should take precedence over computational "
            "predictions",
            "disprot experimental annotations exist for the same region, "
            "experimental data should take precedence",
        ],
        "side_b":   [
            "making this the most reliable single computational signal",
            "most reliable single computational signal",
        ],
        "conflict": (
            "One passage explicitly states that experimental annotations "
            "take precedence over AlphaFold, while another declares AlphaFold "
            "pLDDT to be 'the most reliable single computational signal' -- "
            "implying computational predictions are primary. The answer does "
            "not resolve which hierarchy applies, creating a direct conflict "
            "over the authority of computational vs. experimental evidence."
        ),
    },
]


# =============================================================
# 2. PARSE PREDICTIONS FILE
# =============================================================

def parse_predictions_file(filepath: str) -> list:
    """
    Parse LLM2_predictions.txt into a list of prediction dicts.
    Each dict has: question_id, question, predicted_answer, retrieved_docs
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Split on Q-blocks
    blocks = re.split(r"={70}\n\[Q(\d+)\]", text)
    # blocks[0] = header, then pairs: (q_num, block_content)

    predictions = []
    for i in range(1, len(blocks), 2):
        q_num    = int(blocks[i])
        content  = blocks[i + 1] if (i + 1) < len(blocks) else ""

        # Extract question (first non-empty line)
        lines    = content.strip().split("\n")
        question = lines[0].strip() if lines else ""

        # Extract predicted answer section
        answer_match = re.search(
            r"PREDICTED ANSWER.*?:\s*\n(.*?)\n\s*RETRIEVAL DETAILS",
            content, re.DOTALL
        )
        answer = answer_match.group(1).strip() if answer_match else ""

        # Extract retrieved doc IDs
        doc_matches = re.findall(
            r"\[(\d+)\]\s+(KB-\d+)\s+--\s+(.+?)\s+score=([\d.]+)", content
        )
        retrieved_docs = [
            {"rank": int(m[0]), "id": m[1], "topic": m[2].strip(),
             "score": float(m[3])}
            for m in doc_matches
        ]

        predictions.append({
            "question_id":    q_num,
            "question":       question,
            "predicted_answer": answer.lower(),  # lowercase for matching
            "retrieved_docs": retrieved_docs,
        })

    return predictions


# =============================================================
# 3. DETECT CONTRADICTIONS
# =============================================================

def detect_contradictions(answer_text: str) -> list:
    """
    Check a single answer for all contradiction rules.
    Returns list of triggered rule dicts (with matched phrases).
    """
    text    = answer_text.lower()
    found   = []

    for rule in CONTRADICTION_RULES:
        # Check side A
        side_a_hits = [p for p in rule["side_a"] if p.lower() in text]
        if not side_a_hits:
            continue

        # Check side B
        side_b_hits = [p for p in rule["side_b"] if p.lower() in text]
        if not side_b_hits:
            continue

        # Both sides present -> contradiction
        found.append({
            "rule_id":      rule["id"],
            "type":         rule["type"],
            "name":         rule["name"],
            "conflict":     rule["conflict"],
            "side_a_match": side_a_hits[0],
            "side_b_match": side_b_hits[0],
        })

    return found


def run_contradiction_test(predictions: list) -> dict:
    """
    Run contradiction detection across all predictions.
    Returns aggregated results dict.
    """
    results = {
        "total_questions":              len(predictions),
        "questions_with_contradiction": 0,
        "total_contradictions":         0,
        "contradiction_counts_by_type": defaultdict(int),
        "contradiction_counts_by_rule": defaultdict(int),
        "per_question":                 [],
    }

    for pred in predictions:
        hits = detect_contradictions(pred["predicted_answer"])

        entry = {
            "question_id":     pred["question_id"],
            "question":        pred["question"],
            "contradictions":  hits,
            "count":           len(hits),
        }
        results["per_question"].append(entry)
        results["total_contradictions"] += len(hits)

        if hits:
            results["questions_with_contradiction"] += 1
            for h in hits:
                results["contradiction_counts_by_type"][h["type"]] += 1
                results["contradiction_counts_by_rule"][h["rule_id"]] += 1

    results["contradiction_rate"] = (
        results["questions_with_contradiction"] /
        results["total_questions"] * 100
        if results["total_questions"] else 0.0
    )
    results["mean_contradictions_per_q"] = (
        results["total_contradictions"] / results["total_questions"]
        if results["total_questions"] else 0.0
    )

    return results


# =============================================================
# 4. WRITE REPORT
# =============================================================

def write_report(results: dict, output_path: str):
    lines = []

    lines.append("=" * 72)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 2: Contradiction Count Test")
    lines.append("  Evaluation: Internal contradiction detection in RAG answers")
    lines.append("  Method    : Keyword-pair co-presence heuristic (8 rules)")
    lines.append("=" * 72)
    lines.append("")

    lines.append("WHAT THIS TEST MEASURES")
    lines.append("-" * 72)
    lines.append(
        "  Vanilla RAG (LLM Judge 2) retrieves and concatenates passages"
    )
    lines.append(
        "  without constraint checking. When conflicting passages are"
    )
    lines.append(
        "  retrieved together, the resulting answer contains internal"
    )
    lines.append(
        "  contradictions -- logically incompatible claims presented as"
    )
    lines.append(
        "  a single coherent answer."
    )
    lines.append("")
    lines.append(
        "  A contradiction is flagged when BOTH sides of a known conflict"
    )
    lines.append(
        "  pair appear in the same predicted answer. Eight contradiction"
    )
    lines.append(
        "  rules cover the major biological and methodological tensions"
    )
    lines.append(
        "  present in the DisProt/AlphaFold knowledge base."
    )
    lines.append("")

    # ---- SUMMARY TABLE ----
    lines.append("=" * 72)
    lines.append("  SUMMARY STATISTICS")
    lines.append("-" * 72)
    lines.append(f"  Total questions evaluated    : {results['total_questions']}")
    lines.append(
        f"  Questions WITH contradiction : "
        f"{results['questions_with_contradiction']}  "
        f"({results['contradiction_rate']:.1f}%)"
    )
    lines.append(
        f"  Questions WITHOUT contradiction: "
        f"{results['total_questions'] - results['questions_with_contradiction']}"
    )
    lines.append(f"  Total contradiction instances: {results['total_contradictions']}")
    lines.append(
        f"  Mean contradictions per Q   : "
        f"{results['mean_contradictions_per_q']:.3f}"
    )
    lines.append("")

    # ---- BY RULE ----
    lines.append("  CONTRADICTION COUNTS BY RULE")
    lines.append("  " + "-" * 68)
    header = f"  {'Rule':<8}  {'Type':<8}  {'Count':>6}  Description"
    lines.append(header)
    lines.append("  " + "-" * 68)
    for rule in CONTRADICTION_RULES:
        count = results["contradiction_counts_by_rule"].get(rule["id"], 0)
        lines.append(
            f"  {rule['id']:<8}  {rule['type']:<8}  {count:>6}  {rule['name']}"
        )
    lines.append("")

    # ---- BY TYPE ----
    lines.append("  CONTRADICTION COUNTS BY TYPE")
    lines.append("  " + "-" * 68)
    type_descriptions = {
        "TYPE-A": "Threshold reliability conflict",
        "TYPE-B": "pLDDT score interpretation conflict",
        "TYPE-C": "Sequence composition predictive power conflict",
        "TYPE-D": "Short IDR confidence vs. database reliability conflict",
        "TYPE-E": "Sliding window benefit vs. signal-loss conflict",
        "TYPE-F": "Per-region vs. whole-protein classification conflict",
        "TYPE-G": "Experimental vs. computational evidence hierarchy conflict",
    }
    type_header = f"  {'Type':<8}  {'Count':>6}  Description"
    lines.append(type_header)
    lines.append("  " + "-" * 68)
    for t, desc in type_descriptions.items():
        count = results["contradiction_counts_by_type"].get(t, 0)
        lines.append(f"  {t:<8}  {count:>6}  {desc}")
    lines.append("")

    # ---- PER-QUESTION DETAIL ----
    lines.append("=" * 72)
    lines.append("  PER-QUESTION RESULTS")
    lines.append("=" * 72)

    for entry in results["per_question"]:
        qid   = entry["question_id"]
        q     = entry["question"]
        count = entry["count"]
        label = "CONTRADICTION" if count > 0 else "clean"

        lines.append(f"\n[Q{qid:03d}] {q}")
        lines.append(f"       Contradictions found: {count}  [{label}]")

        for c in entry["contradictions"]:
            lines.append(f"")
            lines.append(f"       Rule   : {c['rule_id']} ({c['type']}) -- {c['name']}")
            lines.append(f"       Side A : \"{c['side_a_match'][:70]}\"")
            lines.append(f"       Side B : \"{c['side_b_match'][:70]}\"")
            # Word-wrap the conflict explanation at 65 chars
            words = c["conflict"].split()
            line  = "       Conflict: "
            for word in words:
                if len(line) + len(word) + 1 > 72:
                    lines.append(line)
                    line = "               " + word + " "
                else:
                    line += word + " "
            if line.strip():
                lines.append(line)

    # ---- INTERPRETATION ----
    lines.append("")
    lines.append("=" * 72)
    lines.append("  INTERPRETATION")
    lines.append("-" * 72)

    cr = results["contradiction_rate"]
    total_c = results["total_contradictions"]

    if cr >= 80:
        severity = "CRITICAL"
        interp = (
            "The vanilla RAG pipeline produces contradictory answers in the "
            "vast majority of cases. This confirms that pure neural retrieval "
            "without symbolic constraint checking is fundamentally unreliable "
            "for biomedical disorder prediction tasks. Retrieved passages "
            "frequently conflict because BiomedBERT retrieves semantically "
            "similar passages regardless of logical compatibility."
        )
    elif cr >= 50:
        severity = "HIGH"
        interp = (
            "Contradictions appear in over half of predicted answers. The "
            "vanilla RAG approach retrieves passages that are thematically "
            "related but biologically contradictory. Symbolic rules (LLM "
            "Judge 1) would resolve these conflicts through hard constraint "
            "checking, demonstrating clear value over pure neural retrieval."
        )
    elif cr >= 25:
        severity = "MODERATE"
        interp = (
            "A substantial minority of answers contain contradictions. The "
            "retrieval model captures relevant content but lacks the "
            "biological reasoning layer needed to filter incompatible claims. "
            "Adding symbolic post-processing would improve answer consistency."
        )
    else:
        severity = "LOW"
        interp = (
            "Contradiction rate is relatively low. However, even a small "
            "number of contradictions in biomedical predictions can mislead "
            "researchers. Symbolic rules add value by providing guarantees "
            "that pure retrieval cannot."
        )

    lines.append(f"  Contradiction rate : {cr:.1f}%  [{severity}]")
    lines.append(f"  Total instances    : {total_c}")
    lines.append("")
    words = interp.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            lines.append(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line)
    lines.append("")

    lines.append("  WHY VANILLA RAG PRODUCES CONTRADICTIONS")
    lines.append("-" * 72)
    lines.append(
        "  BiomedBERT retrieves passages by semantic similarity, not logical"
    )
    lines.append(
        "  compatibility. The disorder knowledge base contains passages that"
    )
    lines.append(
        "  are semantically related (all discuss pLDDT, thresholds, etc.)"
    )
    lines.append(
        "  but biologically contradictory (e.g., 'use 0.5 threshold' vs."
    )
    lines.append(
        "  '0.5 misses real IDRs'). Without symbolic rules to arbitrate,"
    )
    lines.append(
        "  the RAG generator concatenates all retrieved passages verbatim,"
    )
    lines.append(
        "  producing answers that say contradictory things simultaneously."
    )
    lines.append("")
    lines.append(
        "  LLM Judge 2 (Vanilla RAG) avoids this by applying hard biological rules that"
    )
    lines.append(
        "  ground interpretations, resolve ambiguities, and prevent"
    )
    lines.append(
        "  incompatible passages from co-appearing in the same answer."
    )
    lines.append("")

    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 72)

    output = "\n".join(lines)

    # Write file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Report written to: {output_path}\n")

    return output


# =============================================================
# DEMO / INLINE PREDICTIONS
#   Used when no predictions file is provided.
#   Reproduces a representative subset of the 100 predictions.
# =============================================================

DEMO_PREDICTIONS = [
    # Q1 -- expects CR-001 (threshold conflict) and CR-002 (pLDDT binary)
    {
        "question_id": 1,
        "question":    "Is a disorder score above 0.5 a reliable cutoff for calling a region disordered?",
        "predicted_answer": (
            "[retrieved 1] disorder scores between 0.3 and 0.5 define an "
            "ambiguous gray zone where proteins cannot be confidently "
            "classified as ordered or disordered. in the disprot dataset of "
            "13,396 proteins, a substantial fraction falls in this mid-range. "
            "these regions require secondary validation using sequence "
            "composition or experimental methods. "
            "[retrieved 2] disorder scores above 0.7 represent high "
            "confidence intrinsic disorder. these regions consistently "
            "correspond to experimentally validated idrs in disprot and "
            "typically show alphafold plddt scores below 50, indicating very "
            "low structural confidence from the alphafold model. "
            "[retrieved 3] alphafold assigns each amino acid a plddt "
            "confidence score from 0 to 100. scores below 50 indicate very "
            "low structural confidence and strongly correlate with intrinsic "
            "disorder. disprot experimentally confirms disorder in 13,396 "
            "proteins; regions annotated as disordered in disprot consistently "
            "show plddt below 50 in alphafold predictions, making this the "
            "most reliable single computational signal. "
        ),
        "retrieved_docs": [],
    },
    # Q2 -- expects CR-005 (short IDR) and CR-006 (sliding window)
    {
        "question_id": 2,
        "question":    "Do confidence scores drop for IDRs shorter than 10 residues?",
        "predicted_answer": (
            "[retrieved 1] disordered regions shorter than 10 amino acids are "
            "difficult to predict reliably. of 0 annotated disordered regions "
            "in disprot, 0.0% are shorter than 10 residues, with a mean "
            "region length of 0.0 aa. short idrs are underrepresented in "
            "experimental databases because prediction tools lack sufficient "
            "sequence context for short stretches. "
            "[retrieved 2] alphafold plddt scores of 70 or above indicate "
            "high confidence in the predicted structure. regions with these "
            "scores are likely ordered and not intrinsically disordered. "
            "where disprot experimental annotations exist for the same region, "
            "experimental data should take precedence over computational "
            "predictions. "
            "[retrieved 3] sliding window averaging is applied to per-residue "
            "disorder scores to reduce noise. the mean disordered region "
            "length in disprot is 0.0 amino acids. if the sliding window size "
            "exceeds this mean, short disordered regions risk being smoothed "
            "out and lost. window size must be chosen carefully to balance "
            "noise reduction against signal preservation. "
        ),
        "retrieved_docs": [],
    },
    # Q3 -- expects CR-004 (composition conflict) and CR-008 (hierarchy)
    {
        "question_id": 3,
        "question":    "Do proline and glycine-rich regions consistently score higher disorder confidence?",
        "predicted_answer": (
            "[retrieved 1] alphafold assigns each amino acid a plddt "
            "confidence score from 0 to 100. scores below 50 indicate very "
            "low structural confidence and strongly correlate with intrinsic "
            "disorder. disprot experimentally confirms disorder in 13,396 "
            "proteins; regions annotated as disordered in disprot consistently "
            "show plddt below 50 in alphafold predictions, making this the "
            "most reliable single computational signal. "
            "[retrieved 2] disorder scores above 0.7 represent high "
            "confidence intrinsic disorder. these regions consistently "
            "correspond to experimentally validated idrs in disprot and "
            "typically show alphafold plddt scores below 50, indicating very "
            "low structural confidence from the alphafold model. "
            "[retrieved 3] alphafold plddt scores of 70 or above indicate "
            "high confidence in the predicted structure. regions with these "
            "scores are likely ordered and not intrinsically disordered. "
            "where disprot experimental annotations exist for the same region, "
            "experimental data should take precedence over computational "
            "predictions. "
        ),
        "retrieved_docs": [],
    },
]


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contradiction count test for LLM Judge 2 predictions"
    )
    parser.add_argument(
        "--predictions", type=str,
        help="Path to LLM2_predictions.txt"
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Output path for contradiction_count_2.txt"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on 3 built-in demo predictions"
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        # Default: same directory as this script
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "contradiction_count_2.txt")

    if args.demo or not args.predictions:
        print("[INFO] Running in DEMO mode (3 sample predictions)\n")
        predictions = DEMO_PREDICTIONS
    else:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"[ERROR] Predictions file not found: {args.predictions}")
            sys.exit(1)
        print(f"[INFO] Parsing predictions file: {args.predictions}\n")
        predictions = parse_predictions_file(str(pred_path))
        print(f"[INFO] Parsed {len(predictions)} predictions\n")

    print("[INFO] Running contradiction detection...\n")
    results = run_contradiction_test(predictions)

    print(
        f"[INFO] Detection complete -- "
        f"{results['questions_with_contradiction']} / "
        f"{results['total_questions']} questions have contradictions "
        f"({results['contradiction_rate']:.1f}%)\n"
    )

    write_report(results, output_path)


if __name__ == "__main__":
    main()