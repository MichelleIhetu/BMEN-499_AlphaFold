"""
BMEN-499 AlphaFold -- Symbolic Rule Calibration
-------------------------------------------------
Purpose:
    Calibration checks whether the confidence scores assigned to each
    symbolic rule actually match how often the rule is correct.

    Example: If Rule DR-001 claims 85% confidence, it should be correct
    roughly 85% of the time when tested against DisProt ground truth.
    If it is only correct 60% of the time, the rule is OVERCONFIDENT
    and needs recalibration.

Calibration methods used:
    1. Empirical Accuracy  -- how often each rule fires correctly
    2. Expected Calibration Error (ECE) -- gap between confidence and accuracy
    3. Reliability Diagram  -- written to calibration_report.txt

Usage:
    python Data/LLM_judge1/calibration.py --disprot Data/DisProt_ProteinData.json
    python Data/LLM_judge1/calibration.py --demo
"""

import json
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict


# =============================================================
# 1. LOAD DISPROT
# =============================================================

def load_disprot(filepath: str) -> list:
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])
    print(f"[INFO] {len(raw)} DisProt proteins loaded\n")
    return raw


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
# 2. BUILD TEST CASES FROM DISPROT
#    Each protein becomes a test case with known ground truth
# =============================================================

def build_test_cases(proteins: list, stats: dict, max_cases: int = 500) -> list:
    """
    Convert DisProt proteins into test cases for rule calibration.

    Each test case has:
      - context   : the feature values (disorder score, pLDDT, etc.)
      - ground_truth : dict of what we KNOW is true from DisProt

    Since DisProt only has disorder annotations (not pLDDT), pLDDT
    is simulated as correlated with disorder score for calibration.
    """
    cases = []
    sample = proteins[:max_cases] if len(proteins) > max_cases else proteins

    for p in sample:
        dc  = p.get("disorder_content_pure") or p.get("disorder_content_obs") or 0.0
        seq = p.get("sequence", "")
        pro = seq.count("P") / len(seq) if seq else 0.0
        gly = seq.count("G") / len(seq) if seq else 0.0
        regions  = [r for r in p.get("regions", []) if isinstance(r, dict)]
        has_pfam = len(p.get("features", {}).get("pfam", [])) > 0

        # Simulate pLDDT as inversely correlated with disorder score
        # pLDDT = 100 * (1 - disorder_score) + small noise
        simulated_plddt = max(0, min(100, (1 - dc) * 100 + random.gauss(0, 5)))

        context = {
            "disorder_score":   dc,
            "plddt_score":      simulated_plddt,
            "region_length":    regions[0].get("end", 0) - regions[0].get("start", 0) + 1
                                if regions else 0,
            "proline_fraction": pro,
            "glycine_fraction": gly,
            "has_pfam_domain":  has_pfam,
        }

        # Ground truth labels derived from DisProt annotations
        ground_truth = {
            "is_disordered":       dc > 0.3,
            "high_disorder":       dc > 0.7,
            "gray_zone":           0.3 <= dc <= 0.5,
            "has_short_region":    any(
                (r.get("end", 0) - r.get("start", 0) + 1) < 10 for r in regions
            ),
            "pro_enriched":        pro > stats["mean_proline"],
            "gly_enriched":        gly > stats["mean_glycine"],
            "has_pfam":            has_pfam,
            "plddt_low":           simulated_plddt < 50,
            "plddt_moderate":      50 <= simulated_plddt < 70,
            "plddt_high":          simulated_plddt >= 70,
        }

        cases.append({"context": context, "ground_truth": ground_truth})

    return cases


# =============================================================
# 3. RULE DEFINITIONS FOR CALIBRATION
#    Each rule has: condition, correct_if, assigned_confidence
# =============================================================

def get_calibration_rules(stats: dict) -> list:
    """
    Define each rule as a testable unit with:
      - condition    : when the rule fires (same as SymbolicRules.py)
      - correct_if   : what ground truth label means the rule was right
      - confidence   : the confidence score currently assigned to the rule
    """
    return [
        {
            "rule_id":    "DR-001",
            "name":       "0.5 Cutoff Reliability",
            "category":   "Disorder Threshold",
            "condition":  lambda ctx: ctx.get("disorder_score") is not None,
            "correct_if": lambda ctx, gt: (
                (ctx["disorder_score"] > 0.5 and gt["is_disordered"]) or
                (ctx["disorder_score"] <= 0.5 and not gt["high_disorder"])
            ),
            "confidence": 0.85,
        },
        {
            "rule_id":    "DR-002",
            "name":       "Gray Zone Detection",
            "category":   "Disorder Threshold",
            "condition":  lambda ctx: 0.3 <= ctx.get("disorder_score", -1) <= 0.5,
            "correct_if": lambda ctx, gt: gt["gray_zone"],
            "confidence": 0.75,
        },
        {
            "rule_id":    "DR-003",
            "name":       "High Confidence Disorder",
            "category":   "Disorder Threshold",
            "condition":  lambda ctx: ctx.get("disorder_score", 0) > 0.7,
            "correct_if": lambda ctx, gt: gt["high_disorder"],
            "confidence": 0.95,
        },
        {
            "rule_id":    "SC-001",
            "name":       "Proline Enrichment",
            "category":   "Sequence Composition",
            "condition":  lambda ctx: ctx.get("proline_fraction", 0) > stats["mean_proline"] * 1.5,
            "correct_if": lambda ctx, gt: gt["pro_enriched"] and gt["is_disordered"],
            "confidence": 0.82,
        },
        {
            "rule_id":    "SC-002",
            "name":       "Glycine Enrichment",
            "category":   "Sequence Composition",
            "condition":  lambda ctx: ctx.get("glycine_fraction", 0) > stats["mean_glycine"] * 1.5,
            "correct_if": lambda ctx, gt: gt["gly_enriched"] and gt["is_disordered"],
            "confidence": 0.80,
        },
        {
            "rule_id":    "SC-003",
            "name":       "Combined Pro-Gly Signal",
            "category":   "Sequence Composition",
            "condition":  lambda ctx: (
                ctx.get("proline_fraction", 0) > stats["mean_proline"] and
                ctx.get("glycine_fraction", 0) > stats["mean_glycine"]
            ),
            "correct_if": lambda ctx, gt: gt["pro_enriched"] and gt["gly_enriched"],
            "confidence": 0.88,
        },
        {
            "rule_id":    "RL-001",
            "name":       "Short IDR Warning",
            "category":   "Region Length",
            "condition":  lambda ctx: 0 < ctx.get("region_length", 999) < 10,
            "correct_if": lambda ctx, gt: gt["has_short_region"],
            "confidence": 0.78,
        },
        {
            "rule_id":    "RL-002",
            "name":       "Typical IDR Length",
            "category":   "Region Length",
            "condition":  lambda ctx: ctx.get("region_length", 0) >= 10,
            "correct_if": lambda ctx, gt: not gt["has_short_region"],
            "confidence": 0.85,
        },
        {
            "rule_id":    "AF-001",
            "name":       "Very Low pLDDT -- High Disorder",
            "category":   "AlphaFold pLDDT",
            "condition":  lambda ctx: ctx.get("plddt_score", 100) < 50,
            "correct_if": lambda ctx, gt: gt["plddt_low"] and gt["is_disordered"],
            "confidence": 0.92,
        },
        {
            "rule_id":    "AF-002",
            "name":       "Moderate pLDDT -- Ambiguous",
            "category":   "AlphaFold pLDDT",
            "condition":  lambda ctx: 50 <= ctx.get("plddt_score", 100) < 70,
            "correct_if": lambda ctx, gt: gt["plddt_moderate"],
            "confidence": 0.72,
        },
        {
            "rule_id":    "AF-003",
            "name":       "High pLDDT -- Structured",
            "category":   "AlphaFold pLDDT",
            "condition":  lambda ctx: ctx.get("plddt_score", 0) >= 70,
            "correct_if": lambda ctx, gt: gt["plddt_high"] and not gt["is_disordered"],
            "confidence": 0.90,
        },
        {
            "rule_id":    "SD-001",
            "name":       "Pfam Domain Co-occurrence",
            "category":   "Structural Domain",
            "condition":  lambda ctx: ctx.get("has_pfam_domain") is True,
            "correct_if": lambda ctx, gt: gt["has_pfam"],
            "confidence": 0.87,
        },
        {
            "rule_id":    "SD-002",
            "name":       "No Pfam Domain -- Likely Full IDR",
            "category":   "Structural Domain",
            "condition":  lambda ctx: ctx.get("has_pfam_domain") is False,
            "correct_if": lambda ctx, gt: not gt["has_pfam"] and gt["is_disordered"],
            "confidence": 0.80,
        },
    ]


# =============================================================
# 4. CALIBRATION ENGINE
# =============================================================

def calibrate(rules: list, test_cases: list) -> list:
    """
    For each rule, compute:
      - fired_count     : how many times the rule triggered
      - correct_count   : how many times it was right
      - empirical_acc   : correct / fired (actual accuracy)
      - assigned_conf   : the confidence we originally claimed
      - calibration_gap : assigned_conf - empirical_acc
                          positive = overconfident
                          negative = underconfident
      - calibrated_conf : adjusted confidence based on empirical data
    """
    results = []

    for rule in rules:
        fired   = 0
        correct = 0

        for case in test_cases:
            ctx = case["context"]
            gt  = case["ground_truth"]

            try:
                if rule["condition"](ctx):
                    fired += 1
                    if rule["correct_if"](ctx, gt):
                        correct += 1
            except Exception:
                continue

        empirical_acc   = correct / fired if fired > 0 else None
        assigned_conf   = rule["confidence"]
        cal_gap         = (assigned_conf - empirical_acc) if empirical_acc is not None else None

        # Calibrated confidence: blend assigned and empirical (70/30 weight)
        if empirical_acc is not None:
            calibrated = round(0.3 * assigned_conf + 0.7 * empirical_acc, 3)
        else:
            calibrated = assigned_conf

        results.append({
            "rule_id":        rule["rule_id"],
            "name":           rule["name"],
            "category":       rule["category"],
            "fired_count":    fired,
            "correct_count":  correct,
            "empirical_acc":  empirical_acc,
            "assigned_conf":  assigned_conf,
            "calibration_gap": cal_gap,
            "calibrated_conf": calibrated,
            "status": (
                "WELL CALIBRATED" if cal_gap is not None and abs(cal_gap) < 0.05 else
                "OVERCONFIDENT"   if cal_gap is not None and cal_gap > 0.05 else
                "UNDERCONFIDENT"  if cal_gap is not None and cal_gap < -0.05 else
                "NOT FIRED"
            )
        })

    return results


def expected_calibration_error(results: list) -> float:
    """
    ECE: weighted average of |assigned_confidence - empirical_accuracy|
    across all rules that fired. Lower is better.
    """
    total_fired = sum(r["fired_count"] for r in results if r["fired_count"] > 0)
    if total_fired == 0:
        return 0.0
    ece = sum(
        r["fired_count"] * abs(r["calibration_gap"])
        for r in results
        if r["calibration_gap"] is not None
    ) / total_fired
    return round(ece, 4)


# =============================================================
# 5. SAVE CALIBRATION REPORT
# =============================================================

def save_report(results: list, stats: dict, output_path: str):
    ece    = expected_calibration_error(results)
    lines  = []

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- Symbolic Rule Calibration Report")
    lines.append(f"  DisProt proteins tested: {stats['total_proteins']:,}")
    lines.append(f"  Rules evaluated: {len(results)}")
    lines.append(f"  Expected Calibration Error (ECE): {ece:.4f}")
    lines.append("  (ECE closer to 0.0 = better calibrated)")
    lines.append("=" * 70)
    lines.append("")

    lines.append("WHAT IS CALIBRATION?")
    lines.append("-" * 70)
    lines.append("  Calibration checks if a rule's confidence score matches")
    lines.append("  how often it is actually correct.")
    lines.append("  Example: A rule claiming 85% confidence should be right")
    lines.append("  about 85% of the time. If it is only right 60% of the")
    lines.append("  time, it is OVERCONFIDENT and needs adjustment.")
    lines.append("")

    # Group by category
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    for category, cat_rules in categories.items():
        lines.append(f"CATEGORY: {category}")
        lines.append("-" * 70)

        for r in cat_rules:
            lines.append(f"  [{r['rule_id']}] {r['name']}")
            lines.append(f"  Status           : {r['status']}")
            lines.append(f"  Times fired      : {r['fired_count']}")
            lines.append(f"  Times correct    : {r['correct_count']}")

            if r["empirical_acc"] is not None:
                lines.append(f"  Empirical accuracy : {r['empirical_acc']:.1%}")
                lines.append(f"  Assigned confidence: {r['assigned_conf']:.1%}")
                lines.append(f"  Calibration gap    : {r['calibration_gap']:+.1%} "
                              f"({'overconfident' if r['calibration_gap'] > 0 else 'underconfident'})")
                lines.append(f"  Calibrated confidence (adjusted): {r['calibrated_conf']:.1%}")

                # Simple reliability bar
                emp_bar  = "#" * int(r["empirical_acc"]  * 20) + "." * (20 - int(r["empirical_acc"]  * 20))
                conf_bar = "#" * int(r["assigned_conf"]  * 20) + "." * (20 - int(r["assigned_conf"]  * 20))
                lines.append(f"  Empirical   [{emp_bar}] {r['empirical_acc']:.0%}")
                lines.append(f"  Assigned    [{conf_bar}] {r['assigned_conf']:.0%}")
            else:
                lines.append(f"  Result: Rule never fired on this dataset")

            lines.append("")

        lines.append("")

    # Summary table
    lines.append("CALIBRATION SUMMARY TABLE")
    lines.append("-" * 70)
    lines.append(f"  {'Rule ID':<10} {'Assigned':>10} {'Empirical':>10} {'Gap':>8} {'Status':<20}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*20}")
    for r in results:
        emp = f"{r['empirical_acc']:.1%}" if r["empirical_acc"] is not None else "N/A"
        gap = f"{r['calibration_gap']:+.1%}" if r["calibration_gap"] is not None else "N/A"
        lines.append(
            f"  {r['rule_id']:<10} {r['assigned_conf']:>9.1%} {emp:>10} {gap:>8}  {r['status']:<20}"
        )

    lines.append("")
    lines.append(f"  Overall ECE: {ece:.4f} (target < 0.05 for well-calibrated system)")
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)
    overconfident  = [r for r in results if r["status"] == "OVERCONFIDENT"]
    underconfident = [r for r in results if r["status"] == "UNDERCONFIDENT"]
    well_calibrated = [r for r in results if r["status"] == "WELL CALIBRATED"]

    lines.append(f"  Well calibrated : {len(well_calibrated)} rules -- no action needed")
    lines.append(f"  Overconfident   : {len(overconfident)} rules -- lower confidence scores")
    lines.append(f"  Underconfident  : {len(underconfident)} rules -- raise confidence scores")

    if overconfident:
        lines.append("")
        lines.append("  Rules to lower confidence:")
        for r in overconfident:
            lines.append(f"    [{r['rule_id']}] {r['name']}: "
                         f"{r['assigned_conf']:.0%} -> {r['calibrated_conf']:.0%}")

    if underconfident:
        lines.append("")
        lines.append("  Rules to raise confidence:")
        for r in underconfident:
            lines.append(f"    [{r['rule_id']}] {r['name']}: "
                         f"{r['assigned_conf']:.0%} -> {r['calibrated_conf']:.0%}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    # Add calibrated scores section
    lines.append("")
    lines.append("CALIBRATED RULE CONFIDENCE SCORES")
    lines.append("(Use these updated values in SymbolicRules.py)")
    lines.append("-" * 70)
    for r in results:
        if r["empirical_acc"] is not None:
            note = "  (no change needed)" if r["status"] == "WELL CALIBRATED" else ""
            lines.append(
                f"  [{r['rule_id']}] {r['name']:<35} "
                f"{r['assigned_conf']:.0%} -> {r['calibrated_conf']:.0%}{note}"
            )
        else:
            lines.append(
                f"  [{r['rule_id']}] {r['name']:<35} "
                f"{r['assigned_conf']:.0%} -> NOT FIRED (keep assigned value)"
            )

    lines.append("")
    lines.append("HOW TO APPLY CALIBRATION")
    lines.append("-" * 70)
    lines.append("  1. Open Data/LLM_judge1/SymbolicRules.py")
    lines.append("  2. Find each rule by its Rule ID (e.g. DR-001)")
    lines.append("  3. Replace confidence value with the calibrated value above")
    lines.append("  4. Re-run the pipeline -- calibrated rules produce more")
    lines.append("     reliable inferences because confidence matches real accuracy")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)

    # Write directly to same folder as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, "calibration_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n[SAVED] Calibration report written to: {out_path}\n")


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
] * 50   # repeat to get enough test cases


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate symbolic rules against DisProt ground truth"
    )
    parser.add_argument("--disprot", type=str, help="Path to DisProt JSON")
    parser.add_argument("--output",  type=str,
                        default="Data/LLM_judge1/calibration_report.txt",
                        help="Output path for calibration report")
    parser.add_argument("--demo",    action="store_true", help="Run with built-in sample data")
    args = parser.parse_args()

    random.seed(42)   # reproducible pLDDT simulation

    if args.demo or not args.disprot:
        print("[INFO] Running in DEMO mode\n")
        proteins = DEMO_PROTEINS
    else:
        proteins = load_disprot(args.disprot)

    stats      = compute_stats(proteins)
    rules      = get_calibration_rules(stats)
    test_cases = build_test_cases(proteins, stats)
    results    = calibrate(rules, test_cases)

    print(f"[INFO] {len(test_cases)} test cases built from DisProt\n")

    save_report(results, stats, args.output)


if __name__ == "__main__":
    main()