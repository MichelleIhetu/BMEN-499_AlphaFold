"""
BMEN-499 AlphaFold — Symbolic Rules for Neuro-Symbolic RAG System
------------------------------------------------------------------
Purpose:
    Defines a symbolic reasoning layer that sits on top of BioGPT's
    neural retrieval. Each rule encodes factual domain knowledge derived
    from DisProt statistics and protein disorder biology.

How neuro-symbolic RAG works here:
    1. RETRIEVE  — BioGPT (neural) retrieves relevant context
    2. REASON    — Symbolic rules (this file) validate, filter, and
                   augment the neural answer with hard biological constraints
    3. GENERATE  — Final answer combines neural fluency + symbolic accuracy

Rule structure:
    Each rule has:
      - condition  : a function that checks if the rule applies
      - inference  : what conclusion to draw if the condition is met
      - confidence : how certain the rule is (0.0 - 1.0)
      - source     : where the rule comes from (DisProt stats, literature)

Usage:
    python symbolic_rules.py --disprot Data/DisProt_ProteinData.json --demo
"""

import json
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable


# =============================================================
# 1. RULE DATA STRUCTURE
# =============================================================

@dataclass
class SymbolicRule:
    """
    A single symbolic rule in the reasoning engine.

    Attributes:
        rule_id    : unique identifier (e.g. DR-001)
        category   : topic area (Disorder Threshold, Composition, etc.)
        name       : short human-readable name
        condition  : function(context: dict) -> bool
                     Returns True if this rule should fire
        inference  : function(context: dict) -> str
                     Returns the conclusion when the rule fires
        confidence : float 0-1, how reliable this rule is
        source     : evidence basis for the rule
        keywords   : question keywords that trigger retrieval of this rule
    """
    rule_id:    str
    category:   str
    name:       str
    condition:  Callable
    inference:  Callable
    confidence: float
    source:     str
    keywords:   list[str] = field(default_factory=list)


# =============================================================
# 2. SYMBOLIC RULE DEFINITIONS
#    Derived from:
#      - DisProt database statistics (13,396 proteins)
#      - Established protein disorder biology
#      - AlphaFold pLDDT literature
# =============================================================

def build_rules(stats: dict) -> list[SymbolicRule]:
    """
    Build the full symbolic rule base using dataset statistics.
    Stats are injected at runtime so rules reflect your actual DisProt data.
    """

    rules = [

        # ── CATEGORY 1: Disorder Threshold Rules ─────────────────────────

        SymbolicRule(
            rule_id    = "DR-001",
            category   = "Disorder Threshold",
            name       = "0.5 Cutoff Reliability",
            condition  = lambda ctx: ctx.get("disorder_score") is not None,
            inference  = lambda ctx: (
                f"FIRE: Disorder score {ctx['disorder_score']:.3f} "
                + ("EXCEEDS" if ctx['disorder_score'] > 0.5 else "FALLS BELOW")
                + f" the 0.5 threshold. "
                + (f"Region is classified as DISORDERED. "
                   f"Note: only {stats['pct_above_0.5']:.1f}% of DisProt proteins "
                   f"exceed 0.5 — this is a conservative cutoff."
                   if ctx['disorder_score'] > 0.5
                   else
                   f"Region may still be disordered — {stats['pct_above_0.3']:.1f}% "
                   f"of DisProt proteins exceed 0.3. Consider lowering threshold.")
            ),
            confidence = 0.85,
            source     = f"DisProt: {stats['total_proteins']:,} proteins, "
                         f"mean disorder={stats['mean_disorder']:.3f}",
            keywords   = ["disorder score", "cutoff", "threshold", "0.5"]
        ),

        SymbolicRule(
            rule_id    = "DR-002",
            category   = "Disorder Threshold",
            name       = "Gray Zone Detection",
            condition  = lambda ctx: (
                ctx.get("disorder_score") is not None and
                0.3 <= ctx["disorder_score"] <= 0.5
            ),
            inference  = lambda ctx: (
                f"FIRE: Score {ctx['disorder_score']:.3f} falls in the GRAY ZONE (0.3-0.5). "
                f"This region is ambiguously disordered. "
                f"Recommend secondary validation with sequence composition analysis "
                f"(check for elevated Pro/Gly content) before classifying."
            ),
            confidence = 0.75,
            source     = "DisProt distribution analysis — mid-range IDR zone",
            keywords   = ["gray zone", "ambiguous", "borderline", "0.3", "0.4"]
        ),

        SymbolicRule(
            rule_id    = "DR-003",
            category   = "Disorder Threshold",
            name       = "High Confidence Disorder",
            condition  = lambda ctx: ctx.get("disorder_score", 0) > 0.7,
            inference  = lambda ctx: (
                f"FIRE: Score {ctx['disorder_score']:.3f} indicates HIGH CONFIDENCE disorder. "
                f"Region is almost certainly an IDR. "
                f"AlphaFold pLDDT for this region is expected to be below 50."
            ),
            confidence = 0.95,
            source     = "DisProt + AlphaFold pLDDT correlation literature",
            keywords   = ["high disorder", "confident", "pLDDT", "alphafold"]
        ),


        # ── CATEGORY 2: Sequence Composition Rules ────────────────────────

        SymbolicRule(
            rule_id    = "SC-001",
            category   = "Sequence Composition",
            name       = "Proline Enrichment",
            condition  = lambda ctx: ctx.get("proline_fraction", 0) > stats["mean_proline"] * 1.5,
            inference  = lambda ctx: (
                f"FIRE: Proline fraction {ctx['proline_fraction']*100:.1f}% exceeds "
                f"1.5x the DisProt mean ({stats['mean_proline']*100:.1f}%). "
                f"Elevated proline content strongly predicts intrinsic disorder. "
                f"Proline's rigid pyrrolidine ring disrupts alpha-helices and beta-sheets."
            ),
            confidence = 0.82,
            source     = f"DisProt mean proline={stats['mean_proline']*100:.1f}%",
            keywords   = ["proline", "Pro-rich", "composition"]
        ),

        SymbolicRule(
            rule_id    = "SC-002",
            category   = "Sequence Composition",
            name       = "Glycine Enrichment",
            condition  = lambda ctx: ctx.get("glycine_fraction", 0) > stats["mean_glycine"] * 1.5,
            inference  = lambda ctx: (
                f"FIRE: Glycine fraction {ctx['glycine_fraction']*100:.1f}% exceeds "
                f"1.5x the DisProt mean ({stats['mean_glycine']*100:.1f}%). "
                f"Elevated glycine content increases backbone conformational entropy, "
                f"a hallmark of intrinsically disordered regions."
            ),
            confidence = 0.80,
            source     = f"DisProt mean glycine={stats['mean_glycine']*100:.1f}%",
            keywords   = ["glycine", "Gly-rich", "flexible backbone"]
        ),

        SymbolicRule(
            rule_id    = "SC-003",
            category   = "Sequence Composition",
            name       = "Combined Pro-Gly Disorder Signal",
            condition  = lambda ctx: (
                ctx.get("proline_fraction", 0) > stats["mean_proline"] and
                ctx.get("glycine_fraction", 0) > stats["mean_glycine"]
            ),
            inference  = lambda ctx: (
                f"FIRE: Both proline ({ctx['proline_fraction']*100:.1f}%) AND "
                f"glycine ({ctx['glycine_fraction']*100:.1f}%) exceed dataset means. "
                f"Combined Pro+Gly enrichment is a strong composite disorder signal. "
                f"Confidence in IDR classification is elevated."
            ),
            confidence = 0.88,
            source     = "DisProt composition analysis — composite disorder signal",
            keywords   = ["proline", "glycine", "Pro-Gly", "disordered"]
        ),


        # ── CATEGORY 3: Region Length Rules ──────────────────────────────

        SymbolicRule(
            rule_id    = "RL-001",
            category   = "Region Length",
            name       = "Short IDR Warning",
            condition  = lambda ctx: 0 < ctx.get("region_length", 999) < 10,
            inference  = lambda ctx: (
                f"FIRE: Region length {ctx['region_length']} aa is BELOW 10 residues. "
                f"Short IDRs are difficult to predict reliably. "
                f"Only {stats['pct_short_regions']:.1f}% of DisProt regions are this short. "
                f"Treat prediction confidence as LOW — consider experimental validation."
            ),
            confidence = 0.78,
            source     = f"DisProt: {stats['pct_short_regions']:.1f}% of regions < 10 aa",
            keywords   = ["short IDR", "short region", "< 10 residues", "length"]
        ),

        SymbolicRule(
            rule_id    = "RL-002",
            category   = "Region Length",
            name       = "Typical IDR Length",
            condition  = lambda ctx: ctx.get("region_length", 0) >= 10,
            inference  = lambda ctx: (
                f"FIRE: Region length {ctx['region_length']} aa meets minimum length threshold. "
                f"DisProt mean region length = {stats['mean_region_length']:.1f} aa. "
                f"Prediction confidence is STANDARD for this region size."
            ),
            confidence = 0.85,
            source     = f"DisProt mean region length={stats['mean_region_length']:.1f} aa",
            keywords   = ["region length", "IDR length", "residues"]
        ),


        # ── CATEGORY 4: AlphaFold pLDDT Rules ────────────────────────────

        SymbolicRule(
            rule_id    = "AF-001",
            category   = "AlphaFold pLDDT",
            name       = "Very Low pLDDT — High Disorder",
            condition  = lambda ctx: ctx.get("plddt_score", 100) < 50,
            inference  = lambda ctx: (
                f"FIRE: pLDDT {ctx['plddt_score']} < 50. "
                f"AlphaFold has VERY LOW confidence in this region's structure. "
                f"This is strong computational evidence of intrinsic disorder. "
                f"Cross-reference with DisProt annotation if available."
            ),
            confidence = 0.92,
            source     = "AlphaFold pLDDT < 50 = disordered (Jumper et al. 2021)",
            keywords   = ["pLDDT", "alphafold", "confidence", "low confidence"]
        ),

        SymbolicRule(
            rule_id    = "AF-002",
            category   = "AlphaFold pLDDT",
            name       = "Moderate pLDDT — Ambiguous",
            condition  = lambda ctx: 50 <= ctx.get("plddt_score", 100) < 70,
            inference  = lambda ctx: (
                f"FIRE: pLDDT {ctx['plddt_score']} is in the MODERATE range (50-70). "
                f"Structure prediction is uncertain. Region may be conditionally disordered "
                f"(disordered alone, structured when bound to a partner). "
                f"Check for molecular recognition features (MoRFs)."
            ),
            confidence = 0.72,
            source     = "AlphaFold pLDDT 50-70 = low confidence region",
            keywords   = ["pLDDT", "moderate", "MoRF", "conditional disorder"]
        ),

        SymbolicRule(
            rule_id    = "AF-003",
            category   = "AlphaFold pLDDT",
            name       = "High pLDDT — Structured",
            condition  = lambda ctx: ctx.get("plddt_score", 0) >= 70,
            inference  = lambda ctx: (
                f"FIRE: pLDDT {ctx['plddt_score']} >= 70. "
                f"AlphaFold is CONFIDENT in this region's structure. "
                f"Region is likely NOT intrinsically disordered. "
                f"If disorder annotations exist here, investigate further."
            ),
            confidence = 0.90,
            source     = "AlphaFold pLDDT >= 70 = confident prediction",
            keywords   = ["pLDDT", "structured", "high confidence", "folded"]
        ),


        # ── CATEGORY 5: Structural Domain Rules ──────────────────────────

        SymbolicRule(
            rule_id    = "SD-001",
            category   = "Structural Domain",
            name       = "Pfam Domain Co-occurrence",
            condition  = lambda ctx: ctx.get("has_pfam_domain") is True,
            inference  = lambda ctx: (
                f"FIRE: Protein contains a Pfam domain. "
                f"{stats['pct_with_pfam']:.1f}% of DisProt proteins have Pfam domains, "
                f"confirming that structured domains and IDRs frequently co-occur. "
                f"Classify as a MIXED protein: partially structured, partially disordered. "
                f"Analyze each region independently."
            ),
            confidence = 0.87,
            source     = f"DisProt: {stats['pct_with_pfam']:.1f}% proteins have Pfam domains",
            keywords   = ["pfam", "domain", "structured domain", "mixed protein"]
        ),

        SymbolicRule(
            rule_id    = "SD-002",
            category   = "Structural Domain",
            name       = "No Pfam Domain — Likely Full IDR",
            condition  = lambda ctx: ctx.get("has_pfam_domain") is False,
            inference  = lambda ctx: (
                f"FIRE: No Pfam domain detected. "
                f"Protein may be a fully disordered protein (FDP). "
                f"Evaluate whole-sequence disorder content — if mean disorder > 0.5, "
                f"classify as intrinsically disordered protein (IDP)."
            ),
            confidence = 0.80,
            source     = "DisProt IDP classification criteria",
            keywords   = ["no domain", "fully disordered", "IDP", "no pfam"]
        ),

    ]

    return rules


# =============================================================
# 3. RULE ENGINE  — applies rules to a query context
# =============================================================

class SymbolicRuleEngine:
    """
    The reasoning engine for the neuro-symbolic RAG system.

    In the full pipeline:
      - BioGPT retrieves and generates a neural answer
      - This engine fires applicable rules on the same context
      - The fired rules' inferences are appended as symbolic constraints
        to ground and correct the neural answer
    """

    def __init__(self, rules: list[SymbolicRule]):
        self.rules = rules

    def query(self, context: dict, question: str = "") -> list[dict]:
        """
        Given a context dict and optional question string,
        fire all applicable rules and return their inferences.

        Context keys (any subset):
            disorder_score   : float 0-1
            plddt_score      : float 0-100
            region_length    : int (amino acids)
            proline_fraction : float 0-1
            glycine_fraction : float 0-1
            has_pfam_domain  : bool
        """
        fired = []
        q     = question.lower()

        for rule in self.rules:
            # Check if rule condition is met
            try:
                if rule.condition(context):
                    inference = rule.inference(context)
                    fired.append({
                        "rule_id":    rule.rule_id,
                        "category":   rule.category,
                        "name":       rule.name,
                        "inference":  inference,
                        "confidence": rule.confidence,
                        "source":     rule.source,
                    })
            except Exception:
                continue

        # Sort by confidence descending
        fired.sort(key=lambda x: x["confidence"], reverse=True)
        return fired

    def retrieve_by_question(self, question: str) -> list[SymbolicRule]:
        """
        Retrieve rules relevant to a natural language question
        based on keyword matching — the symbolic retrieval step.
        """
        q       = question.lower()
        matches = []
        for rule in self.rules:
            if any(kw in q for kw in rule.keywords):
                matches.append(rule)
        return matches

    def print_fired_rules(self, fired: list[dict]):
        if not fired:
            print("  No rules fired for this context.")
            return
        for r in fired:
            conf_bar = "█" * int(r["confidence"] * 10) + "░" * (10 - int(r["confidence"] * 10))
            print(f"  [{r['rule_id']}] {r['name']}")
            print(f"  Confidence : {conf_bar} {r['confidence']:.0%}")
            print(f"  Category   : {r['category']}")
            print(f"  Inference  : {r['inference']}")
            print(f"  Source     : {r['source']}")
            print()


# =============================================================
# 4. LOAD DISPROT + COMPUTE STATS
# =============================================================

def load_disprot(filepath: str) -> list:
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] DisProt file not found: {filepath}")
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
# 5. DEMO — show rules firing on example protein contexts
# =============================================================

DEMO_CONTEXTS = [
    {
        "label":            "Alpha-synuclein C-terminal IDR",
        "question":         "Is this region intrinsically disordered?",
        "context": {
            "disorder_score":   0.72,
            "plddt_score":      38,
            "region_length":    45,
            "proline_fraction": 0.02,
            "glycine_fraction": 0.09,
            "has_pfam_domain":  False,
        }
    },
    {
        "label":            "Short ambiguous loop region",
        "question":         "Should I classify this short region as disordered?",
        "context": {
            "disorder_score":   0.41,
            "plddt_score":      62,
            "region_length":    7,
            "proline_fraction": 0.05,
            "glycine_fraction": 0.06,
            "has_pfam_domain":  True,
        }
    },
    {
        "label":            "Pro-Gly rich disordered linker",
        "question":         "Do proline and glycine content predict disorder here?",
        "context": {
            "disorder_score":   0.65,
            "plddt_score":      45,
            "region_length":    32,
            "proline_fraction": 0.18,
            "glycine_fraction": 0.15,
            "has_pfam_domain":  False,
        }
    },
]

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


def run_demo(stats: dict):
    rules  = build_rules(stats)
    engine = SymbolicRuleEngine(rules)

    print("=" * 70)
    print("  SYMBOLIC RULE ENGINE — Neuro-Symbolic RAG Demo")
    print(f"  {len(rules)} rules loaded | {stats['total_proteins']:,} DisProt proteins")
    print("=" * 70)

    # Show all rules
    print("\n── RULE REGISTRY ──────────────────────────────────────────────────\n")
    for rule in rules:
        print(f"  [{rule.rule_id}] {rule.name}")
        print(f"  Category   : {rule.category}")
        print(f"  Confidence : {rule.confidence:.0%}")
        print(f"  Keywords   : {', '.join(rule.keywords)}")
        print(f"  Source     : {rule.source}")
        print()

    # Fire rules on demo contexts
    print("\n── RULE FIRING ON EXAMPLE PROTEINS ────────────────────────────────\n")
    for demo in DEMO_CONTEXTS:
        print(f"  PROTEIN CONTEXT: {demo['label']}")
        print(f"  Question       : {demo['question']}")
        print(f"  Context values : {demo['context']}")
        print()
        fired = engine.query(demo["context"], demo["question"])
        print(f"  Rules fired ({len(fired)}):")
        engine.print_fired_rules(fired)
        print("-" * 70 + "\n")


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Symbolic rule engine for neuro-symbolic RAG — BMEN-499"
    )
    parser.add_argument("--disprot", type=str, help="Path to DisProt JSON")
    parser.add_argument("--demo",    action="store_true", help="Run with built-in sample data")
    args = parser.parse_args()

    if args.demo or not args.disprot:
        print("[INFO] Running in DEMO mode\n")
        stats = compute_stats(DEMO_PROTEINS)
    else:
        proteins = load_disprot(args.disprot)
        stats    = compute_stats(proteins)

    run_demo(stats)


if __name__ == "__main__":
    main()