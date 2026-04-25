"""
BMEN-499 AlphaFold -- LLM Judge 3: FACT Score Analysis
--------------------------------------------------------
File    : FACT_score3.py
Output  : FACT_score3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Computes a FACT score for each predicted answer -- a measure of
    factual accuracy and grounding against the DisProt knowledge base.

    FACT score is composed of four sub-scores:

        F  -- Fidelity (0-1)
               Are the facts in the answer supported by the DisProt KB?
               Measures: known DisProt statistics present and correct.

        A  -- Accuracy (0-1)
               Are numerical claims (thresholds, percentages, counts)
               consistent with ground-truth DisProt values?

        C  -- Completeness (0-1)
               Does the answer cover the key factual claims needed to
               fully answer the question?

        T  -- Traceability (0-1)
               Can every major claim in the answer be traced back to
               a specific KB passage (KB-001 through KB-014)?

    Final FACT score = (F + A + C + T) / 4  (range 0.0 - 1.0)

    Score interpretation:
        >= 0.85  : Highly Factual     -- all claims grounded and accurate
        >= 0.70  : Factual            -- most claims grounded
        >= 0.55  : Partially Factual  -- some unsupported claims
        >= 0.40  : Weakly Factual     -- limited grounding
        <  0.40  : Unfactual          -- claims lack KB support

Output: FACT_score3_output.txt
"""

import os
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "FACT_score3_output.txt")

# ── Ground truth DisProt constants ────────────────────────────────────
GT = {
    "n_proteins":    13396,
    "mean_disorder": 0.378,
    "pct_above_05":  29.1,
    "pct_above_03":  44.2,
    "mean_proline":  6.0,
    "mean_glycine":  7.0,
    "pct_pfam":      92.6,
}
NUMERIC_TOL = 0.05   # 5% relative tolerance for numeric checks

# ── KB passage fingerprints (key facts per passage) ───────────────────
KB_FACTS = {
    "KB-001": ["0.5 threshold", "13,396", "29.1", "44.2", "0.378", "missed"],
    "KB-002": ["gray zone", "0.3", "0.5", "ambiguous", "secondary validation"],
    "KB-003": ["0.7", "high confidence", "plddt below 50", "experimentally validated"],
    "KB-004": ["proline", "6.0", "pyrrolidine", "alpha-heli", "beta-sheet"],
    "KB-005": ["glycine", "7.0", "conformational freedom", "weaker", "combination"],
    "KB-006": ["proline", "glycine", "composite", "backbone kink", "conformational"],
    "KB-007": ["shorter than 10", "short idrs", "sequence context", "unreliable"],
    "KB-008": ["sliding window", "smooths", "short idrs", "risk", "noise"],
    "KB-009": ["plddt below 50", "most reliable", "13,396", "computational signal"],
    "KB-010": ["50", "70", "morf", "molecular recognition", "conditionally disordered"],
    "KB-011": ["plddt.*70", "ordered", "experimental.*precedence", "confident"],
    "KB-012": ["92.6", "pfam", "co-occur", "independently"],
    "KB-013": ["no pfam", "idp", "signaling", "transcription", "hub protein"],
    "KB-014": ["13,396", "0.378", "29.1", "summary", "experimentally validated"],
}

# ── Fidelity signal groups (F sub-score) ─────────────────────────────
FIDELITY_SIGNALS = [
    # Each tuple: (signal_phrase, weight)
    (r"plddt.*below.*50",              0.15),
    (r"disprot.*confirm",              0.10),
    (r"experimentally.*validated",     0.10),
    (r"pyrrolidine.*ring",             0.10),
    (r"conformational.*freedom",       0.08),
    (r"gray.*zone|ambiguous.*zone",    0.08),
    (r"molecular.*recognition|morf",   0.08),
    (r"pfam.*domain",                  0.07),
    (r"sliding.*window",               0.07),
    (r"intrinsically.*disordered",     0.07),
    (r"experimentally.*annotations.*precedence", 0.05),
    (r"secondary.*structure",          0.05),
    (r"sequence.*context",             0.05),
    (r"hub.*protein|signaling.*network", 0.05),
]

# ── Accuracy checks (A sub-score) ─────────────────────────────────────

def check_numeric_accuracy(answer):
    """
    Check that every number in the answer that matches a GT constant
    is within tolerance. Returns (correct_count, total_checked).
    """
    a         = answer.lower()
    correct   = 0
    total     = 0

    def safe_float(s):
        """Convert string to float, return None if invalid or bare period."""
        if not s:
            return None
        s = s.strip().replace(",", "")
        if s in (".", "", "-"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def safe_int(s):
        """Convert string to int, return None if invalid."""
        if not s:
            return None
        s = s.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            return None

    # Protein count checks
    for match in re.finditer(r"([\d,]+)\s*(?:disp?rot\s*)?proteins?", a):
        val = safe_int(match.group(1))
        if val is not None and val > 1000:
            total += 1
            if abs(val - GT["n_proteins"]) / GT["n_proteins"] <= NUMERIC_TOL:
                correct += 1

    # Disorder threshold percentages
    for match in re.finditer(r"([\d]+\.?[\d]*)\s*%?\s*(?:exceed|above|over).*?0\.5", a):
        val = safe_float(match.group(1))
        if val is not None:
            total += 1
            if abs(val - GT["pct_above_05"]) <= 3.0:
                correct += 1

    for match in re.finditer(r"([\d]+\.?[\d]*)\s*%?\s*(?:exceed|above|over).*?0\.3", a):
        val = safe_float(match.group(1))
        if val is not None:
            total += 1
            if abs(val - GT["pct_above_03"]) <= 3.0:
                correct += 1

    # Mean disorder
    for match in re.finditer(r"mean\s*[=:]\s*(0\.\d+)", a):
        val = safe_float(match.group(1))
        if val is not None:
            total += 1
            if abs(val - GT["mean_disorder"]) <= 0.02:
                correct += 1

    # Proline percentage
    for match in re.finditer(r"([\d]+\.?[\d]*)\s*%?\s*(?:mean.*proline|proline.*mean)", a):
        val = safe_float(match.group(1))
        if val is not None:
            total += 1
            if abs(val - GT["mean_proline"]) <= 1.0:
                correct += 1

    # Glycine percentage
    for match in re.finditer(r"([\d]+\.?[\d]*)\s*%?\s*(?:mean.*glycine|glycine.*mean)", a):
        val = safe_float(match.group(1))
        if val is not None:
            total += 1
            if abs(val - GT["mean_glycine"]) <= 1.0:
                correct += 1

    # Pfam percentage
    for match in re.finditer(r"([\d]+\.?[\d]*)%?\s*of.*disprot.*pfam|pfam.*([\d]+\.?[\d]*)%", a):
        raw = match.group(1) or match.group(2)
        val = safe_float(raw)
        if val is not None:
            total += 1
            if abs(val - GT["pct_pfam"]) <= 3.0:
                correct += 1

    return correct, total


# ── Completeness checks (C sub-score) ────────────────────────────────

COMPLETENESS_KEYWORDS = {
    "disorder":       ["disorder", "disordered", "idp", "idr"],
    "evidence":       ["experimentally", "validated", "confirmed", "disprot"],
    "mechanism":      ["because", "due to", "mechanism", "disrupts", "prevents"],
    "quantification": [r"\d+\.?\d*%", r"0\.\d+", r"\d+,\d+"],
    "limitation":     ["however", "but", "limitation", "caveat", "although",
                       "cannot", "unreliable", "ambiguous"],
    "recommendation": ["should", "must", "take precedence", "evaluated",
                       "independently", "require", "combination"],
}

def completeness_score(answer, question):
    """Score 0-1 based on how many completeness dimensions are covered."""
    a     = answer.lower()
    q     = question.lower()
    hits  = 0
    total = len(COMPLETENESS_KEYWORDS)

    for dim, signals in COMPLETENESS_KEYWORDS.items():
        for sig in signals:
            if re.search(sig, a):
                hits += 1
                break

    # Bonus: answer directly addresses question topic
    q_tokens = set(re.sub(r"[^a-z0-9\s]", " ", q).split())
    a_tokens = set(re.sub(r"[^a-z0-9\s]", " ", a).split())
    q_content = {t for t in q_tokens if len(t) > 3}
    overlap   = len(q_content & a_tokens) / len(q_content) if q_content else 0
    bonus     = min(0.2, overlap * 0.3)

    return min(1.0, hits / total + bonus)


# ── Traceability checks (T sub-score) ────────────────────────────────

def traceability_score(answer):
    """
    Score 0-1 based on what fraction of major claims can be traced
    to a specific KB passage.
    """
    a         = answer.lower()
    traced    = 0
    total_kb  = len(KB_FACTS)

    for kb_id, facts in KB_FACTS.items():
        kb_hits = sum(1 for fact in facts if re.search(fact.lower(), a))
        if kb_hits >= 2:   # At least 2 facts from this KB entry
            traced += 1

    # Normalize: we expect ~3 KB passages to be traceable per answer
    expected_traces = 3
    return min(1.0, traced / expected_traces)


# ── Combined FACT score ───────────────────────────────────────────────

def fact_score(question, answer):
    """Compute all four FACT sub-scores and combine."""
    a_lower = answer.lower()

    # F -- Fidelity
    f_score = 0.0
    for pattern, weight in FIDELITY_SIGNALS:
        if re.search(pattern, a_lower):
            f_score += weight
    f_score = min(1.0, f_score)

    # A -- Accuracy
    correct, total = check_numeric_accuracy(answer)
    if total == 0:
        # No numbers to check -- assign partial score based on
        # whether factual KB constants appear in text form
        text_facts = sum(1 for kw in [
            "13,396", "0.378", "29.1", "44.2", "6.0", "7.0", "92.6"
        ] if kw in answer)
        a_score = min(1.0, text_facts / 4) if text_facts > 0 else 0.5
    else:
        a_score = correct / total

    # C -- Completeness
    c_score = completeness_score(answer, question)

    # T -- Traceability
    t_score = traceability_score(answer)

    # Final FACT score
    fact    = round((f_score + a_score + c_score + t_score) / 4, 4)

    return {
        "F": round(f_score, 4),
        "A": round(a_score, 4),
        "C": round(c_score, 4),
        "T": round(t_score, 4),
        "FACT": fact,
    }


def fact_label(score):
    if score >= 0.85:   return "HIGHLY FACTUAL  "
    elif score >= 0.70: return "FACTUAL         "
    elif score >= 0.55: return "PARTLY FACTUAL  "
    elif score >= 0.40: return "WEAKLY FACTUAL  "
    else:               return "UNFACTUAL       "


# ── Math helpers ──────────────────────────────────────────────────────

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std(lst):
    mu = mean(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / len(lst)) if lst else 0.0

def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

def percentile(lst, p):
    s   = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ── Parse predictions ─────────────────────────────────────────────────

def parse_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    blocks      = re.split(r"={6,}", text)
    predictions = []
    q_pat = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )

    for block in blocks:
        q_m = q_pat.search(block)
        a_m = a_pat.search(block)
        if q_m and a_m:
            predictions.append({
                "q_num":    int(q_m.group(1)),
                "question": q_m.group(2).strip(),
                "answer":   re.sub(r"\s+", " ", a_m.group(1)).strip(),
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions


# ── Run FACT score analysis ───────────────────────────────────────────

def run_fact_score(predictions):
    results = []
    for p in predictions:
        scores = fact_score(p["question"], p["answer"])
        results.append({
            "q_num":    p["q_num"],
            "question": p["question"],
            **scores,
            "label":    fact_label(scores["FACT"]),
        })
    return results


# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = len(results)

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: FACT Score Analysis")
    lines.append("  Script  : FACT_score3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("FACT SCORE FRAMEWORK")
    lines.append("-" * 70)
    lines.append("  FACT = (F + A + C + T) / 4   (range 0.0 - 1.0)")
    lines.append("")
    lines.append("  F  Fidelity     -- claims supported by DisProt KB signal phrases")
    lines.append("  A  Accuracy     -- numerical values match ground-truth DisProt stats")
    lines.append("  C  Completeness -- key factual dimensions covered for the question")
    lines.append("  T  Traceability -- claims traceable to specific KB passages (KB-001..014)")
    lines.append("")
    lines.append("  Ground-truth DisProt constants:")
    lines.append(f"    Proteins : {GT['n_proteins']:,}     Mean disorder : {GT['mean_disorder']}")
    lines.append(f"    Pct>0.5  : {GT['pct_above_05']}%   Pct>0.3       : {GT['pct_above_03']}%")
    lines.append(f"    Proline  : {GT['mean_proline']}%    Glycine       : {GT['mean_glycine']}%")
    lines.append(f"    Pfam     : {GT['pct_pfam']}%   Numeric tolerance : {NUMERIC_TOL*100:.0f}%")
    lines.append("")
    lines.append("  Score interpretation:")
    lines.append("    >=0.85  HIGHLY FACTUAL    >=0.70  FACTUAL")
    lines.append("    >=0.55  PARTLY FACTUAL    >=0.40  WEAKLY FACTUAL   <0.40  UNFACTUAL")
    lines.append("")

    # Per-question table
    lines.append("PER-QUESTION FACT SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'F':>6}  {'A':>6}  {'C':>6}  {'T':>6}  "
                 f"{'FACT':>7}  {'Label':<18}")
    lines.append("  " + "-" * 60)

    for r in results:
        lines.append(
            f"  {r['q_num']:>4}  {r['F']:>6.4f}  {r['A']:>6.4f}  "
            f"{r['C']:>6.4f}  {r['T']:>6.4f}  "
            f"{r['FACT']:>7.4f}  {r['label']}"
        )

    # Aggregate stats per sub-score
    lines.append("")
    lines.append("AGGREGATE STATISTICS PER SUB-SCORE")
    lines.append("-" * 70)
    lines.append(f"  {'Sub':<4}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  "
                 f"{'Max':>8}  {'Median':>8}  {'P25':>8}  {'P75':>8}")
    lines.append("  " + "-" * 66)

    for key, label in [("F","Fidelity"), ("A","Accuracy"),
                        ("C","Completeness"), ("T","Traceability"), ("FACT","FACT")]:
        vals = [r[key] for r in results]
        lines.append(
            f"  {key:<4}  {mean(vals):>8.4f}  {std(vals):>8.4f}  "
            f"{min(vals):>8.4f}  {max(vals):>8.4f}  {median(vals):>8.4f}  "
            f"{percentile(vals,25):>8.4f}  {percentile(vals,75):>8.4f}"
        )

    # FACT score distribution
    lines.append("")
    lines.append("FACT SCORE DISTRIBUTION")
    lines.append("-" * 70)
    fact_vals = [r["FACT"] for r in results]
    buckets   = [
        ("HIGHLY FACTUAL  (>=0.85)", 0.85, 1.01),
        ("FACTUAL         (>=0.70)", 0.70, 0.85),
        ("PARTLY FACTUAL  (>=0.55)", 0.55, 0.70),
        ("WEAKLY FACTUAL  (>=0.40)", 0.40, 0.55),
        ("UNFACTUAL       (< 0.40)", 0.00, 0.40),
    ]
    for label, lo, hi in buckets:
        count = sum(1 for v in fact_vals if lo <= v < hi)
        bar   = "#" * int(count / n * 40)
        lines.append(f"  {label} : {count:>4} ({count/n*100:>5.1f}%)  {bar}")

    # Sub-score radar summary
    lines.append("")
    lines.append("SUB-SCORE RADAR SUMMARY")
    lines.append("-" * 70)
    sub_means = {k: mean([r[k] for r in results]) for k in ["F","A","C","T"]}
    best_sub  = max(sub_means, key=sub_means.get)
    worst_sub = min(sub_means, key=sub_means.get)
    sub_labels = {"F":"Fidelity","A":"Accuracy","C":"Completeness","T":"Traceability"}
    for k, lbl in sub_labels.items():
        bar = "#" * int(sub_means[k] * 40)
        marker = " <-- STRONGEST" if k == best_sub else \
                 " <-- WEAKEST"   if k == worst_sub else ""
        lines.append(f"  {k}  {lbl:<14} : {sub_means[k]:.4f}  {bar}{marker}")

    # Top and bottom performers
    sorted_results = sorted(results, key=lambda r: r["FACT"], reverse=True)
    lines.append("")
    lines.append("TOP 5 MOST FACTUAL ANSWERS")
    lines.append("-" * 70)
    for r in sorted_results[:5]:
        lines.append(
            f"  Q{r['q_num']:>3}  FACT={r['FACT']:.4f}  "
            f"F={r['F']:.3f} A={r['A']:.3f} C={r['C']:.3f} T={r['T']:.3f}"
        )
        lines.append(f"       \"{r['question'][:62]}\"")

    lines.append("")
    lines.append("BOTTOM 5 LEAST FACTUAL ANSWERS")
    lines.append("-" * 70)
    for r in sorted_results[-5:]:
        lines.append(
            f"  Q{r['q_num']:>3}  FACT={r['FACT']:.4f}  "
            f"F={r['F']:.3f} A={r['A']:.3f} C={r['C']:.3f} T={r['T']:.3f}"
        )
        lines.append(f"       \"{r['question'][:62]}\"")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  F (Fidelity) is expected to be highest in extractive fallback")
    lines.append("  mode since answers directly contain KB signal phrases.")
    lines.append("")
    lines.append("  A (Accuracy) depends on whether numeric values from the KB")
    lines.append("  appear in the answer. Questions that retrieve KB-001 or KB-014")
    lines.append("  score highest on A since those passages are statistics-dense.")
    lines.append("")
    lines.append("  C (Completeness) rewards answers that cover disorder evidence,")
    lines.append("  mechanism, quantification, limitation, and recommendation.")
    lines.append("  Extractive answers covering multiple KB passages score higher.")
    lines.append("")
    lines.append("  T (Traceability) scores how many distinct KB passages contributed")
    lines.append("  facts to the answer. Answers drawing from 3+ KB entries score")
    lines.append("  highest since the RAG system retrieves top-3 passages per query.")
    lines.append("")
    lines.append("  Low FACT scores indicate questions where the retrieved passages")
    lines.append("  were topically mismatched, reducing all four sub-scores together.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC")
    lines.append("=" * 70)

    output = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n[SAVED] {out_path}")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Loading predictions from:\n       {LLM3_PATH}\n")
    predictions = parse_predictions(LLM3_PATH)
    print(f"[INFO] Parsed {len(predictions)} questions")
    print("[INFO] Computing FACT scores...\n")
    results = run_fact_score(predictions)
    write_output(results, OUT_PATH)