"""
BMEN-499 AlphaFold -- LLM Judge 3: Likert Score Analysis
----------------------------------------------------------
File    : likert_score3.py
Output  : likert_score3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Scores each predicted answer on a 1-5 Likert scale across
    six explainability dimensions. Each dimension is evaluated
    using rule-based heuristics derived from the answer text.

Likert Dimensions:
    L1  -- Factual Grounding
           Does the answer cite specific numerical values, thresholds,
           or dataset statistics (e.g. 13,396 proteins, 0.5 cutoff)?
           1 = no facts cited  ...  5 = multiple specific facts cited

    L2  -- Causal Explanation
           Does the answer explain WHY something is true, not just WHAT?
           (e.g. "proline disrupts alpha-helices BECAUSE its ring...")
           1 = no causal language  ...  5 = clear causal chain present

    L3  -- Uncertainty Acknowledgment
           Does the answer acknowledge limits, ambiguity, or gray zones?
           1 = overconfident with no hedging  ...  5 = explicit uncertainty

    L4  -- Biological Specificity
           Does the answer use domain-specific biomedical terminology
           (IDR, pLDDT, MoRF, Pfam, IDP, pyrrolidine, etc.)?
           1 = generic language  ...  5 = rich domain terminology

    L5  -- Answer Completeness
           Does the answer address multiple aspects of the question
           (length + content depth + multi-evidence)?
           1 = single sentence / partial  ...  5 = thorough multi-evidence

    L6  -- Actionability
           Does the answer give a practical recommendation or
           implication (e.g. "use experimental validation", "window
           size must be chosen carefully")?
           1 = no guidance  ...  5 = clear actionable recommendation

Likert scale:
    5 = Strongly Present   4 = Present    3 = Partial
    2 = Weak               1 = Absent

Output: likert_score3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "likert_score3_output.txt")

# ── Likert scoring signals ─────────────────────────────────────────────

# L1 -- Factual grounding signals (numbers, thresholds, dataset stats)
L1_SIGNALS = [
    r"\d{1,3}[,\d]*\s*(proteins?|regions?|residues?)",  # e.g. 13,396 proteins
    r"\d+\.\d+\s*%",                                     # e.g. 29.1%
    r"(0\.[0-9]+)\s*(threshold|cutoff|score|mean)",      # e.g. 0.5 threshold
    r"mean\s*[=:]\s*\d+\.\d+",                           # e.g. mean=0.378
    r"plddt\s*(below|above|of|score)?\s*\d+",            # e.g. pLDDT below 50
    r"\d+\s*(aa|amino acid|residue)",                    # e.g. 30 aa
]

# L2 -- Causal explanation signals
L2_SIGNALS = [
    "because", "due to", "since", "therefore", "thus", "hence",
    "as a result", "which causes", "disrupts", "prevents",
    "leads to", "resulting in", "by disrupting", "ring disrupts",
    "preventing", "correlat", "associated with",
]

# L3 -- Uncertainty acknowledgment signals
L3_SIGNALS = [
    "cannot be confidently", "ambiguous", "gray zone", "uncertain",
    "require.*validation", "may be", "might be", "unclear",
    "hard to predict", "difficult to predict", "not yet",
    "conditional", "secondary validation", "experimental methods",
    "limited sequence context", "risk", "must be chosen carefully",
]

# L4 -- Biological specificity signals (domain terminology)
L4_SIGNALS = [
    "idr", "idp", "plddt", "morf", "molecular recognition",
    "pfam", "intrinsically disordered", "pyrrolidine",
    "alpha-heli", "beta-sheet", "disprot", "alphafold",
    "per-residue", "sliding window", "isotonic", "calibrat",
    "biomedbert", "biomistral", "neurosymbolic", "rag",
    "backbone", "conformational", "secondary structure",
]

# L5 -- Completeness signals (length + multi-evidence markers)
L5_LENGTH_THRESHOLDS = [30, 50, 70, 90, 110]  # word count thresholds for scores 1-5

L5_MULTI_EVIDENCE = [
    "however", "additionally", "furthermore", "in contrast",
    "on the other hand", "while", "although", "whereas",
    "consistently", "both", "together", "combined",
    "retrieved", "retrieved 1", "retrieved 2", "retrieved 3",
]

# L6 -- Actionability signals
L6_SIGNALS = [
    "should", "must", "recommend", "use", "apply",
    "take precedence", "requires", "need", "consider",
    "evaluated independently", "window size must",
    "secondary validation", "experimental validation",
    "carefully", "in combination", "consult", "verify",
]

# ── Scoring functions ─────────────────────────────────────────────────

def score_l1(answer: str) -> int:
    """Factual grounding: count regex pattern hits."""
    hits = sum(1 for pat in L1_SIGNALS
               if re.search(pat, answer, re.IGNORECASE))
    if hits >= 5: return 5
    elif hits == 4: return 4
    elif hits == 3: return 3
    elif hits == 2: return 2
    elif hits == 1: return 2
    else:           return 1


def score_l2(answer: str) -> int:
    """Causal explanation: count causal signal phrases."""
    a_lower = answer.lower()
    hits    = sum(1 for sig in L2_SIGNALS if sig in a_lower)
    if hits >= 4:   return 5
    elif hits == 3: return 4
    elif hits == 2: return 3
    elif hits == 1: return 2
    else:           return 1


def score_l3(answer: str) -> int:
    """Uncertainty acknowledgment: count hedging signals."""
    a_lower = answer.lower()
    hits    = sum(1 for sig in L3_SIGNALS
                  if re.search(sig, a_lower))
    if hits >= 4:   return 5
    elif hits == 3: return 4
    elif hits == 2: return 3
    elif hits == 1: return 2
    else:           return 1


def score_l4(answer: str) -> int:
    """Biological specificity: count domain terms."""
    a_lower = answer.lower()
    hits    = sum(1 for sig in L4_SIGNALS if sig in a_lower)
    if hits >= 8:   return 5
    elif hits >= 6: return 4
    elif hits >= 4: return 3
    elif hits >= 2: return 2
    else:           return 1


def score_l5(answer: str) -> int:
    """Completeness: word count + multi-evidence markers."""
    wc      = len(answer.split())
    a_lower = answer.lower()
    evid    = sum(1 for sig in L5_MULTI_EVIDENCE if sig in a_lower)

    # Base score from length
    if wc >= L5_LENGTH_THRESHOLDS[4]:   base = 5
    elif wc >= L5_LENGTH_THRESHOLDS[3]: base = 4
    elif wc >= L5_LENGTH_THRESHOLDS[2]: base = 3
    elif wc >= L5_LENGTH_THRESHOLDS[1]: base = 2
    else:                               base = 1

    # Boost by 1 if multiple evidence markers present
    if evid >= 3:
        base = min(5, base + 1)
    return base


def score_l6(answer: str) -> int:
    """Actionability: count recommendation/guidance signals."""
    a_lower = answer.lower()
    hits    = sum(1 for sig in L6_SIGNALS if sig in a_lower)
    if hits >= 5:   return 5
    elif hits >= 4: return 4
    elif hits >= 3: return 3
    elif hits >= 2: return 2
    elif hits == 1: return 2
    else:           return 1


def score_answer(answer: str) -> dict:
    l1 = score_l1(answer)
    l2 = score_l2(answer)
    l3 = score_l3(answer)
    l4 = score_l4(answer)
    l5 = score_l5(answer)
    l6 = score_l6(answer)
    total = l1 + l2 + l3 + l4 + l5 + l6
    mean  = round(total / 6, 4)
    return {
        "L1": l1, "L2": l2, "L3": l3,
        "L4": l4, "L5": l5, "L6": l6,
        "total": total, "mean": mean,
    }


def likert_label(score: float) -> str:
    if score >= 4.5:   return "EXCELLENT  "
    elif score >= 3.5: return "GOOD       "
    elif score >= 2.5: return "MODERATE   "
    elif score >= 1.5: return "WEAK       "
    else:              return "POOR       "


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

# ── Parse predictions ─────────────────────────────────────────────────

def parse_predictions(filepath: str) -> list:
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

# ── Write output ──────────────────────────────────────────────────────

def write_output(predictions: list, out_path: str):
    scored = []
    for p in predictions:
        s = score_answer(p["answer"])
        scored.append({**p, **s})

    lines = []
    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Likert Score Analysis")
    lines.append("  Script  : likert_score3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions scored : {len(scored)}")
    lines.append("  Scale   : 1 (Absent) --> 5 (Strongly Present)")
    lines.append("=" * 70)
    lines.append("")

    lines.append("LIKERT DIMENSION DEFINITIONS")
    lines.append("-" * 70)
    lines.append("  L1  Factual Grounding         -- specific numbers, thresholds, stats")
    lines.append("  L2  Causal Explanation         -- WHY not just WHAT (causal language)")
    lines.append("  L3  Uncertainty Acknowledgment -- hedging, gray zones, limits")
    lines.append("  L4  Biological Specificity     -- domain terminology depth")
    lines.append("  L5  Answer Completeness        -- length + multi-evidence")
    lines.append("  L6  Actionability              -- practical guidance or recommendation")
    lines.append("")

    # Per-question scores
    lines.append("PER-QUESTION LIKERT SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'L1':>4} {'L2':>4} {'L3':>4} {'L4':>4} {'L5':>4} {'L6':>4}  "
                 f"{'Total':>6}  {'Mean':>6}  {'Label':<12}")
    lines.append("  " + "-" * 60)

    for s in scored:
        lines.append(
            f"  {s['q_num']:>4}  "
            f"{s['L1']:>4} {s['L2']:>4} {s['L3']:>4} "
            f"{s['L4']:>4} {s['L5']:>4} {s['L6']:>4}  "
            f"{s['total']:>6}  {s['mean']:>6.2f}  "
            f"{likert_label(s['mean'])}"
        )

    # Aggregate per dimension
    dims = ["L1", "L2", "L3", "L4", "L5", "L6"]
    dim_labels = {
        "L1": "Factual Grounding",
        "L2": "Causal Explanation",
        "L3": "Uncertainty Acknowledgment",
        "L4": "Biological Specificity",
        "L5": "Answer Completeness",
        "L6": "Actionability",
    }

    lines.append("")
    lines.append("AGGREGATE SCORES PER DIMENSION")
    lines.append("-" * 70)
    lines.append(f"  {'Dim':<4} {'Label':<30} {'Mean':>6} {'Std':>6} "
                 f"{'Min':>4} {'Max':>4} {'Median':>7}  {'Overall Label':<12}")
    lines.append("  " + "-" * 68)

    dim_means = {}
    for dim in dims:
        vals     = [s[dim] for s in scored]
        mu       = mean(vals)
        sd       = std(vals)
        mn       = min(vals)
        mx       = max(vals)
        med      = median(vals)
        dim_means[dim] = mu
        lines.append(
            f"  {dim:<4} {dim_labels[dim]:<30} {mu:>6.2f} {sd:>6.2f} "
            f"{mn:>4} {mx:>4} {med:>7.2f}  {likert_label(mu)}"
        )

    # Overall mean score
    all_means = [s["mean"] for s in scored]
    overall   = mean(all_means)
    lines.append("")
    lines.append(f"  Overall Mean Likert Score : {overall:.4f} / 5.00  "
                 f"-->  {likert_label(overall)}")

    # Score distribution
    lines.append("")
    lines.append("MEAN SCORE DISTRIBUTION")
    lines.append("-" * 70)
    buckets = {
        "EXCELLENT  (4.5-5.0)": sum(1 for s in scored if s["mean"] >= 4.5),
        "GOOD       (3.5-4.4)": sum(1 for s in scored if 3.5 <= s["mean"] < 4.5),
        "MODERATE   (2.5-3.4)": sum(1 for s in scored if 2.5 <= s["mean"] < 3.5),
        "WEAK       (1.5-2.4)": sum(1 for s in scored if 1.5 <= s["mean"] < 2.5),
        "POOR       (1.0-1.4)": sum(1 for s in scored if s["mean"] < 1.5),
    }
    n = len(scored)
    for label, count in buckets.items():
        bar = "#" * int(count / n * 40)
        lines.append(f"  {label} : {count:>4} ({count/n*100:>5.1f}%)  {bar}")

    # Top and bottom 5 by mean
    sorted_scored = sorted(scored, key=lambda s: s["mean"], reverse=True)
    lines.append("")
    lines.append("TOP 5 HIGHEST SCORING ANSWERS")
    lines.append("-" * 70)
    for s in sorted_scored[:5]:
        lines.append(f"  Q{s['q_num']:>3}  mean={s['mean']:.2f}  "
                     f"L1={s['L1']} L2={s['L2']} L3={s['L3']} "
                     f"L4={s['L4']} L5={s['L5']} L6={s['L6']}")
        lines.append(f"       \"{s['question'][:60]}\"")

    lines.append("")
    lines.append("BOTTOM 5 LOWEST SCORING ANSWERS")
    lines.append("-" * 70)
    for s in sorted_scored[-5:]:
        lines.append(f"  Q{s['q_num']:>3}  mean={s['mean']:.2f}  "
                     f"L1={s['L1']} L2={s['L2']} L3={s['L3']} "
                     f"L4={s['L4']} L5={s['L5']} L6={s['L6']}")
        lines.append(f"       \"{s['question'][:60]}\"")

    # Weakest dimension
    weakest_dim = min(dim_means, key=dim_means.get)
    strongest_dim = max(dim_means, key=dim_means.get)

    lines.append("")
    lines.append("DIMENSION RANKINGS")
    lines.append("-" * 70)
    ranked = sorted(dim_means.items(), key=lambda x: -x[1])
    for rank, (dim, mu) in enumerate(ranked, 1):
        bar = "#" * int(mu / 5 * 30)
        lines.append(f"  #{rank}  {dim}  {dim_labels[dim]:<30}  "
                     f"{mu:.2f}  {bar}")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append(f"  Strongest dimension : {strongest_dim} -- {dim_labels[strongest_dim]}")
    lines.append(f"  Weakest dimension   : {weakest_dim}  -- {dim_labels[weakest_dim]}")
    lines.append("")
    lines.append("  LLM Judge 3 in extractive fallback mode scores well on L4")
    lines.append("  (Biological Specificity) because KB passages are rich in domain")
    lines.append("  terms (pLDDT, IDR, MoRF, DisProt). L3 (Uncertainty) scores")
    lines.append("  moderately because the KB includes gray zone language. L2")
    lines.append("  (Causal Explanation) and L6 (Actionability) are typically")
    lines.append("  lower in extractive mode since causal synthesis requires the")
    lines.append("  generative model (BioMistral-7B) to connect retrieved facts.")
    lines.append("  Loading BioMistral-7B would improve L2, L5, and L6 scores.")
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
    print("[INFO] Computing Likert scores...\n")
    write_output(predictions, OUT_PATH)