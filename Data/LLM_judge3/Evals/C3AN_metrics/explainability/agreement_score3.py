"""
BMEN-499 AlphaFold -- LLM Judge 3: Agreement Score Analysis
-------------------------------------------------------------
File    : agreement_score3.py
Output  : agreement_score3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Measures inter-rater style agreement between three simulated
    "judges" applied to each predicted answer. Each judge evaluates
    the answer from a different perspective. Agreement is then
    computed using:

        - Percent Agreement     : raw proportion of matching judgments
        - Cohen's Kappa         : agreement corrected for chance
        - Fleiss' Kappa         : multi-rater generalization (3 judges)
        - Krippendorff's Alpha  : ordinal agreement (no sklearn needed)

    The three simulated judges are:

        Judge A -- Retrieval Fidelity Judge
                   Does the answer faithfully reflect what was retrieved?
                   Scores 1-3: 1=poor fidelity, 2=partial, 3=high fidelity

        Judge B -- Question Relevance Judge
                   Does the answer actually address the question asked?
                   Scores 1-3: 1=off-topic, 2=partial, 3=directly relevant

        Judge C -- Scientific Validity Judge
                   Are the claims biologically sound and consistent
                   with known IDR/AlphaFold literature?
                   Scores 1-3: 1=invalid/contradictory, 2=acceptable, 3=valid

    Agreement metrics:
        Kappa >= 0.80  : Almost Perfect
        Kappa 0.60-0.79: Substantial
        Kappa 0.40-0.59: Moderate
        Kappa 0.20-0.39: Fair
        Kappa < 0.20   : Slight / Poor

Output: agreement_score3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "agreement_score3_output.txt")

# ── Judge signal definitions ──────────────────────────────────────────

# Judge A -- Retrieval Fidelity
# High (3): answer contains KB passage phrases verbatim/near-verbatim
# Partial (2): answer contains some KB vocabulary
# Low (1): answer diverges from retrieved content
JUDGE_A_HIGH = [
    "most reliable single computational",
    "disprot-annotated disordered regions",
    "pyrrolidine ring disrupts",
    "folding upon.*binding",
    "each region must be evaluated independently",
    "experimental annotations take precedence",
    "risk smoothing out short idrs",
    "sliding window averaging smooths",
    "strongly predicts intrinsic disorder",
    "weaker independent.*predictor",
    "no pfam domains and disorder content",
]
JUDGE_A_MED = [
    "plddt",
    "disprot",
    "pfam",
    "proline",
    "glycine",
    "disorder score",
    "threshold",
    "sliding window",
    "alphafold",
    "idp",
    "idr",
]

# Judge B -- Question Relevance
# Scored by overlap of question keywords with answer
# High (3): answer contains 4+ question content words
# Partial (2): 2-3 question content words
# Low (1): 0-1 question content words
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","this","that","these","those","it","its",
    "as","not","no","so","if","than","then","when","which","who",
    "what","how","also","more","most","after","before","between",
    "into","through","while","both","each","further","once","above",
    "below","up","down","out","about","over","under","again","here",
    "there","where","why","all","any","few","very","just","because",
    "such","only","their","they","them","we","our","can","cannot",
    "per","does","do","did","i","mean","well","across","without",
    "after","during","whether","among","within","against","along",
}

# Judge C -- Scientific Validity
# High (3): answer contains validated scientific claims from IDR literature
# Partial (2): answer contains partially correct claims
# Low (1): answer contains contradictory or unsupported claims
JUDGE_C_HIGH = [
    "plddt below 50.*intrinsic disorder",
    "experimentally validated",
    "disprot.*confirms",
    "proline.*disrupts.*alpha-heli",
    "proline.*disrupts.*beta-sheet",
    "pyrrolidine ring",
    "conformational freedom",
    "molecular recognition feature",
    "morf",
    "intrinsically disordered protein",
    "disorder content.*0\.\d",
    "alphafold.*plddt",
]
JUDGE_C_LOW = [
    "plddt.*reliable.*experimental.*precedence",
    "0\.5.*reliable.*missed",
    "short idrs.*unreliable.*confirmed",
    "smoothed out.*sliding window.*solution",
]

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

# ── Agreement metrics ─────────────────────────────────────────────────

def percent_agreement(ratings_a, ratings_b):
    """Raw proportion of matching ratings between two raters."""
    if not ratings_a or not ratings_b:
        return 0.0
    matches = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    return matches / len(ratings_a)


def cohens_kappa(ratings_a, ratings_b):
    """
    Cohen's Kappa for two raters.
    kappa = (P_o - P_e) / (1 - P_e)
    """
    n      = len(ratings_a)
    if n == 0:
        return 0.0
    categories = sorted(set(ratings_a) | set(ratings_b))
    p_o    = percent_agreement(ratings_a, ratings_b)

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        p_a = ratings_a.count(cat) / n
        p_b = ratings_b.count(cat) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(all_ratings, n_categories=3):
    """
    Fleiss' Kappa for multiple raters.
    all_ratings: list of lists, one per item, each containing ratings
                 from all raters for that item.
    """
    n_items  = len(all_ratings)
    n_raters = len(all_ratings[0]) if all_ratings else 0
    if n_items == 0 or n_raters == 0:
        return 0.0

    categories = list(range(1, n_categories + 1))

    # n_ij = number of raters assigning category j to item i
    n_ij = []
    for item_ratings in all_ratings:
        counts = Counter(item_ratings)
        n_ij.append([counts.get(c, 0) for c in categories])

    # P_j = proportion of all assignments in category j
    total_assignments = n_items * n_raters
    p_j = []
    for j_idx in range(len(categories)):
        p_j.append(sum(row[j_idx] for row in n_ij) / total_assignments)

    # P_i = extent of agreement for item i
    p_i = []
    for row in n_ij:
        if n_raters <= 1:
            p_i.append(1.0)
        else:
            val = (sum(x * (x - 1) for x in row)) / (n_raters * (n_raters - 1))
            p_i.append(val)

    p_bar  = mean(p_i)
    p_e    = sum(pj ** 2 for pj in p_j)

    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def krippendorffs_alpha(all_ratings_by_rater, metric="ordinal"):
    """
    Krippendorff's Alpha for ordinal data.
    all_ratings_by_rater: list of lists, one per rater, each containing
                          one rating per item (or None for missing).
    """
    n_raters = len(all_ratings_by_rater)
    n_items  = len(all_ratings_by_rater[0]) if all_ratings_by_rater else 0

    # Collect all valid pairs
    coincidences = {}
    n_pairable   = 0

    for i in range(n_items):
        item_vals = [all_ratings_by_rater[r][i]
                     for r in range(n_raters)
                     if all_ratings_by_rater[r][i] is not None]
        m_i = len(item_vals)
        if m_i < 2:
            continue
        n_pairable += m_i * (m_i - 1)
        for u in item_vals:
            for v in item_vals:
                if u != v:
                    key = (min(u, v), max(u, v))
                    coincidences[key] = coincidences.get(key, 0) + 1

    if n_pairable == 0:
        return 1.0

    # All values present
    all_vals = [v for rater in all_ratings_by_rater for v in rater if v is not None]
    value_counts = Counter(all_vals)
    total_vals   = sum(value_counts.values())

    # Observed disagreement D_o
    d_o = 0.0
    for (u, v), count in coincidences.items():
        if metric == "ordinal":
            # ordinal distance: (sum of n_c for c between u and v inclusive) - (n_u+n_v)/2
            cats_between = sorted(value_counts.keys())
            idx_u = cats_between.index(u) if u in cats_between else 0
            idx_v = cats_between.index(v) if v in cats_between else 0
            lo, hi = min(idx_u, idx_v), max(idx_u, idx_v)
            g = sum(value_counts[cats_between[k]] for k in range(lo, hi + 1))
            g -= (value_counts[u] + value_counts[v]) / 2
            diff = g ** 2
        else:
            diff = 0 if u == v else 1
        d_o += count * diff

    d_o /= n_pairable

    # Expected disagreement D_e
    d_e = 0.0
    vals_list = sorted(value_counts.keys())
    for i_idx, u in enumerate(vals_list):
        for v in vals_list:
            if u == v:
                continue
            if metric == "ordinal":
                lo  = min(vals_list.index(u), vals_list.index(v))
                hi  = max(vals_list.index(u), vals_list.index(v))
                g   = sum(value_counts[vals_list[k]] for k in range(lo, hi + 1))
                g  -= (value_counts[u] + value_counts[v]) / 2
                diff = g ** 2
            else:
                diff = 1
            d_e += (value_counts[u] * value_counts[v] * diff)

    d_e /= (total_vals * (total_vals - 1)) if total_vals > 1 else 1

    if d_e == 0:
        return 1.0
    return 1 - d_o / d_e


def kappa_label(k):
    if k >= 0.80:   return "Almost Perfect "
    elif k >= 0.60: return "Substantial    "
    elif k >= 0.40: return "Moderate       "
    elif k >= 0.20: return "Fair           "
    else:           return "Slight/Poor    "

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

# ── Judge scoring functions ───────────────────────────────────────────

def judge_a_score(answer):
    """Retrieval Fidelity Judge."""
    a = answer.lower()
    high_hits = sum(1 for sig in JUDGE_A_HIGH if re.search(sig, a))
    med_hits  = sum(1 for sig in JUDGE_A_MED  if sig in a)
    if high_hits >= 2:  return 3
    elif high_hits >= 1 or med_hits >= 5: return 2
    else: return 1


def judge_b_score(question, answer):
    """Question Relevance Judge."""
    q_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", question.lower()).split()
        if t not in STOPWORDS and len(t) > 2
    )
    a_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", answer.lower()).split()
        if t not in STOPWORDS and len(t) > 2
    )
    overlap = len(q_tokens & a_tokens)
    if overlap >= 4:   return 3
    elif overlap >= 2: return 2
    else:              return 1


def judge_c_score(answer):
    """Scientific Validity Judge."""
    a = answer.lower()
    high_hits = sum(1 for sig in JUDGE_C_HIGH if re.search(sig, a))
    low_hits  = sum(1 for sig in JUDGE_C_LOW  if re.search(sig, a))
    if low_hits >= 2:       return 1
    elif high_hits >= 3:    return 3
    elif high_hits >= 1:    return 2
    else:                   return 1

# ── Run agreement analysis ────────────────────────────────────────────

def run_agreement(predictions):
    per_q = []
    for p in predictions:
        sa = judge_a_score(p["answer"])
        sb = judge_b_score(p["question"], p["answer"])
        sc = judge_c_score(p["answer"])
        per_q.append({
            "q_num":    p["q_num"],
            "question": p["question"],
            "judge_a":  sa,
            "judge_b":  sb,
            "judge_c":  sc,
            "mean":     round(mean([sa, sb, sc]), 4),
            "agree_ab": sa == sb,
            "agree_ac": sa == sc,
            "agree_bc": sb == sc,
            "full_agree": sa == sb == sc,
        })

    ratings_a = [p["judge_a"] for p in per_q]
    ratings_b = [p["judge_b"] for p in per_q]
    ratings_c = [p["judge_c"] for p in per_q]

    # Pairwise percent agreement
    pct_ab = percent_agreement(ratings_a, ratings_b)
    pct_ac = percent_agreement(ratings_a, ratings_c)
    pct_bc = percent_agreement(ratings_b, ratings_c)
    pct_all = sum(1 for p in per_q if p["full_agree"]) / len(per_q)

    # Cohen's Kappa pairwise
    kappa_ab = cohens_kappa(ratings_a, ratings_b)
    kappa_ac = cohens_kappa(ratings_a, ratings_c)
    kappa_bc = cohens_kappa(ratings_b, ratings_c)

    # Fleiss' Kappa
    all_ratings = [[p["judge_a"], p["judge_b"], p["judge_c"]] for p in per_q]
    fk = fleiss_kappa(all_ratings, n_categories=3)

    # Krippendorff's Alpha
    ka = krippendorffs_alpha([ratings_a, ratings_b, ratings_c], metric="ordinal")

    return {
        "per_q":    per_q,
        "ratings_a": ratings_a,
        "ratings_b": ratings_b,
        "ratings_c": ratings_c,
        "pct_ab":   round(pct_ab, 4),
        "pct_ac":   round(pct_ac, 4),
        "pct_bc":   round(pct_bc, 4),
        "pct_all":  round(pct_all, 4),
        "kappa_ab": round(kappa_ab, 4),
        "kappa_ac": round(kappa_ac, 4),
        "kappa_bc": round(kappa_bc, 4),
        "fleiss_k": round(fk, 4),
        "kripp_a":  round(ka, 4),
        "n":        len(per_q),
    }

# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = results["n"]

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Agreement Score Analysis")
    lines.append("  Script  : agreement_score3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("THREE-JUDGE AGREEMENT FRAMEWORK")
    lines.append("-" * 70)
    lines.append("  Judge A -- Retrieval Fidelity    (1=poor  2=partial  3=high)")
    lines.append("             Does the answer faithfully reflect retrieved passages?")
    lines.append("  Judge B -- Question Relevance    (1=off-topic  2=partial  3=direct)")
    lines.append("             Does the answer address what the question actually asks?")
    lines.append("  Judge C -- Scientific Validity   (1=invalid  2=acceptable  3=valid)")
    lines.append("             Are claims consistent with IDR/AlphaFold literature?")
    lines.append("")
    lines.append("  Agreement metrics computed:")
    lines.append("    Percent Agreement  -- raw proportion of matching judgments")
    lines.append("    Cohen's Kappa      -- chance-corrected pairwise agreement")
    lines.append("    Fleiss' Kappa      -- multi-rater generalization (3 judges)")
    lines.append("    Krippendorff Alpha -- ordinal reliability coefficient")
    lines.append("")
    lines.append("  Kappa interpretation:")
    lines.append("    >= 0.80  Almost Perfect   0.60-0.79  Substantial")
    lines.append("    0.40-0.59  Moderate       0.20-0.39  Fair   < 0.20  Slight")
    lines.append("")

    # Per-question table
    lines.append("PER-QUESTION JUDGE SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'JA':>4} {'JB':>4} {'JC':>4}  {'Mean':>6}  "
                 f"{'AB':>4} {'AC':>4} {'BC':>4}  {'All':>4}")
    lines.append("  " + "-" * 52)

    for p in results["per_q"]:
        lines.append(
            f"  {p['q_num']:>4}  "
            f"{p['judge_a']:>4} {p['judge_b']:>4} {p['judge_c']:>4}  "
            f"{p['mean']:>6.3f}  "
            f"{'Y' if p['agree_ab'] else 'N':>4} "
            f"{'Y' if p['agree_ac'] else 'N':>4} "
            f"{'Y' if p['agree_bc'] else 'N':>4}  "
            f"{'Y' if p['full_agree'] else 'N':>4}"
        )

    # Score distributions
    lines.append("")
    lines.append("SCORE DISTRIBUTIONS PER JUDGE")
    lines.append("-" * 70)
    for judge_key, judge_label in [
        ("ratings_a", "Judge A (Retrieval Fidelity)"),
        ("ratings_b", "Judge B (Question Relevance)"),
        ("ratings_c", "Judge C (Scientific Validity)"),
    ]:
        vals   = results[judge_key]
        counts = Counter(vals)
        mu     = mean(vals)
        lines.append(f"  {judge_label}")
        lines.append(f"    Score 1: {counts.get(1,0):>4} ({counts.get(1,0)/n*100:>5.1f}%)"
                     f"  Score 2: {counts.get(2,0):>4} ({counts.get(2,0)/n*100:>5.1f}%)"
                     f"  Score 3: {counts.get(3,0):>4} ({counts.get(3,0)/n*100:>5.1f}%)")
        lines.append(f"    Mean: {mu:.4f}   Std: {std(vals):.4f}   Median: {median(vals):.1f}")
        lines.append("")

    # Agreement matrix
    lines.append("PAIRWISE PERCENT AGREEMENT")
    lines.append("-" * 70)
    lines.append(f"  Judge A vs Judge B : {results['pct_ab']:.4f}  ({results['pct_ab']*100:.1f}%)")
    lines.append(f"  Judge A vs Judge C : {results['pct_ac']:.4f}  ({results['pct_ac']*100:.1f}%)")
    lines.append(f"  Judge B vs Judge C : {results['pct_bc']:.4f}  ({results['pct_bc']*100:.1f}%)")
    lines.append(f"  All three agree    : {results['pct_all']:.4f}  ({results['pct_all']*100:.1f}%)")
    lines.append("")

    # Cohen's Kappa
    lines.append("COHEN'S KAPPA (pairwise, chance-corrected)")
    lines.append("-" * 70)
    lines.append(f"  Kappa (A vs B) : {results['kappa_ab']:>7.4f}  {kappa_label(results['kappa_ab'])}")
    lines.append(f"  Kappa (A vs C) : {results['kappa_ac']:>7.4f}  {kappa_label(results['kappa_ac'])}")
    lines.append(f"  Kappa (B vs C) : {results['kappa_bc']:>7.4f}  {kappa_label(results['kappa_bc'])}")
    mean_kappa = mean([results["kappa_ab"], results["kappa_ac"], results["kappa_bc"]])
    lines.append(f"  Mean Kappa     : {mean_kappa:>7.4f}  {kappa_label(mean_kappa)}")
    lines.append("")

    # Fleiss and Krippendorff
    lines.append("MULTI-RATER AGREEMENT METRICS")
    lines.append("-" * 70)
    lines.append(f"  Fleiss' Kappa         : {results['fleiss_k']:>7.4f}  {kappa_label(results['fleiss_k'])}")
    lines.append(f"  Krippendorff's Alpha  : {results['kripp_a']:>7.4f}  {kappa_label(results['kripp_a'])}")
    lines.append("")

    # Disagreement analysis
    disagree_all = [p for p in results["per_q"] if not p["full_agree"]]
    agree_all    = [p for p in results["per_q"] if p["full_agree"]]
    lines.append("AGREEMENT / DISAGREEMENT SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Full agreement (all 3 judges match) : {len(agree_all):>4} / {n}  ({len(agree_all)/n*100:.1f}%)")
    lines.append(f"  Any disagreement                    : {len(disagree_all):>4} / {n}  ({len(disagree_all)/n*100:.1f}%)")
    lines.append("")

    if agree_all:
        lines.append("  Questions with full agreement (sample):")
        for p in agree_all[:5]:
            lines.append(f"    Q{p['q_num']:>3}  JA={p['judge_a']} JB={p['judge_b']} JC={p['judge_c']}"
                         f"  \"{p['question'][:55]}\"")
    lines.append("")

    # Worst disagreements
    max_spread = [(p, max(p["judge_a"], p["judge_b"], p["judge_c"]) -
                      min(p["judge_a"], p["judge_b"], p["judge_c"]))
                  for p in results["per_q"]]
    max_spread.sort(key=lambda x: -x[1])
    lines.append("  Questions with largest judge disagreement (spread):")
    for p, spread in max_spread[:5]:
        lines.append(f"    Q{p['q_num']:>3}  spread={spread}  "
                     f"JA={p['judge_a']} JB={p['judge_b']} JC={p['judge_c']}"
                     f"  \"{p['question'][:48]}\"")

    # Overall verdict
    lines.append("")
    lines.append("OVERALL AGREEMENT VERDICT")
    lines.append("-" * 70)
    fk    = results["fleiss_k"]
    ka    = results["kripp_a"]
    lines.append(f"  Fleiss' Kappa        : {fk:.4f}  --> {kappa_label(fk)}")
    lines.append(f"  Krippendorff's Alpha : {ka:.4f}  --> {kappa_label(ka)}")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  Judge A (Retrieval Fidelity) is expected to score HIGH for")
    lines.append("  extractive fallback mode because answers ARE the retrieved")
    lines.append("  passages concatenated directly.")
    lines.append("")
    lines.append("  Judge B (Question Relevance) is expected to show more variance")
    lines.append("  because BiomedBERT retrieval sometimes pulls passages that are")
    lines.append("  topically related but do not directly answer the specific question.")
    lines.append("")
    lines.append("  Judge C (Scientific Validity) should be moderate-to-high since")
    lines.append("  the KB passages are grounded in DisProt experimental data, but")
    lines.append("  contradictory co-retrieved passages can reduce validity scores.")
    lines.append("")
    lines.append("  Low Kappa between Judge A and Judge B reflects the core tension")
    lines.append("  in RAG systems: high retrieval fidelity does not guarantee high")
    lines.append("  question relevance when the retriever pulls semantically similar")
    lines.append("  but question-mismatched passages.")
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
    print("[INFO] Running agreement score analysis...\n")
    results = run_agreement(predictions)
    write_output(results, OUT_PATH)