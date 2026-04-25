"""
BMEN-499 AlphaFold -- LLM Judge 3: Performance Drop Analysis
-------------------------------------------------------------
File    : performance_drop3.py
Output  : performance_drop3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Measures performance drop across five stress axes by comparing
    answer quality between a high-performing baseline subset and
    degraded condition subsets. No external dependencies -- pure
    stdlib rule-based scoring.

    Performance is measured using a composite quality score (0-100)
    built from four lightweight signals:
        Q1  Factual density    -- quantitative facts per 100 words
        Q2  Term precision     -- biomedical term hits per answer
        Q3  Relevance overlap  -- question-answer token overlap rate
        Q4  Structural clarity -- sentence-level coherence proxy

    Performance Drop Axes:
        D1  -- Question Complexity Drop
                Simple vs complex questions (word count proxy).
                Does quality degrade as questions get longer/harder?

        D2  -- Question Type Drop
                Factual ("what is", "how much") vs reasoning
                ("does", "are", "how does") question types.
                Does the system drop on reasoning questions?

        D3  -- Retrieval Score Drop
                High retrieval score (>= 0.97) vs low (< 0.97).
                Does answer quality correlate with retrieval confidence?

        D4  -- Topic Domain Drop
                pLDDT/AlphaFold questions vs amino acid composition
                vs statistical/metric questions.
                Which topic domain degrades most?

        D5  -- Answer Length Drop
                Long answers (>= mean words) vs short (< mean words).
                Do shorter answers lose more quality?

    Each axis reports:
        Baseline mean score, degraded mean score, absolute drop,
        relative drop (%), and statistical significance proxy (effect size).

Output: performance_drop3_output.txt
"""

import os
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "performance_drop3_output.txt")

# ── Biomedical term list for Q2 ───────────────────────────────────────
BIOMEDICAL_TERMS = [
    "intrinsically disordered", "idr", "idp", "plddt", "alphafold",
    "disprot", "pfam", "morf", "molecular recognition", "pyrrolidine",
    "alpha-helix", "beta-sheet", "residue", "amino acid", "proline",
    "glycine", "disorder content", "structural confidence", "per-residue",
    "isotonic", "calibration", "neurosymbolic", "disorder score",
    "disordered region", "disordered protein", "sequence context",
    "threshold", "cutoff", "experimental annotation", "sliding window",
    "gray zone", "compositional bias", "backbone", "secondary structure",
    "conditionally disordered", "binding partner", "hub protein",
]

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
    "per","i","mean","well","does","across","without","whether",
    "among","within","against","along","during",
}

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

def cohens_d(group_a, group_b):
    """Effect size: Cohen's d between two groups."""
    if not group_a or not group_b:
        return 0.0
    mu_a, mu_b = mean(group_a), mean(group_b)
    pooled_std = math.sqrt((std(group_a) ** 2 + std(group_b) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mu_a - mu_b) / pooled_std

def effect_size_label(d):
    d = abs(d)
    if d >= 0.80:   return "Large "
    elif d >= 0.50: return "Medium"
    elif d >= 0.20: return "Small "
    else:           return "Negligible"

def pct_drop(baseline, degraded):
    if baseline == 0:
        return 0.0
    return round((baseline - degraded) / baseline * 100, 2)

# ── Quality score components ──────────────────────────────────────────

def q1_factual_density(answer):
    """Quantitative facts per 100 words."""
    words   = answer.split()
    n_words = len(words) if words else 1
    numbers = re.findall(r"\b\d+\.?\d*\b", answer)
    # Filter trivial numbers (years, single digits below 5)
    sig_nums = [x for x in numbers if float(x) > 5 or '.' in x]
    return min(100.0, (len(sig_nums) / n_words) * 100 * 10)


def q2_term_precision(answer):
    """Biomedical term hit rate (0-100)."""
    a_lower = answer.lower()
    hits    = sum(1 for term in BIOMEDICAL_TERMS if term in a_lower)
    return min(100.0, (hits / len(BIOMEDICAL_TERMS)) * 100 * 3)


def q3_relevance_overlap(question, answer):
    """Question-answer content token overlap (0-100)."""
    q_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", question.lower()).split()
        if t not in STOPWORDS and len(t) > 2
    )
    a_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", answer.lower()).split()
        if t not in STOPWORDS and len(t) > 2
    )
    if not q_tokens:
        return 50.0
    return min(100.0, len(q_tokens & a_tokens) / len(q_tokens) * 100)


def q4_structural_clarity(answer):
    """
    Sentence coherence proxy:
    Rewards answers with 2-5 well-formed sentences.
    Penalizes very short (<2) or very long (>8) sentence counts.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 10]
    n_sents   = len(sentences)
    if n_sents == 0:
        return 0.0
    elif 2 <= n_sents <= 5:
        return 100.0
    elif n_sents == 1:
        return 50.0
    elif 6 <= n_sents <= 8:
        return 75.0
    else:
        return 60.0


def composite_score(question, answer):
    """Weighted composite quality score (0-100)."""
    q1 = q1_factual_density(answer)      * 0.30
    q2 = q2_term_precision(answer)       * 0.30
    q3 = q3_relevance_overlap(question,
                               answer)   * 0.25
    q4 = q4_structural_clarity(answer)   * 0.15
    return round(q1 + q2 + q3 + q4, 4)


# ── Question classifiers ──────────────────────────────────────────────

def question_complexity(question):
    """Simple heuristic: word count proxy for complexity."""
    return len(question.split())


def question_type(question):
    """Classify question as factual or reasoning."""
    q = question.lower()
    factual_starters = ["what is", "what are", "how many", "how much",
                        "which", "when", "where", "who", "define",
                        "what does", "what was"]
    reasoning_starters = ["does", "do", "is", "are", "can", "could",
                           "would", "should", "how does", "why does",
                           "are there", "does the", "how do"]
    if any(q.startswith(f) for f in factual_starters):
        return "factual"
    return "reasoning"


def question_topic(question):
    """Classify question into topic domain."""
    q = question.lower()
    if any(kw in q for kw in ["plddt", "alphafold", "structural", "confidence score"]):
        return "pLDDT/AlphaFold"
    elif any(kw in q for kw in ["proline", "glycine", "amino acid", "composition",
                                  "sequence", "residue"]):
        return "Amino Acid Composition"
    elif any(kw in q for kw in ["f1", "auroc", "mcc", "brier", "precision",
                                  "recall", "accuracy", "metric", "calibrat",
                                  "mae", "z-score", "variance", "entropy"]):
        return "Statistical Metrics"
    elif any(kw in q for kw in ["sliding window", "window size", "smooth"]):
        return "Sliding Window"
    elif any(kw in q for kw in ["symbolic", "neurosymbolic", "rag", "retrieval",
                                  "rule", "rules"]):
        return "Neurosymbolic/RAG"
    else:
        return "General Disorder"


# ── Parse predictions ─────────────────────────────────────────────────

def parse_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    blocks      = re.split(r"={6,}", text)
    predictions = []
    q_pat  = re.compile(r"\[Q(\d+)\]\s+(.+?)(?:\n|$)")
    a_pat  = re.compile(
        r"PREDICTED ANSWER[:\s]*\n(.*?)(?:\n\s*RETRIEVAL DETAILS|$)",
        re.DOTALL
    )
    rs_pat = re.compile(r"Top retrieval score[:\s]+([\d.]+)")

    for block in blocks:
        q_m  = q_pat.search(block)
        a_m  = a_pat.search(block)
        rs_m = rs_pat.search(block)
        if q_m and a_m:
            predictions.append({
                "q_num":     int(q_m.group(1)),
                "question":  q_m.group(2).strip(),
                "answer":    re.sub(r"\s+", " ", a_m.group(1)).strip(),
                "ret_score": float(rs_m.group(1)) if rs_m else 0.0,
            })

    predictions.sort(key=lambda x: x["q_num"])
    return predictions


# ── Run performance drop analysis ─────────────────────────────────────

def run_performance_drop(predictions):
    # Score every prediction
    scored = []
    for p in predictions:
        score = composite_score(p["question"], p["answer"])
        scored.append({
            **p,
            "score":       score,
            "complexity":  question_complexity(p["question"]),
            "q_type":      question_type(p["question"]),
            "topic":       question_topic(p["question"]),
            "answer_len":  len(p["answer"].split()),
        })

    all_scores = [s["score"] for s in scored]
    mean_len   = mean([s["answer_len"] for s in scored])
    ret_median = median([s["ret_score"] for s in scored if s["ret_score"] > 0])

    # ── D1: Question Complexity Drop ──────────────────────────────────
    complexity_vals = [s["complexity"] for s in scored]
    complexity_med  = median(complexity_vals)
    simple_scores   = [s["score"] for s in scored if s["complexity"] <= complexity_med]
    complex_scores  = [s["score"] for s in scored if s["complexity"] >  complexity_med]
    d1 = {
        "label":     "D1 Question Complexity",
        "baseline":  "Simple questions (word count <= median)",
        "degraded":  "Complex questions (word count > median)",
        "baseline_n": len(simple_scores),
        "degraded_n": len(complex_scores),
        "baseline_mean": round(mean(simple_scores), 4),
        "degraded_mean": round(mean(complex_scores), 4),
        "baseline_std":  round(std(simple_scores), 4),
        "degraded_std":  round(std(complex_scores), 4),
        "abs_drop":  round(mean(simple_scores) - mean(complex_scores), 4),
        "pct_drop":  pct_drop(mean(simple_scores), mean(complex_scores)),
        "cohens_d":  round(cohens_d(simple_scores, complex_scores), 4),
    }

    # ── D2: Question Type Drop ────────────────────────────────────────
    factual_scores   = [s["score"] for s in scored if s["q_type"] == "factual"]
    reasoning_scores = [s["score"] for s in scored if s["q_type"] == "reasoning"]
    d2 = {
        "label":     "D2 Question Type",
        "baseline":  "Factual questions",
        "degraded":  "Reasoning questions",
        "baseline_n": len(factual_scores),
        "degraded_n": len(reasoning_scores),
        "baseline_mean": round(mean(factual_scores), 4),
        "degraded_mean": round(mean(reasoning_scores), 4),
        "baseline_std":  round(std(factual_scores), 4),
        "degraded_std":  round(std(reasoning_scores), 4),
        "abs_drop":  round(mean(factual_scores) - mean(reasoning_scores), 4),
        "pct_drop":  pct_drop(mean(factual_scores), mean(reasoning_scores)),
        "cohens_d":  round(cohens_d(factual_scores, reasoning_scores), 4),
    }

    # ── D3: Retrieval Score Drop ──────────────────────────────────────
    high_ret = [s["score"] for s in scored if s["ret_score"] >= ret_median]
    low_ret  = [s["score"] for s in scored if s["ret_score"] <  ret_median and s["ret_score"] > 0]
    d3 = {
        "label":     "D3 Retrieval Score",
        "baseline":  f"High retrieval score (>= {ret_median:.4f})",
        "degraded":  f"Low retrieval score (< {ret_median:.4f})",
        "baseline_n": len(high_ret),
        "degraded_n": len(low_ret),
        "baseline_mean": round(mean(high_ret), 4),
        "degraded_mean": round(mean(low_ret), 4),
        "baseline_std":  round(std(high_ret), 4),
        "degraded_std":  round(std(low_ret), 4),
        "abs_drop":  round(mean(high_ret) - mean(low_ret), 4),
        "pct_drop":  pct_drop(mean(high_ret), mean(low_ret)),
        "cohens_d":  round(cohens_d(high_ret, low_ret), 4),
    }

    # ── D4: Topic Domain Drop ─────────────────────────────────────────
    topic_groups = defaultdict(list)
    for s in scored:
        topic_groups[s["topic"]].append(s["score"])
    topic_means = {t: mean(v) for t, v in topic_groups.items()}
    best_topic  = max(topic_means, key=topic_means.get)
    worst_topic = min(topic_means, key=topic_means.get)
    d4 = {
        "label":       "D4 Topic Domain",
        "topic_stats": {t: {"n": len(v), "mean": round(mean(v), 4),
                            "std": round(std(v), 4)} for t, v in topic_groups.items()},
        "best_topic":  best_topic,
        "worst_topic": worst_topic,
        "abs_drop":    round(topic_means[best_topic] - topic_means[worst_topic], 4),
        "pct_drop":    pct_drop(topic_means[best_topic], topic_means[worst_topic]),
        "cohens_d":    round(cohens_d(topic_groups[best_topic],
                                      topic_groups[worst_topic]), 4),
    }

    # ── D5: Answer Length Drop ────────────────────────────────────────
    long_scores  = [s["score"] for s in scored if s["answer_len"] >= mean_len]
    short_scores = [s["score"] for s in scored if s["answer_len"] <  mean_len]
    d5 = {
        "label":     "D5 Answer Length",
        "baseline":  f"Long answers (>= {mean_len:.1f} words)",
        "degraded":  f"Short answers (< {mean_len:.1f} words)",
        "baseline_n": len(long_scores),
        "degraded_n": len(short_scores),
        "baseline_mean": round(mean(long_scores), 4),
        "degraded_mean": round(mean(short_scores), 4),
        "baseline_std":  round(std(long_scores), 4),
        "degraded_std":  round(std(short_scores), 4),
        "abs_drop":  round(mean(long_scores) - mean(short_scores), 4),
        "pct_drop":  pct_drop(mean(long_scores), mean(short_scores)),
        "cohens_d":  round(cohens_d(long_scores, short_scores), 4),
    }

    return {
        "scored":      scored,
        "all_scores":  all_scores,
        "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
        "n":           len(scored),
        "mean_len":    round(mean_len, 2),
        "ret_median":  round(ret_median, 4),
    }


# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = results["n"]

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Performance Drop Analysis")
    lines.append("  Script  : performance_drop3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("COMPOSITE QUALITY SCORE FORMULA")
    lines.append("-" * 70)
    lines.append("  Score = 0.30 * Q1 + 0.30 * Q2 + 0.25 * Q3 + 0.15 * Q4")
    lines.append("  Q1  Factual Density    -- quantitative facts per 100 words (x10)")
    lines.append("  Q2  Term Precision     -- biomedical term coverage (x3 scaled)")
    lines.append("  Q3  Relevance Overlap  -- question-answer token overlap %")
    lines.append("  Q4  Structural Clarity -- sentence count coherence proxy")
    lines.append("  All components scaled 0-100; final score 0-100.")
    lines.append("")

    lines.append("PERFORMANCE DROP AXES")
    lines.append("-" * 70)
    lines.append("  D1  Question Complexity   -- simple vs complex questions")
    lines.append("  D2  Question Type         -- factual vs reasoning questions")
    lines.append("  D3  Retrieval Score       -- high vs low retrieval confidence")
    lines.append("  D4  Topic Domain          -- best vs worst performing topic")
    lines.append("  D5  Answer Length         -- long vs short answers")
    lines.append("")
    lines.append("  Effect size (Cohen's d):  Large>=0.80  Medium>=0.50  Small>=0.20")
    lines.append("")

    # Per-question score table
    lines.append("PER-QUESTION COMPOSITE SCORES")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'Score':>7}  {'Type':<10}  {'Topic':<24}  "
                 f"{'Len':>5}  {'RetScr':>7}")
    lines.append("  " + "-" * 62)
    for s in results["scored"]:
        lines.append(
            f"  {s['q_num']:>4}  {s['score']:>7.3f}  {s['q_type']:<10}  "
            f"{s['topic']:<24}  {s['answer_len']:>5}  {s['ret_score']:>7.4f}"
        )

    # Overall score stats
    all_scores = results["all_scores"]
    lines.append("")
    lines.append("OVERALL SCORE STATISTICS")
    lines.append("-" * 70)
    lines.append(f"  Mean   : {mean(all_scores):.4f}")
    lines.append(f"  Std    : {std(all_scores):.4f}")
    lines.append(f"  Median : {median(all_scores):.4f}")
    lines.append(f"  Min    : {min(all_scores):.4f}")
    lines.append(f"  Max    : {max(all_scores):.4f}")
    lines.append("")

    # Score distribution
    buckets = [
        ("90-100", 90, 101), ("75-89",  75,  90),
        ("60-74",  60,  75), ("45-59",  45,  60),
        ("< 45",    0,  45),
    ]
    lines.append("SCORE DISTRIBUTION")
    lines.append("-" * 70)
    for label, lo, hi in buckets:
        count = sum(1 for s in all_scores if lo <= s < hi)
        bar   = "#" * int(count / n * 40)
        lines.append(f"  {label:<8} : {count:>4} ({count/n*100:>5.1f}%)  {bar}")
    lines.append("")

    # D1 - D5 drop tables
    for axis_key in ["d1", "d2", "d3", "d5"]:
        d = results[axis_key]
        lines.append(f"{d['label']} -- Performance Drop")
        lines.append("-" * 70)
        lines.append(f"  Baseline : {d['baseline']}  (n={d['baseline_n']})")
        lines.append(f"  Degraded : {d['degraded']}  (n={d['degraded_n']})")
        lines.append(f"  Baseline mean score : {d['baseline_mean']:.4f}  "
                     f"(std={d['baseline_std']:.4f})")
        lines.append(f"  Degraded mean score : {d['degraded_mean']:.4f}  "
                     f"(std={d['degraded_std']:.4f})")
        lines.append(f"  Absolute drop       : {d['abs_drop']:.4f} points")
        lines.append(f"  Relative drop       : {d['pct_drop']:.2f}%")
        lines.append(f"  Cohen's d           : {d['cohens_d']:.4f}  "
                     f"({effect_size_label(d['cohens_d'])})")
        direction = "DEGRADED" if d['abs_drop'] > 0 else "IMPROVED"
        lines.append(f"  Verdict             : {direction}")
        lines.append("")

    # D4 topic domain -- special table
    d4 = results["d4"]
    lines.append("D4 Topic Domain -- Performance Drop")
    lines.append("-" * 70)
    lines.append(f"  {'Topic':<26} {'N':>4}  {'Mean':>8}  {'Std':>8}")
    lines.append("  " + "-" * 50)
    for topic, stats in sorted(d4["topic_stats"].items(),
                                key=lambda x: -x[1]["mean"]):
        marker = " <-- BEST " if topic == d4["best_topic"]  else \
                 " <-- WORST" if topic == d4["worst_topic"] else ""
        lines.append(f"  {topic:<26} {stats['n']:>4}  "
                     f"{stats['mean']:>8.4f}  {stats['std']:>8.4f}{marker}")
    lines.append("")
    lines.append(f"  Best topic  : {d4['best_topic']}")
    lines.append(f"  Worst topic : {d4['worst_topic']}")
    lines.append(f"  Absolute drop : {d4['abs_drop']:.4f} points")
    lines.append(f"  Relative drop : {d4['pct_drop']:.2f}%")
    lines.append(f"  Cohen's d     : {d4['cohens_d']:.4f}  "
                 f"({effect_size_label(d4['cohens_d'])})")
    lines.append("")

    # Drop summary comparison table
    lines.append("PERFORMANCE DROP SUMMARY (all axes)")
    lines.append("-" * 70)
    lines.append(f"  {'Axis':<26} {'AbsDrop':>9} {'PctDrop':>9} "
                 f"{'CohenD':>8} {'Effect':<12} {'Verdict'}")
    lines.append("  " + "-" * 72)

    for axis_key in ["d1", "d2", "d3", "d5"]:
        d = results[axis_key]
        verdict = "DEGRADED" if d["abs_drop"] > 0 else "IMPROVED"
        lines.append(
            f"  {d['label']:<26} {d['abs_drop']:>9.4f} "
            f"{d['pct_drop']:>8.2f}% {d['cohens_d']:>8.4f} "
            f"{effect_size_label(d['cohens_d']):<12} {verdict}"
        )

    d4 = results["d4"]
    lines.append(
        f"  {'D4 Topic Domain':<26} {d4['abs_drop']:>9.4f} "
        f"{d4['pct_drop']:>8.2f}% {d4['cohens_d']:>8.4f} "
        f"{effect_size_label(d4['cohens_d']):<12} DEGRADED"
    )

    # Largest drop axis
    drop_vals = {
        "D1": results["d1"]["abs_drop"],
        "D2": results["d2"]["abs_drop"],
        "D3": results["d3"]["abs_drop"],
        "D4": results["d4"]["abs_drop"],
        "D5": results["d5"]["abs_drop"],
    }
    largest_drop_axis  = max(drop_vals, key=drop_vals.get)
    smallest_drop_axis = min(drop_vals, key=drop_vals.get)

    lines.append("")
    lines.append(f"  Largest performance drop  : {largest_drop_axis} "
                 f"({drop_vals[largest_drop_axis]:.4f} pts)")
    lines.append(f"  Smallest performance drop : {smallest_drop_axis} "
                 f"({drop_vals[smallest_drop_axis]:.4f} pts)")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  D2 (Question Type) drop reflects BiomedBERT retrieval's")
    lines.append("  known weakness: semantic similarity does not distinguish")
    lines.append("  reasoning questions from factual ones, so the same passages")
    lines.append("  are retrieved regardless of whether the question needs")
    lines.append("  a fact or a causal explanation.")
    lines.append("")
    lines.append("  D3 (Retrieval Score) drop quantifies how much answer quality")
    lines.append("  depends on retrieval confidence. A large drop here means the")
    lines.append("  system is brittle when BiomedBERT retrieval is less certain.")
    lines.append("")
    lines.append("  D4 (Topic Domain) reveals which question domains the KB")
    lines.append("  covers well vs poorly. Statistical/metric questions are")
    lines.append("  expected to show the largest drop since the KB passages")
    lines.append("  focus on biology rather than computational methodology.")
    lines.append("")
    lines.append("  D1 and D5 measure structural robustness -- a large D1 drop")
    lines.append("  means longer/harder questions degrade quality significantly,")
    lines.append("  and a large D5 drop means shorter answers sacrifice content.")
    lines.append("")
    lines.append("  Loading BioMistral-7B as the generator is expected to")
    lines.append("  reduce D2 and D4 drops by generating question-type-aware")
    lines.append("  answers rather than uniform passage concatenations.")
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
    print("[INFO] Running performance drop analysis...\n")
    results = run_performance_drop(predictions)
    write_output(results, OUT_PATH)