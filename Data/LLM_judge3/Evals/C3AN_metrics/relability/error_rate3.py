"""
BMEN-499 AlphaFold -- LLM Judge 3: Error Rate Analysis
--------------------------------------------------------
File    : error_rate3.py
Output  : error_rate3_output.txt (same folder as this script)
Source  : LLM3_predictions.txt (BioMistral RAG, 100 questions)

What this script does:
    Computes error rates across eight error categories for each of
    the 100 predicted answers in LLM3_predictions.txt. No external
    dependencies -- pure stdlib rule-based detection.

Error Categories:
    E1  -- Factual Omission Error
           Answer fails to include any quantitative fact or named
           source when the question explicitly requires one.

    E2  -- Threshold Misapplication Error
           Answer cites a disorder threshold (0.3, 0.5, 0.7) but
           applies it to the wrong context or inverts its meaning.

    E3  -- pLDDT Inversion Error
           Answer confuses the directionality of pLDDT scores
           (e.g. implies high pLDDT = disordered, or low = ordered).

    E4  -- Retrieval-Question Mismatch Error
           Answer content is topically unrelated to the question
           (retriever pulled wrong passages).

    E5  -- Numeric Hallucination Error
           Answer contains a number that contradicts the known
           DisProt dataset statistics (13,396 proteins, 0.378 mean,
           29.1% above 0.5, 44.2% above 0.3).

    E6  -- Redundancy Error
           Answer repeats the same claim more than once verbatim
           or near-verbatim within a single response.

    E7  -- Incomplete Answer Error
           Answer is below minimum viable length (< 30 words) or
           contains no content beyond boilerplate retrieval text.

    E8  -- Confidence Miscalibration Error
           Answer makes an absolute claim ("always", "never",
           "definitively") about something that is inherently
           probabilistic or context-dependent in IDR biology.

Scoring:
    Each error is binary per question: 1 = error present, 0 = absent.
    Error rate = errors detected / total questions per category.
    Severity tiers:
        CRITICAL  : error rate >= 0.40
        HIGH      : error rate >= 0.25
        MODERATE  : error rate >= 0.10
        LOW       : error rate >= 0.01
        NONE      : error rate == 0.00

Output: error_rate3_output.txt
"""

import os
import re
import math
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM3_PATH  = r"C:\Users\Michelle Ihetu\OneDrive - University of South Carolina\Desktop\MIHETU\AI_Insitute_Work\BMEN 499\BMEN-499_AlphaFold\Data\LLM_judge3\LLM3_predictions.txt"
OUT_PATH   = os.path.join(SCRIPT_DIR, "error_rate3_output.txt")

# ── Known DisProt ground-truth constants ──────────────────────────────
DISPROT_N_PROTEINS   = 13396
DISPROT_MEAN_DISORDER = 0.378
DISPROT_PCT_ABOVE_05  = 29.1
DISPROT_PCT_ABOVE_03  = 44.2
NUMERIC_TOLERANCE     = 0.05   # 5% tolerance for numeric checks

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

# ── Error detection functions ─────────────────────────────────────────

def check_e1_factual_omission(question, answer):
    """
    E1: Question requires a fact but answer has no quantitative content.
    Triggered when question contains stat keywords but answer has no numbers.
    """
    stat_keywords = [
        "how many", "how much", "what percent", "what proportion",
        "what fraction", "how far", "how often", "rate", "score",
        "threshold", "cutoff", "mean", "average", "value", "number",
        "count", "frequency", "magnitude", "extent", "degree",
    ]
    q_lower   = question.lower()
    a_lower   = answer.lower()
    needs_stat = any(kw in q_lower for kw in stat_keywords)
    has_number = bool(re.search(r"\d+\.?\d*", a_lower))
    return needs_stat and not has_number


def check_e2_threshold_misapplication(answer):
    """
    E2: Threshold cited but applied in wrong direction or wrong context.
    Flags cases where 0.5 is called unreliable immediately after being
    described as the standard cutoff without qualification.
    """
    a = answer.lower()
    # Calls 0.5 the cutoff AND then immediately says it misses IDRs
    # without framing it as a limitation
    has_cutoff     = bool(re.search(r"0\.5.*threshold.*classif", a))
    has_miss       = bool(re.search(r"fall below.*0\.5|missed|miss the.*cutoff", a))
    no_limitation  = "however" not in a and "but" not in a and "limitation" not in a
    return has_cutoff and has_miss and no_limitation


def check_e3_plddt_inversion(answer):
    """
    E3: pLDDT directionality confused.
    Flags if answer implies pLDDT >= 70 correlates with disorder,
    or pLDDT < 50 correlates with structure.
    """
    a = answer.lower()
    inversion_patterns = [
        r"plddt.*(?:above|>=|>|high).*(?:70|80|90).*disorder",
        r"plddt.*(?:below|<|low).*(?:50|40|30).*(?:structured|ordered|folded)",
        r"high.*plddt.*disordered",
        r"low.*plddt.*(?:structured|folded|ordered)",
    ]
    return any(re.search(pat, a) for pat in inversion_patterns)


def check_e4_retrieval_mismatch(question, answer):
    """
    E4: Answer is topically mismatched with the question.
    Uses token overlap -- if < 10% of question keywords appear in answer.
    """
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","are","was","were","be","been",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","this","that","these","those","it",
        "its","as","not","no","so","if","than","then","when","which",
        "who","what","how","also","more","most","after","before",
        "between","into","through","while","both","each","further",
        "once","above","below","up","down","out","about","over",
        "under","again","here","there","where","why","all","any",
        "few","very","just","because","such","only","their","they",
        "them","we","our","can","cannot","per","i","mean","well",
        "does","do","did","across","without","whether","among",
    }
    q_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", question.lower()).split()
        if t not in stopwords and len(t) > 2
    )
    a_tokens = set(
        t.lower() for t in re.sub(r"[^a-z0-9\s]", " ", answer.lower()).split()
        if t not in stopwords and len(t) > 2
    )
    if not q_tokens:
        return False
    overlap_rate = len(q_tokens & a_tokens) / len(q_tokens)
    return overlap_rate < 0.10


def check_e5_numeric_hallucination(answer):
    """
    E5: Answer contains a number that contradicts DisProt ground truth.
    Checks protein count, mean disorder, and percentage thresholds.
    """
    a = answer.lower()

    # Check protein count -- should be ~13,396
    protein_counts = re.findall(r"([\d,]+)\s*(?:disp?rot\s*)?proteins?", a)
    for pc_str in protein_counts:
        pc = int(pc_str.replace(",", ""))
        if pc > 100:   # filter out small counts like "3 proteins"
            expected = DISPROT_N_PROTEINS
            if abs(pc - expected) / expected > NUMERIC_TOLERANCE:
                return True

    # Check mean disorder value -- should be ~0.378
    mean_vals = re.findall(r"mean\s*[=:]\s*(0\.\d+)", a)
    for mv_str in mean_vals:
        mv = float(mv_str)
        if abs(mv - DISPROT_MEAN_DISORDER) > 0.05:
            return True

    # Check pct above 0.5 -- should be ~29.1%
    pct_05 = re.findall(r"([\d.]+)%?\s*exceed.*0\.5", a)
    for p_str in pct_05:
        p = float(p_str)
        if abs(p - DISPROT_PCT_ABOVE_05) > 5.0:
            return True

    return False


def check_e6_redundancy(answer):
    """
    E6: Answer repeats a substantive phrase (>= 8 tokens) more than once.
    """
    tokens  = answer.lower().split()
    n       = len(tokens)
    window  = 8
    seen    = set()
    for i in range(n - window):
        phrase = " ".join(tokens[i:i + window])
        if phrase in seen:
            return True
        seen.add(phrase)
    return False


def check_e7_incomplete(answer):
    """
    E7: Answer is too short or contains only boilerplate.
    """
    word_count     = len(answer.split())
    boilerplate    = [
        "no relevant passages found",
        "i don't know",
        "i cannot answer",
        "not enough information",
    ]
    is_boilerplate = any(b in answer.lower() for b in boilerplate)
    return word_count < 30 or is_boilerplate


def check_e8_confidence_miscalibration(answer):
    """
    E8: Answer makes absolute claims about probabilistic IDR biology.
    """
    a = answer.lower()
    absolute_patterns = [
        r"\balways\b",
        r"\bnever\b",
        r"\bdefinitively\b",
        r"\bcertainly\b",
        r"\bproves?\b",
        r"\bguarantees?\b",
        r"\bwithout exception\b",
        r"\bin all cases\b",
        r"\buniversally\b",
        r"\babsolutely\b",
    ]
    has_absolute = any(re.search(pat, a) for pat in absolute_patterns)
    # Only flag if the absolute claim is about disorder/structure
    disorder_context = any(kw in a for kw in [
        "disorder", "plddt", "idr", "structured", "folded", "predict"
    ])
    return has_absolute and disorder_context


# ── Error severity label ──────────────────────────────────────────────

def severity_label(rate):
    if rate >= 0.40:   return "CRITICAL "
    elif rate >= 0.25: return "HIGH     "
    elif rate >= 0.10: return "MODERATE "
    elif rate > 0.00:  return "LOW      "
    else:              return "NONE     "


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


# ── Run error rate analysis ───────────────────────────────────────────

def run_error_rate(predictions):
    per_q   = []
    totals  = {f"e{i}": 0 for i in range(1, 9)}

    for p in predictions:
        q   = p["question"]
        a   = p["answer"]

        e1  = int(check_e1_factual_omission(q, a))
        e2  = int(check_e2_threshold_misapplication(a))
        e3  = int(check_e3_plddt_inversion(a))
        e4  = int(check_e4_retrieval_mismatch(q, a))
        e5  = int(check_e5_numeric_hallucination(a))
        e6  = int(check_e6_redundancy(a))
        e7  = int(check_e7_incomplete(a))
        e8  = int(check_e8_confidence_miscalibration(a))

        errors      = [e1, e2, e3, e4, e5, e6, e7, e8]
        total_errs  = sum(errors)

        for i, e in enumerate(errors, 1):
            totals[f"e{i}"] += e

        per_q.append({
            "q_num":    p["q_num"],
            "question": p["question"],
            "e1": e1, "e2": e2, "e3": e3, "e4": e4,
            "e5": e5, "e6": e6, "e7": e7, "e8": e8,
            "total":    total_errs,
        })

    n       = len(predictions)
    rates   = {k: round(v / n, 4) for k, v in totals.items()}
    overall = sum(totals.values())

    return {
        "per_q":    per_q,
        "totals":   totals,
        "rates":    rates,
        "overall":  overall,
        "overall_rate": round(overall / (n * 8), 4),
        "n":        n,
    }


# ── Write output ──────────────────────────────────────────────────────

def write_output(results, out_path):
    lines = []
    n     = results["n"]

    lines.append("=" * 70)
    lines.append("  BMEN-499 AlphaFold -- LLM Judge 3: Error Rate Analysis")
    lines.append("  Script  : error_rate3.py")
    lines.append("  Source  : LLM3_predictions.txt (BioMistral RAG)")
    lines.append(f"  Questions analyzed : {n}")
    lines.append(f"  Error categories   : 8")
    lines.append("=" * 70)
    lines.append("")

    lines.append("ERROR CATEGORY DEFINITIONS")
    lines.append("-" * 70)
    lines.append("  E1  Factual Omission        -- stat question answered without numbers")
    lines.append("  E2  Threshold Misapplication -- threshold cited but context inverted")
    lines.append("  E3  pLDDT Inversion          -- pLDDT directionality confused")
    lines.append("  E4  Retrieval-Q Mismatch     -- answer topically unrelated to question")
    lines.append("  E5  Numeric Hallucination    -- number contradicts DisProt ground truth")
    lines.append("  E6  Redundancy               -- substantive phrase repeated in answer")
    lines.append("  E7  Incomplete Answer        -- answer < 30 words or boilerplate only")
    lines.append("  E8  Confidence Miscalibration-- absolute claim about probabilistic biology")
    lines.append("")
    lines.append("  Severity: CRITICAL>=0.40  HIGH>=0.25  MODERATE>=0.10  LOW>0.00  NONE=0.00")
    lines.append("")

    # Per-question table
    lines.append("PER-QUESTION ERROR FLAGS  (1=error present, 0=absent)")
    lines.append("-" * 70)
    lines.append(f"  {'Q':>4}  {'E1':>3} {'E2':>3} {'E3':>3} {'E4':>3} "
                 f"{'E5':>3} {'E6':>3} {'E7':>3} {'E8':>3}  {'Total':>6}")
    lines.append("  " + "-" * 48)

    for p in results["per_q"]:
        lines.append(
            f"  {p['q_num']:>4}  "
            f"{p['e1']:>3} {p['e2']:>3} {p['e3']:>3} {p['e4']:>3} "
            f"{p['e5']:>3} {p['e6']:>3} {p['e7']:>3} {p['e8']:>3}  "
            f"{p['total']:>6}"
        )

    # Error rate summary table
    lines.append("")
    lines.append("ERROR RATE SUMMARY TABLE")
    lines.append("-" * 70)
    lines.append(f"  {'Code':<4} {'Description':<34} {'Count':>6} {'Rate':>7} {'Severity':<12}")
    lines.append("  " + "-" * 66)

    error_info = [
        ("e1", "Factual Omission"),
        ("e2", "Threshold Misapplication"),
        ("e3", "pLDDT Inversion"),
        ("e4", "Retrieval-Question Mismatch"),
        ("e5", "Numeric Hallucination"),
        ("e6", "Redundancy"),
        ("e7", "Incomplete Answer"),
        ("e8", "Confidence Miscalibration"),
    ]

    for key, desc in error_info:
        code  = key.upper()
        count = results["totals"][key]
        rate  = results["rates"][key]
        sev   = severity_label(rate)
        lines.append(f"  {code:<4} {desc:<34} {count:>6} {rate:>7.4f} {sev}")

    # Overall
    lines.append("")
    lines.append(f"  Overall error flag rate : {results['overall_rate']:.4f}  "
                 f"({results['overall']} flags / {n * 8} possible)")
    lines.append(f"  Total error flags       : {results['overall']}")

    # Distribution of total errors per question
    lines.append("")
    lines.append("DISTRIBUTION: ERRORS PER QUESTION")
    lines.append("-" * 70)
    total_counts = Counter(p["total"] for p in results["per_q"])
    for k in range(9):
        count = total_counts.get(k, 0)
        bar   = "#" * int(count / n * 40)
        lines.append(f"  {k} errors : {count:>4}  ({count/n*100:>5.1f}%)  {bar}")

    # Most and least error-prone questions
    sorted_per_q = sorted(results["per_q"], key=lambda p: -p["total"])

    lines.append("")
    lines.append("TOP 5 MOST ERROR-PRONE QUESTIONS")
    lines.append("-" * 70)
    for p in sorted_per_q[:5]:
        active = [f"E{i}" for i in range(1, 9) if p[f"e{i}"] == 1]
        lines.append(f"  Q{p['q_num']:>3}  {p['total']} errors  [{', '.join(active)}]")
        lines.append(f"       \"{p['question'][:60]}\"")

    lines.append("")
    lines.append("TOP 5 CLEANEST QUESTIONS (fewest errors)")
    lines.append("-" * 70)
    for p in sorted_per_q[-5:]:
        active = [f"E{i}" for i in range(1, 9) if p[f"e{i}"] == 1]
        flag_str = ', '.join(active) if active else "none"
        lines.append(f"  Q{p['q_num']:>3}  {p['total']} errors  [{flag_str}]")
        lines.append(f"       \"{p['question'][:60]}\"")

    # Stats on total errors per question
    total_errs_list = [p["total"] for p in results["per_q"]]
    lines.append("")
    lines.append("DESCRIPTIVE STATISTICS: ERRORS PER QUESTION")
    lines.append("-" * 70)
    lines.append(f"  Mean   : {mean(total_errs_list):.4f}")
    lines.append(f"  Std    : {std(total_errs_list):.4f}")
    lines.append(f"  Median : {median(total_errs_list):.4f}")
    lines.append(f"  Min    : {min(total_errs_list)}")
    lines.append(f"  Max    : {max(total_errs_list)}")

    # Questions with zero errors
    zero_error = [p for p in results["per_q"] if p["total"] == 0]
    lines.append(f"  Zero-error questions : {len(zero_error)} / {n}  "
                 f"({len(zero_error)/n*100:.1f}%)")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("  E6 (Redundancy) is expected to be the most frequent error in")
    lines.append("  extractive fallback mode because the same KB passages are")
    lines.append("  retrieved repeatedly across questions, causing identical phrases")
    lines.append("  to appear in multiple answers and sometimes within one answer.")
    lines.append("")
    lines.append("  E4 (Retrieval-Question Mismatch) flags cases where BiomedBERT")
    lines.append("  retrieves semantically similar but question-mismatched passages.")
    lines.append("  This is the core weakness of dense retrieval without re-ranking.")
    lines.append("")
    lines.append("  E3 (pLDDT Inversion) should be near zero since the KB passages")
    lines.append("  correctly state pLDDT directionality. Any flags here indicate")
    lines.append("  a retrieval combination that creates an implicit contradiction.")
    lines.append("")
    lines.append("  E5 (Numeric Hallucination) should also be low since answers")
    lines.append("  are extracted directly from the knowledge base built from the")
    lines.append("  actual DisProt dataset statistics.")
    lines.append("")
    lines.append("  Loading BioMistral-7B would primarily reduce E4 (by generating")
    lines.append("  question-specific answers) and E6 (by synthesizing rather than")
    lines.append("  concatenating passages), at the cost of potentially increasing")
    lines.append("  E5 and E8 if the model generates rather than extracts facts.")
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
    print("[INFO] Running error rate analysis...\n")
    results = run_error_rate(predictions)
    write_output(results, OUT_PATH)