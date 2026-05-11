"""
BMEN-499 -- LLM Rules Patcher
------------------------------
Updates all LLM2 and LLM3 evaluation scripts to use the correct
LLM rules instead of LLM1 rules.

Run from repo root:
    python patch_llm_rules.py

What it does:
    - Finds LLM1_RULES blocks in each file
    - Replaces them with the correct LLM2 or LLM3 rules
    - Saves each file in place
    - Reports which files were updated
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── LLM2 Rules (Vanilla RAG) ──────────────────────────────────
LLM2_RULES = '''
LLM_RULES = [
    (["0.5","cutoff","disorder"],
     lambda s: f"The 0.5 disorder score threshold classifies protein regions as intrinsically disordered. Of {s['total_proteins']:,} DisProt proteins {s['pct_above_0.5']:.1f}% exceed this threshold with mean disorder score {s['mean_disorder']:.3f}. However {s['pct_above_0.3']:.1f}% exceed 0.3 meaning many true IDRs fall below 0.5 and are missed. Disorder scores between 0.3 and 0.5 define an ambiguous gray zone where proteins cannot be confidently classified without secondary validation."),
    (["short","residue"],
     lambda s: f"Of {s['total_regions']:,} DisProt regions {s['pct_short_regions']:.1f}% are shorter than 10 residues with mean {s['mean_region_length']:.1f} aa. Short IDRs are hard to predict reliably due to limited sequence context. Sliding window averaging smooths per-residue disorder scores but windows larger than mean region length risk smoothing out short IDRs entirely."),
    (["proline","glycine"],
     lambda s: f"Proline content DisProt mean {s['mean_proline']*100:.1f}% strongly predicts intrinsic disorder. Proline rigid pyrrolidine ring disrupts alpha-helices and beta-sheets preventing regular secondary structure. Glycine mean {s['mean_glycine']*100:.1f}% adds conformational freedom. Elevated proline and glycine together form a strong composite disorder signal."),
    (["sliding","window"],
     lambda s: f"Sliding window averaging smooths per-residue disorder scores to reduce noise. The mean disordered region length in DisProt is {s['mean_region_length']:.1f} amino acids. If sliding window size exceeds this mean short disordered regions risk being averaged out and lost entirely."),
    (["pfam","domain"],
     lambda s: f"{s['pct_with_pfam']:.1f}% of DisProt proteins contain Pfam domains alongside disordered regions confirming IDRs and structured domains frequently co-occur. Each region must be evaluated independently. Proteins with no Pfam domains and disorder content above 0.5 are classified as intrinsically disordered proteins IDPs."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT below 50 strongly indicates intrinsic disorder. DisProt annotated disordered regions in {s['total_proteins']:,} proteins consistently show pLDDT below 50 the most reliable computational signal. pLDDT scores of 50 to 70 indicate ambiguous structure possibly conditionally disordered MoRF regions."),
]
'''

# ── LLM3 Rules (BioMistral RAG) ───────────────────────────────
LLM3_RULES = '''
LLM_RULES = [
    (["0.5","cutoff","disorder"],
     lambda s: f"Disorder scores quantify structural flexibility in proteins. Based on {s['total_proteins']:,} DisProt entries {s['pct_above_0.5']:.1f}% exceed the 0.5 threshold with mean {s['mean_disorder']:.3f}. The threshold is conservative as {s['pct_above_0.3']:.1f}% exceed 0.3. A lower cutoff of 0.3 to 0.4 may better capture biologically relevant IDRs in functional disordered regions."),
    (["short","residue"],
     lambda s: f"Short disordered regions challenge computational prediction. Among {s['total_regions']:,} DisProt regions {s['pct_short_regions']:.1f}% are below 10 residues with mean {s['mean_region_length']:.1f} aa. Sub-10 residue IDRs have lower prediction confidence due to insufficient sequence context despite their role in molecular recognition events."),
    (["proline","glycine"],
     lambda s: f"Compositional biases characterize disordered proteins. Mean proline is {s['mean_proline']*100:.1f}% and glycine is {s['mean_glycine']*100:.1f}% in {s['total_proteins']:,} DisProt proteins. Proline imposes steric constraints while glycine maximizes conformational entropy. Co-enrichment of both amino acids reliably predicts intrinsic disorder across multiple algorithms."),
    (["sliding","window"],
     lambda s: f"Sliding window smoothing reduces per-residue disorder score noise. With mean DisProt region length of {s['mean_region_length']:.1f} aa windows exceeding this value risk masking genuine short IDR signals. Empirical studies suggest windows of 5 to 9 residues preserve biological signal while smoothing random scoring fluctuations."),
    (["pfam","domain"],
     lambda s: f"Co-occurrence of Pfam domains and disordered regions characterizes {s['pct_with_pfam']:.1f}% of DisProt proteins. Ordered globular domains and flexible disordered linkers are not mutually exclusive. Hub proteins and transcription factors frequently contain both domain types requiring region-level independent structural evaluation."),
    (["alphafold","plddt"],
     lambda s: f"AlphaFold pLDDT scores below 50 reliably identify disordered regions in {s['total_proteins']:,} DisProt proteins. Scores of 50 to 70 represent transitional regions exhibiting conditional disorder that fold upon binding partner interaction. These molecular recognition features require experimental validation beyond computational prediction."),
]
'''

# ── Patterns to search and replace ───────────────────────────

# These patterns match any existing LLM rules variable name
LLM_RULES_PATTERN = re.compile(
    r'(LLM[123]?_RULES\s*=\s*\[.*?\n\])',
    re.DOTALL
)

# Also match the generic LLM_RULES name
LLM_RULES_PATTERN2 = re.compile(
    r'(LLM_RULES\s*=\s*\[.*?\n\])',
    re.DOTALL
)

# get_answer calls to update
GET_ANSWER_LLM1 = re.compile(r'get_answer\(q,\s*LLM1_RULES,\s*stats\)')
GET_ANSWER_GT   = re.compile(r'get_answer\(q,\s*GT_RULES,\s*stats\)')


def patch_file(filepath, new_rules, judge_label):
    """
    Open a file, replace LLM rules, update get_answer calls,
    and save in place.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    original = content

    # Replace any existing LLM rules block
    replaced = False
    for pattern in [LLM_RULES_PATTERN, LLM_RULES_PATTERN2]:
        if pattern.search(content):
            content = pattern.sub(new_rules.strip(), content, count=1)
            replaced = True
            break

    if not replaced:
        # No existing LLM_RULES block found -- inject after GT_RULES
        gt_end = content.find(']', content.find('GT_RULES'))
        if gt_end != -1:
            insert_pos = gt_end + 1
            content = content[:insert_pos] + '\n' + new_rules + content[insert_pos:]
            replaced = True

    # Update get_answer calls to use LLM_RULES instead of LLM1_RULES
    content = GET_ANSWER_LLM1.sub('get_answer(q, LLM_RULES, stats)', content)

    # Update label in docstring/comments if present
    content = content.replace('LLM Judge 1', judge_label)
    content = content.replace('LLM1_RULES', 'LLM_RULES')
    content = content.replace('LLM2_RULES', 'LLM_RULES')
    content = content.replace('LLM3_RULES', 'LLM_RULES')

    if content != original:
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(content)
        return True
    return False


# ── File lists ────────────────────────────────────────────────

LLM2_FILES = [
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\contradiction_count_2.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\cosine_similarity_2.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\variance_output.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\agreement_score_2.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\likert_score_2.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\error_rate2.py",
    r"Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\performance_drop2.py",
    r"Data\LLM_judge2\Evals_2\Custom Evals\BERT_score2.py",
    r"Data\LLM_judge2\Evals_2\Custom Evals\fact_score2.py",
    r"Data\LLM_judge2\Evals_2\K pass tests\K_test2_a.py",
    r"Data\LLM_judge2\Evals_2\K pass tests\K_test2_b.py",
    r"Data\LLM_judge2\Evals_2\K pass tests\K_test3_c.py",
    r"Data\LLM_judge2\Evals_2\K pass tests\K_test2_d.py",
    r"Data\LLM_judge2\Evals_2\K pass tests\K_test2_e.py",
]

LLM3_FILES = [
    r"Data\LLM_judge3\Evals\C3AN_metrics\consistency\contradiction_count3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\consistency\cosine_similarity3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\consistency\output_variance3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\explainability\agreement_score3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\explainability\likert_score3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\relability\error_rate3.py",
    r"Data\LLM_judge3\Evals\C3AN_metrics\relability\preformance_drop3.py",
    r"Data\LLM_judge3\Evals\Custom Evals\BERT_score3.py",
    r"Data\LLM_judge3\Evals\Custom Evals\FACT_score3.py",
    r"Data\LLM_judge3\Evals\K pass tests\k_pass3_a.py",
    r"Data\LLM_judge3\Evals\K pass tests\k_pass3_b.py",
    r"Data\LLM_judge3\Evals\K pass tests\k_pass3_c.py",
    r"Data\LLM_judge3\Evals\K pass tests\k_pass3_d.py",
    r"Data\LLM_judge3\Evals\K pass tests\k_pass3_e.py",
]


# ── Main ──────────────────────────────────────────────────────

def main():
    updated = []
    skipped = []
    missing = []

    print("=" * 60)
    print("  BMEN-499 -- LLM Rules Patcher")
    print("=" * 60)

    # Patch LLM2 files
    print("\nPatching LLM2 files (Vanilla RAG rules)...")
    for rel_path in LLM2_FILES:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [MISSING] {rel_path}")
            missing.append(rel_path)
            continue
        result = patch_file(full_path, LLM2_RULES, "LLM Judge 2 (Vanilla RAG)")
        if result:
            print(f"  [UPDATED] {rel_path}")
            updated.append(rel_path)
        else:
            print(f"  [SKIPPED] {rel_path} (already correct or no LLM rules found)")
            skipped.append(rel_path)

    # Patch LLM3 files
    print("\nPatching LLM3 files (BioMistral RAG rules)...")
    for rel_path in LLM3_FILES:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [MISSING] {rel_path}")
            missing.append(rel_path)
            continue
        result = patch_file(full_path, LLM3_RULES, "LLM Judge 3 (BioMistral RAG)")
        if result:
            print(f"  [UPDATED] {rel_path}")
            updated.append(rel_path)
        else:
            print(f"  [SKIPPED] {rel_path} (already correct or no LLM rules found)")
            skipped.append(rel_path)

    print("\n" + "=" * 60)
    print(f"  Updated : {len(updated)} files")
    print(f"  Skipped : {len(skipped)} files")
    print(f"  Missing : {len(missing)} files")
    if missing:
        print("\n  Missing files (need to be created):")
        for f in missing:
            print(f"    {f}")
    print("\n  Run .\\run_all_evals.ps1 to rerun all evaluations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
