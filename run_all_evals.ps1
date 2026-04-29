# BMEN-499 AlphaFold -- Run All Evaluations
# Run from repo root: .\run_all_evals.ps1

$env:PYTHONIOENCODING="utf-8"
$DISPROT = "Data\Baseline\DisProt_ProteinData.json"
$QA      = "Data\QA_Dataset.json"

# ── GUARDRAIL (runs first) ────────────────────────────────────
Write-Host "Running BioGPT guardrail..."
python biogpt_guardrail.py --disprot $DISPROT --qa $QA

# ── LLM JUDGE 1 ───────────────────────────────────────────────
Write-Host "Running LLM Judge 1 evaluations..."
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\naur_llm1.py"                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\contradiction_count.py"          --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\cosine_similarity1.py"           --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\output_variance1.py"             --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\explanability\agreement_score.py"            --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\explanability\likert_score1.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\relability\error_rate1.py"                   --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\relability\preformance_drop1.py"             --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\bert_score1.py"                              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\fact_score1.py"                              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_a.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_b.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_c.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_d.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_e.py"              --disprot $DISPROT --qa $QA

# ── LLM JUDGE 2 ───────────────────────────────────────────────
Write-Host "Running LLM Judge 2 evaluations..."
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\contradiction_count_2.py"         --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\cosine_similarity_2.py"           --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\variance_output.py"               --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\agreement_score_2.py"           --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\likert_score_2.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\error_rate2.py"                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\performance_drop2.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\Custom Evals\BERT_score2.py"                                  --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\Custom Evals\fact_score2.py"                                  --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\K pass tests\K_test2_a.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\K pass tests\K_test2_b.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\K pass tests\K_test3_c.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\K pass tests\K_test2_d.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\K pass tests\K_test2_e.py"                                    --disprot $DISPROT --qa $QA

# ── LLM JUDGE 3 ───────────────────────────────────────────────
Write-Host "Running LLM Judge 3 evaluations..."
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\contradiction_count3.py"               --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\cosine_similarity3.py"                 --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\output_variance3.py"                   --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\explainability\agreement_score3.py"                --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\explainability\likert_score3.py"                   --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\relability\error_rate3.py"                         --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\relability\preformance_drop3.py"                   --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\Custom Evals\BERT_score3.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\Custom Evals\FACT_score3.py"                                    --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_a.py"                                      --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_b.py"                                      --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_c.py"                                      --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_d.py"                                      --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_e.py"                                      --disprot $DISPROT --qa $QA

Write-Host "All evaluations complete."