$env:PYTHONIOENCODING="utf-8"
$DISPROT = "Data\Baseline\DisProt_ProteinData.json"
$QA      = "Data\QA_Dataset.json"

# STEP 1 -- Run guardrail first to filter BioGPT answers
Write-Host "Running BioGPT guardrail..."
python biogpt_guardrail.py --disprot $DISPROT --qa $QA

# STEP 2 -- LLM1 evals
Write-Host "Running LLM1 evaluations..."
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\naur_llm1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\contradiction_count.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\cosine_similarity1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\consistency\output_variance1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\explanability\agreement_score.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\explanability\likert_score1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\relability\error_rate1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\C3AN_metrics\relability\preformance_drop1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\bert_score1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\fact_score1.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_a.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_b.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_c.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_d.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge1\Judge1_evals\Custom_Evals\K_pass_test1\K_pass_test_e.py" --disprot $DISPROT --qa $QA

# STEP 3 -- LLM2 evals
Write-Host "Running LLM2 evaluations..."
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\contradiction_count2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\cosine_similarity2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\consistency2\output_variance2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\agreement_score_2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\explanability2\likert_score_2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\error_rate2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\C3AN_metrics_2\relability2\performance_drop2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\Custom Evals\BERT_score2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\Evals_2\Custom Evals\fact_score2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\K_pass_test2\K_pass_test_a2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\K_pass_test2\K_pass_test_b2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\K_pass_test2\K_pass_test_c2.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge2\K_pass_test2\K_pass_test_d2.py" --disprot $DISPROST --qa $QA
python "Data\LLM_judge2\K_pass_test2\K_pass_test_e2.py" --disprot $DISPROT --qa $QA

# STEP 4 -- LLM3 evals
Write-Host "Running LLM3 evaluations..."
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\consistency3\contradiction_count3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\consistency3\cosine_similarity3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\consistency3\output_variance3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\explanability3\agreement_score_3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\explanability3\likert_score_3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\relability3\error_rate3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\C3AN_metrics_3\relability3\performance_drop3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\Custom_Evals_3\bert_score3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals_3\Custom_Evals_3\fact_score3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\K_pass_test3\K_pass_test_a3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\K_pass_test3\K_pass_test_b3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\K_pass_test3\K_pass_test_c3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\K_pass_test3\K_pass_test_d3.py" --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\K_pass_test3\K_pass_test_e3.py" --disprot $DISPROT --qa $QA

Write-Host "All evaluations complete."