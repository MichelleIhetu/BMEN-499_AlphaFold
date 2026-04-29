# BMEN-499 AlphaFold -- Run LLM Judge 3 Evaluations Only
# Run from repo root: .\run_llm3_evals.ps1

$env:PYTHONIOENCODING="utf-8"
$DISPROT = "Data\Baseline\DisProt_ProteinData.json"
$QA      = "Data\QA_Dataset.json"

Write-Host "Running LLM Judge 3 (BioMistral RAG) evaluations..."

# C3AN Metrics -- Consistency
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\contradiction_count3.py"    --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\cosine_similarity3.py"      --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\consistency\output_variance3.py"        --disprot $DISPROT --qa $QA

# C3AN Metrics -- Explainability
python "Data\LLM_judge3\Evals\C3AN_metrics\explainability\agreement_score3.py"     --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\explainability\likert_score3.py"        --disprot $DISPROT --qa $QA

# C3AN Metrics -- Reliability
python "Data\LLM_judge3\Evals\C3AN_metrics\relability\error_rate3.py"              --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\C3AN_metrics\relability\preformance_drop3.py"        --disprot $DISPROT --qa $QA

# Custom Evals
python "Data\LLM_judge3\Evals\Custom Evals\BERT_score3.py"                         --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\Custom Evals\FACT_score3.py"                         --disprot $DISPROT --qa $QA

# K Pass Tests
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_a.py"                           --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_b.py"                           --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_c.py"                           --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_d.py"                           --disprot $DISPROT --qa $QA
python "Data\LLM_judge3\Evals\K pass tests\k_pass3_e.py"                           --disprot $DISPROT --qa $QA

Write-Host "LLM Judge 3 evaluations complete."
