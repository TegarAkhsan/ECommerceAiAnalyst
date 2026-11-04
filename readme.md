# Agentic AI and MSME Sales Forecasting: Code and Evaluation Artifacts

Code, test queries, system prompt, and evaluation logs for the JISI manuscript.

## Contents
- agent_core_v2.py: agent, tools, sandbox
- eval_metrics.py: scoring functions
- eval_agentic_ai_v2.py: evaluation harness (28 queries)
- eval_requirements.txt: dependencies
- system_prompt_v2.txt: agent system prompt
- ecommerce_cleaned.csv: cleaned data (derived from public Kaggle dataset, CC BY-SA 4.0)
- hasil_eval_v2/, hasil_eval_v3_run2/: logs and summary metrics for two runs
- Supplementary_Material_Rev26_v2.docx: query protocol, scoring rules, configuration

## Data
Raw data: https://www.kaggle.com/datasets/bakitacos/indonesia-e-commerce-sales-and-shipping-20232025

## Reproduction
pip install -r eval_requirements.txt
python eval_agentic_ai_v2.py --data ecommerce_cleaned.csv --raw-dir <path to raw files> --out results --check-urls

Requires a Google Gemini API key set as GOOGLE_API_KEY. The key is not included.