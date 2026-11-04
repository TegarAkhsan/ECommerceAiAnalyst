# AI Agent Analisis E-commerce
## Cara Deploy ke Streamlit Cloud

### Struktur repo yang harus ada di GitHub:
```
repo/
├── streamlit_app.py
├── requirements.txt
├── model_xgb_tuned.pkl          ← dari hasil training (03b)
├── scaler.pkl                   ← dari hasil training (02c)
├── feature_columns.json         ← dari hasil training (02c)
├── ecommerce_cleaned.csv        ← dari hasil cleaning (01)
└── .streamlit/
    └── secrets.toml             ← JANGAN di-commit, isi di Streamlit Cloud
```

### Langkah deploy:
1. Upload semua file ke GitHub (kecuali secrets.toml)
2. Buka https://share.streamlit.io → New app → pilih repo
3. Main file path: `streamlit_app.py`
4. Klik Advanced → Secrets → isi OPENAI_API_KEY dan TAVILY_API_KEY
5. Deploy

### Cara test lokal:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
