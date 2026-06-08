# ============================================================
# streamlit_app.py — AI Agent Analisis E-commerce v3
# ============================================================
# Perubahan v3:
#   1. Upload mendukung CSV dan XLSX (Excel) — langsung dari Shopee
#   2. Bisa upload multi-file sekaligus (semua 24 Excel sekaligus)
#   3. Model dan scaler diload dari file di folder project
#      (model_xgb_tuned.pkl, scaler.pkl, feature_columns.json)
#   4. Cleaning identik dengan 01_data_preparation.py
#   5. Forecast lag dibangun dari histori aktual (fix Rp 27 bug)
#   6. [FIX] Chatbot AI Agent: _init_agent() & _demo_response()
#      diimplementasikan lengkap dengan LangChain ReAct architecture
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import joblib, json, os, io, warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Agent E-commerce Analitik",
    page_icon="🤖", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Modern typography and premium styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Force Obsidian Theme on ALL Streamlit structural panels to override Light Theme splits */
    .stApp, 
    .main, 
    .stMain,
    section.main,
    [data-testid="stAppViewContainer"], 
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .block-container, 
    [data-testid="stHeader"],
    div[role="main"],
    div[data-testid="stDataFrame"], 
    div[data-testid="stTable"] {
        background-color: #080a0f !important;
        background-image: radial-gradient(circle at 85% 15%, rgba(0, 229, 255, 0.15) 0%, transparent 50%),
                          radial-gradient(circle at 15% 85%, rgba(250, 204, 21, 0.12) 0%, transparent 50%) !important;
        color: #f1f5f9 !important;
    }

    /* Sidebar Panel Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0f16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
    }
    
    /* Premium Custom Sidebar Brand Logo */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding: 10px 4px;
    }
    .logo-icon {
        width: 26px;
        height: 26px;
        background: radial-gradient(circle at 30% 30%, #00f0ff 0%, #facc15 65%, #000b33 100%);
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4), 0 0 15px rgba(250, 204, 21, 0.2), inset -3px -3px 6px rgba(0,0,0,0.6);
    }
    .logo-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar Radio Buttons -> Hide default circle bullets safely */
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label p {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    /* Style stRadio label wrapper as a sleek button card */
    div[data-testid="stRadio"] [role="radiogroup"] label {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #94a3b8 !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(0, 229, 255, 0.3) !important;
        color: #ffffff !important;
        padding-left: 20px !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.16) 0%, rgba(250, 204, 21, 0.04) 100%) !important;
        border-color: #00e5ff !important;
        color: #00e5ff !important;
        font-weight: 600 !important;
        border-left: 4px solid #00e5ff !important;
        border-right: 4px solid #facc15 !important;
        border-radius: 8px !important;
        padding-left: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.1), 0 0 10px rgba(250, 204, 21, 0.1) !important;
    }

    /* Premium Status Cards & Streamlit Metric Containers styling */
    .status-card, 
    div[data-testid="metric-container"],
    div[data-testid="stMetric"],
    .stMetric {
        background: rgba(20, 25, 40, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(0, 229, 255, 0.05), inset 0 0 20px rgba(250, 204, 21, 0.02) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        display: flex;
        flex-direction: column;
        margin-bottom: 12px;
        transition: all 0.3s ease !important;
    }
    .status-card:hover, 
    div[data-testid="metric-container"]:hover,
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 229, 255, 0.4) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 229, 255, 0.2), 0 0 20px rgba(250, 204, 21, 0.1) !important;
    }
    
    /* Label styling */
    .status-label, 
    div[data-testid="metric-container"] label,
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 6px !important;
    }
    
    /* Value styling */
    .status-val, 
    div[data-testid="metric-container"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Clean up default Streamlit metric margin */
    div[data-testid="metric-container"] > div,
    div[data-testid="stMetric"] > div {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Premium File Uploader Dropzone Override */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(18, 24, 38, 0.5) !important;
        border: 1px dashed rgba(6, 182, 212, 0.3) !important;
        border-radius: 12px !important;
        padding: 30px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00e5ff !important;
        background-color: rgba(6, 182, 212, 0.05) !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.1) !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background: rgba(6, 182, 212, 0.1) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        color: #00e5ff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background: rgba(6, 182, 212, 0.25) !important;
        border-color: #00e5ff !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #94a3b8 !important;
    }

    /* HTML Tables and DataFrame styling */
    div[data-testid="stDataFrame"], 
    div[data-testid="stTable"], 
    .stTable,
    div[data-testid="stDataFrame"] iframe {
        background-color: rgba(18, 24, 38, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    table {
        background-color: rgba(18, 24, 38, 0.5) !important;
        color: #f1f5f9 !important;
        border-collapse: collapse !important;
        width: 100% !important;
        border: none !important;
    }
    thead tr {
        background-color: rgba(6, 182, 212, 0.1) !important;
    }
    th {
        background-color: transparent !important;
        color: #00e5ff !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(6, 182, 212, 0.2) !important;
        padding: 12px 16px !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    td {
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 12px 16px !important;
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    tr:hover {
        background-color: rgba(255, 255, 255, 0.02) !important;
    }

    /* Glassmorphic Chat Messages overrides */
    div[data-testid="stChatMessage"] {
        background-color: rgba(22, 28, 38, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }

    /* Expanders styling */
    div[data-testid="stExpander"] {
        background: rgba(22, 28, 38, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(6, 182, 212, 0.3) !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    /* Inputs, Selectboxes, Chat Inputs */
    div[data-testid="stSelectbox"] > div, div[data-testid="stTextInput"] > div {
        background-color: rgba(18, 24, 38, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #121620 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Action Buttons */
    button[kind="secondaryFormSubmit"], button[kind="primary"], button[kind="secondary"] {
        background: rgba(6, 182, 212, 0.1) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 10px !important;
        color: #00e5ff !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondaryFormSubmit"]:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background: rgba(6, 182, 212, 0.25) !important;
        border-color: #00e5ff !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sleek scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #080a0f;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(6, 182, 212, 0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(6, 182, 212, 0.4);
    }

    /* Dashboard Mockup - Layout, floating orb, option cards */
    .welcome-card-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin-top: 30px;
    }
    .welcome-option-card {
        background: rgba(22, 28, 38, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 24px;
        text-align: left;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    .welcome-option-card:hover {
        transform: translateY(-4px);
        border-color: rgba(6, 182, 212, 0.4);
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.08);
    }
    .welcome-card-icon {
        font-size: 1.5rem;
        color: #00e5ff;
        margin-bottom: 12px;
    }
    .welcome-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .welcome-card-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    .welcome-card-arrow {
        position: absolute;
        top: 20px;
        right: 20px;
        color: rgba(255, 255, 255, 0.3);
        font-size: 1rem;
    }
    
    .welcome-header {
        text-align: center;
        margin: 40px 0 20px 0;
    }
    .welcome-sphere-container {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 24px auto;
    }
    .welcome-sphere {
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 30% 30%, #00f0ff 0%, #007bff 40%, #000c3b 90%);
        border-radius: 50%;
        box-shadow: 0 0 50px rgba(0, 229, 255, 0.4), inset -12px -12px 25px rgba(0, 0, 0, 0.8);
        animation: sphereFloat 4s ease-in-out infinite;
    }
    .welcome-sphere-glow {
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: radial-gradient(circle, rgba(0, 229, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulseGlow 4s ease-in-out infinite;
    }
    @keyframes sphereFloat {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-12px); }
        100% { transform: translateY(0px); }
    }
    @keyframes pulseGlow {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    .welcome-title-small {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .welcome-title-main {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Load model & artefak dari folder project ─────────────────
# Tanggal mulai training (Dec 2023) — untuk day_index yang konsisten
TRAINING_START = pd.Timestamp('2023-12-31')

@st.cache_resource
def load_artifacts():
    model  = joblib.load('model_xgb_tuned.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_columns.json') as f:
        meta = json.load(f)
    return model, scaler, meta

# ── Baca file: support CSV dan XLSX ──────────────────────────
def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    else:
        # Coba beberapa encoding umum untuk CSV Shopee
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')

# ── Cleaning RAW Shopee → DataFrame bersih ───────────────────
def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Identik dengan logika 01_data_preparation.py"""
    df = df_raw.copy()

    # Normalisasi kolom tanggal
    if 'Waktu Pengiriman Diatur' in df.columns and 'Waktu Pesanan Dibuat' not in df.columns:
        df = df.rename(columns={'Waktu Pengiriman Diatur': 'Waktu Pesanan Dibuat'})

    # Fix Total Pembayaran: "35.663" → 35663.0
    if 'Total Pembayaran' in df.columns:
        df['Total Pembayaran'] = (
            df['Total Pembayaran'].astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df['Total Pembayaran'] = pd.to_numeric(df['Total Pembayaran'], errors='coerce')

    # Fix Total Berat: "600 gr" → 600.0
    if 'Total Berat' in df.columns:
        df['Total Berat'] = (
            df['Total Berat'].astype(str)
            .str.extract(r'([\d,]+)')[0]
            .str.replace(',', '.', regex=False)
        )
        df['Total Berat'] = pd.to_numeric(df['Total Berat'], errors='coerce')

    # Normalisasi & filter status Selesai
    if 'Status Pesanan' in df.columns:
        df['Status Pesanan'] = df['Status Pesanan'].apply(
            lambda x: 'Selesai' if str(x).startswith('Pesanan diterima') else x
        )
        df = df[df['Status Pesanan'] == 'Selesai'].copy()

    # Buang pembayaran 0 / NaN
    if 'Total Pembayaran' in df.columns:
        df = df[df['Total Pembayaran'] > 0].copy()

    # Isi NaN kolom non-kritis
    for col, val in [('Alasan Pembatalan', 'Tidak Ada'),
                     ('Returned quantity', 0),
                     ('Total Diskon', 0)]:
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Parse tanggal
    if 'Waktu Pesanan Dibuat' in df.columns:
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(
            df['Waktu Pesanan Dibuat'], errors='coerce')
        df['order_date'] = df['Waktu Pesanan Dibuat'].dt.normalize()

    return df.reset_index(drop=True)

# ── Agregasi ke level harian ──────────────────────────────────
def make_daily(df_clean: pd.DataFrame) -> pd.DataFrame:
    # Deduplikasi order_id (multi-item order)
    df_order = (df_clean.drop_duplicates(subset='order_id', keep='first')
                if 'order_id' in df_clean.columns else df_clean.copy())

    daily = (
        df_order.groupby('order_date')['Total Pembayaran']
        .sum().reset_index()
        .rename(columns={'Total Pembayaran': 'total_revenue'})
    )
    # Isi missing dates dengan 0
    full = pd.date_range(daily['order_date'].min(),
                         daily['order_date'].max(), freq='D')
    daily = (daily.set_index('order_date')
             .reindex(full).reset_index()
             .rename(columns={'index': 'order_date'}))
    daily['total_revenue'] = daily['total_revenue'].fillna(0)
    daily['order_date']    = pd.to_datetime(daily['order_date'])
    return daily.sort_values('order_date').reset_index(drop=True)

# ── Feature engineering (identik dengan 02c) ─────────────────
def build_features_daily(df: pd.DataFrame,
                         median_nonzero: float = None,
                         training_start: pd.Timestamp = None) -> pd.DataFrame:
    """
    median_nonzero: jika diberikan, hari dengan revenue=0 akan di-impute
    dengan nilai ini HANYA untuk komputasi lag (bukan untuk kolom total_revenue).
    Ini mencegah lag=0 saat data punya banyak hari tutup (Lebaran, dll).
    """
    df  = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    doy = df['order_date'].dt.dayofyear

    df['year']           = df['order_date'].dt.year
    df['month']          = df['order_date'].dt.month
    df['day']            = df['order_date'].dt.day
    df['dayofweek']      = df['order_date'].dt.dayofweek
    df['quarter']        = df['order_date'].dt.quarter
    df['week_of_year']   = df['order_date'].dt.isocalendar().week.astype(int)
    df['is_weekend']     = df['dayofweek'].isin([5,6]).astype(int)
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end']   = (df['day'] >= 25).astype(int)
    df['is_ramadan']     = (
        ((df['month'] == 3) & (df['year'].isin([2024,2025,2026]))) |
        ((df['month'] == 4) & (df['year'].isin([2024,2025])))
    ).astype(int)
    df['is_harbolnas']   = (
        (df['month']==12) & (df['day']>=10) & (df['day']<=12)
    ).astype(int)
    # is_holiday_gap: untuk baris FORECAST (placeholder), jangan tandai sebagai hari tutup
    if 'is_forecast_placeholder' in df.columns:
        df['is_holiday_gap'] = ((df['total_revenue'] == 0) &
                                (df['is_forecast_placeholder'] != 1)).astype(int)
    else:
        df['is_holiday_gap'] = (df['total_revenue'] == 0).astype(int)
    df['sin_week']       = np.sin(2*np.pi*df['dayofweek']/7)
    df['cos_week']       = np.cos(2*np.pi*df['dayofweek']/7)
    df['sin_month']      = np.sin(2*np.pi*df['month']/12)
    df['cos_month']      = np.cos(2*np.pi*df['month']/12)
    df['sin_year']       = np.sin(2*np.pi*doy/365)
    df['cos_year']       = np.cos(2*np.pi*doy/365)
    # day_index harus relatif terhadap tanggal mulai TRAINING
    ref_date = training_start if training_start is not None else df['order_date'].min()
    df['day_index'] = (df['order_date'] - ref_date).dt.days

    # Untuk lag: hari tutup (revenue=0) di-impute dengan median non-zero
    if median_nonzero is not None and median_nonzero > 0:
        rev_for_lag = df['total_revenue'].replace(0, median_nonzero)
    else:
        rev_for_lag = df['total_revenue']

    df['lag_1']           = rev_for_lag.shift(1)
    df['lag_7']           = rev_for_lag.shift(7)
    df['lag_30']          = rev_for_lag.shift(30)
    df['rolling_7d_mean'] = rev_for_lag.shift(1).rolling(7,  min_periods=1).mean()
    df['rolling_30d_mean']= rev_for_lag.shift(1).rolling(30, min_periods=1).mean()
    return df

# ── Forecast rolling ──────────────────────────────────────────
def run_forecast(model, scaler, meta, daily_hist: pd.DataFrame, n_days: int):
    df_roll    = daily_hist.copy()
    preds      = []
    last_dt    = df_roll['order_date'].max()
    SCALE_COLS = meta['scale_cols']

    nonzero_vals = df_roll.loc[df_roll['total_revenue'] > 0, 'total_revenue']
    median_nz = float(nonzero_vals.median()) if len(nonzero_vals) > 0 else 1_000_000.0

    for i in range(n_days):
        future_dt = last_dt + timedelta(days=i+1)
        placeholder = pd.DataFrame([{'order_date': future_dt,
                                      'total_revenue': 0,
                                      'is_forecast_placeholder': 1}])
        temp = pd.concat([df_roll, placeholder], ignore_index=True)
        temp = build_features_daily(temp, median_nonzero=median_nz,
                                    training_start=TRAINING_START)
        row  = temp.iloc[[-1]].copy()

        for col in meta['features']:
            if col not in row.columns:
                row[col] = 0

        X = row[meta['features']].copy()

        # Scale fitur menggunakan scaler yang di-fit saat training
        X_scaled = X.copy()
        scale_present = [c for c in SCALE_COLS if c in X_scaled.columns]
        if scale_present:
            X_scaled[scale_present] = scaler.transform(X[scale_present])

        y_idr = max(0.0, float(np.expm1(model.predict(X_scaled)[0])))
        preds.append({'order_date': future_dt, 'predicted_revenue': y_idr})

        df_roll = pd.concat([
            df_roll,
            pd.DataFrame([{'order_date': future_dt, 'total_revenue': y_idr}])
        ], ignore_index=True)

        if y_idr > 0:
            median_nz = float(
                df_roll.loc[df_roll['total_revenue'] > 0, 'total_revenue'].median()
            )

    return pd.DataFrame(preds)


# ════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS — CHATBOT AI AGENT
#  (didefinisikan SEBELUM routing agar tidak NameError)
# ════════════════════════════════════════════════════════════

def _demo_response(prompt: str) -> str:
    """Fallback response saat GOOGLE_API_KEY belum diset."""
    p = prompt.lower()
    if any(k in p for k in ['produk', 'terlaris', 'kategori']):
        return (
            '**Analisis Produk** membutuhkan `GOOGLE_API_KEY`.\n\n'
            'Setelah key diset, agent akan otomatis membuat grafik top produk terlaris.'
        )
    elif any(k in p for k in ['revenue', 'pendapatan', 'omzet', 'penjualan']):
        return (
            '**Lihat Dashboard Forecasting** untuk prediksi revenue 1–3 bulan ke depan.\n\n'
            'Atau tambahkan `GOOGLE_API_KEY` untuk analisis mendalam via chatbot.'
        )
    elif any(k in p for k in ['kompetitor', 'harga', 'shopee', 'tokopedia', 'pasar']):
        return (
            '**Riset Kompetitor** membutuhkan `TAVILY_API_KEY`.\n\n'
            'Daftar gratis di [tavily.com](https://tavily.com) untuk Web Search tool.'
        )
    elif any(k in p for k in ['grafik', 'chart', 'plot', 'visualisasi']):
        return (
            '**Buat Grafik** membutuhkan `GOOGLE_API_KEY` untuk Python REPL tool.\n\n'
            'Agent akan mengeksekusi kode Python dan menampilkan grafik otomatis.'
        )
    else:
        return (
            '**API Key belum diset.**\n\n'
            'Tambahkan di `.streamlit/secrets.toml`:\n'
            '```toml\n'
            'GOOGLE_API_KEY = "AIza..."  # wajib\n'
            'TAVILY_API_KEY = "tvly-..."  # opsional (Web Search)\n'
            '```\n\n'
            'Dapatkan gratis di [aistudio.google.com](https://aistudio.google.com/app/apikey)'
        )


@st.cache_resource
def _init_agent(_google_api_key: str, _tavily_api_key: str = "", _model: str = "gemini-1.5-flash"):
    """
    Inisialisasi LangGraph ReAct Agent (kompatibel LangChain 1.x).

    Arsitektur sesuai brief:
    - LLM   : Gemini (Google GenAI) via langchain-google-genai 4.x
    - Tool 1: Python REPL — analisis data & buat grafik otomatis
    - Tool 2: Tavily Web Search — riset harga kompetitor (opsional)
    - Pola  : ReAct (Reasoning + Acting) via langgraph.prebuilt

    CATATAN BREAKING CHANGES LangChain 1.x:
    - AgentExecutor DIHAPUS → gunakan langgraph.prebuilt.create_react_agent
    - convert_system_message_to_human DIHAPUS dari ChatGoogleGenerativeAI
    - Invoke format berubah: {messages: [(human, input)]} → result[messages][-1]
    """
    import warnings
    warnings.filterwarnings('ignore')

    # ── Import LangGraph (LangChain 1.x) ─────────────────────
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.tools.python.tool import PythonREPLTool
    from langgraph.prebuilt import create_react_agent   # BUKAN langchain.agents

    # ── LLM: Gemini dengan model yang dipilih ────────────────
    # Model fallback order: gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-flash-8b
    # convert_system_message_to_human dihapus di langchain-google-genai 4.x
    llm = ChatGoogleGenerativeAI(
        model=_model,
        google_api_key=_google_api_key,
        temperature=0.2,
    )

    # ── Tool 1: Python REPL ───────────────────────────────────
    python_repl = PythonREPLTool()
    python_repl.description = (
        "Eksekusi kode Python untuk analisis data dan visualisasi. "
        "Data tersedia di: 'user_data.csv' (data upload user) ATAU "
        "'ecommerce_cleaned.csv' (data demo bawaan). "
        "Selalu cek file mana yang tersedia dengan os.path.exists(). "
        "Kolom utama: order_id, product_category, Status Pesanan, "
        "Waktu Pesanan Dibuat, Total Pembayaran, Provinsi, Metode Pembayaran. "
        "Untuk membuat grafik: WAJIB akhiri kode dengan "
        "plt.savefig('temp_plot.png', dpi=120, bbox_inches='tight'); plt.close() "
        "agar grafik dapat ditampilkan di UI. "
        "Gunakan matplotlib dengan style modern dan label dalam Bahasa Indonesia."
    )

    tools = [python_repl]

    # ── Tool 2: Tavily Web Search (opsional) ─────────────────
    if _tavily_api_key:
        os.environ["TAVILY_API_KEY"] = _tavily_api_key
        try:
            # langchain-community masih bisa dipakai, tapi perlu tavily_api_key eksplisit
            from langchain_community.tools.tavily_search import TavilySearchResults
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                web_search = TavilySearchResults(
                    max_results=5,
                    tavily_api_key=_tavily_api_key,
                )
            tools.append(web_search)
        except Exception:
            pass  # Tavily tidak tersedia, lanjut tanpa web search

    # ── System Prompt untuk LangGraph ReAct ──────────────────
    system_prompt = """Kamu adalah AI Analyst E-commerce Indonesia yang ahli, ramah, dan profesional.
Kamu membantu pemilik bisnis UMKM menganalisis data penjualan dan mendapatkan insight bisnis yang actionable.

=== ATURAN ROUTING TOOLS (WAJIB DIIKUTI) ===

Gunakan Python_REPL HANYA untuk:
- Analisis data INTERNAL dari file CSV (revenue, transaksi, dll)
- Membuat grafik/chart dari data yang ada
- Menghitung statistik: rata-rata, total, tren dari data
- Pertanyaan mengandung: "berapa", "tampilkan data", "grafik", "chart", "analisis data", "produk terlaris dari data"

Gunakan tavily_search_results_json untuk:
- Strategi promosi, tips pemasaran, cara berjualan online
- Riset harga kompetitor di marketplace (Shopee, Tokopedia, Lazada)
- Tren pasar e-commerce terkini
- Berita bisnis online, SEO, iklan digital, konten marketing
- Pertanyaan mengandung: "strategi", "cara", "tips", "promosi", "marketing", "kompetitor", 
  "harga pasar", "tren terkini", "Shopee", "Tokopedia", "Lazada"

PENTING: Jika pertanyaan tentang STRATEGI atau CARA melakukan sesuatu
→ LANGSUNG gunakan Web Search, JANGAN gunakan Python REPL!

=== PANDUAN PYTHON REPL ===
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Cek file data
data_file = 'user_data.csv' if os.path.exists('user_data.csv') else 'ecommerce_cleaned.csv'
df = pd.read_csv(data_file)
print(df.shape, df.columns.tolist())

# ... analisis ...

# WAJIB untuk grafik — simpan ke file agar tampil di UI
plt.savefig('temp_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print('Grafik berhasil dibuat')
```

=== PANDUAN JAWABAN FINAL ===
- Gunakan Bahasa Indonesia yang jelas dan profesional
- Format angka: Rp 1.234.567
- Berikan minimal 3 rekomendasi actionable setelah setiap jawaban
- Untuk jawaban Web Search: berikan sumber informasi dan konteks
- Untuk jawaban data: jelaskan insight dari grafik yang dibuat"""

    # ── Buat LangGraph ReAct Agent ───────────────────────────
    # Signature: create_react_agent(model, tools, prompt=system_str, ...)
    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent_graph


# ════════════════════════════════════════════════════════════
#  GLOBAL ONBOARDING
# ════════════════════════════════════════════════════════════
if "intro_completed" not in st.session_state:
    st.session_state.intro_completed = False
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

if not st.session_state.intro_completed:
    st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="welcome-header" style="margin-top: 2vh;">
            <div class="welcome-sphere-container">
                <div class="welcome-sphere-glow"></div>
                <div class="welcome-sphere"></div>
            </div>
            <div class="welcome-title-small">SELAMAT DATANG DI</div>
            <div class="welcome-title-main">E-commerce AI Analyst</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    if st.session_state.slide_index == 0:
        st.markdown(
            """
            <div style="margin-top: 20px; margin-bottom: 20px; text-align: center;">
                <h3 style="color: #f8fafc; font-weight: 600;">Yang Dapat Kami Lakukan</h3>
                <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">Aplikasi ini merubah cara Anda menganalisis data penjualan e-commerce secara otomatis.</p>
            </div>
            <div class="welcome-card-grid">
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">⚡</div>
                    <div class="welcome-card-title">1. Automasi Analisis</div>
                    <div class="welcome-card-desc">Data mentah langsung diolah secara otomatis menjadi laporan siap baca tanpa rumus rumit.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">🧠</div>
                    <div class="welcome-card-title">2. Insight AI Cerdas</div>
                    <div class="welcome-card-desc">AI menganalisis tren penjualan tersembunyi dan memberikan rekomendasi strategi yang konkrit.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">📊</div>
                    <div class="welcome-card-title">3. Visualisasi Interaktif</div>
                    <div class="welcome-card-desc">Ubah ratusan baris data transaksi menjadi grafik interaktif yang mudah dipahami.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
            if st.button("Selanjutnya", type="primary", use_container_width=True):
                st.session_state.slide_index = 1
                st.rerun()

    elif st.session_state.slide_index == 1:
        st.markdown(
            """
            <div style="margin-top: 20px; margin-bottom: 20px; text-align: center;">
                <h3 style="color: #f8fafc; font-weight: 600;">Fitur Yang Dapat Digunakan</h3>
                <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">Tiga alat utama untuk mengoptimalkan operasional dan omzet bisnis Anda.</p>
            </div>
            <div class="welcome-card-grid">
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">📂</div>
                    <div class="welcome-card-title">1. Upload Data</div>
                    <div class="welcome-card-desc">Upload puluhan file laporan penjualan sekaligus. Sistem akan membersihkan dan menggabungkannya otomatis.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">📈</div>
                    <div class="welcome-card-title">2. Dashboard Forecasting</div>
                    <div class="welcome-card-desc">Lihat dasbor yang memetakan tren harian dan produk terlaris, serta memprediksi potensi omzet.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">🤖</div>
                    <div class="welcome-card-title">3. Chatbot AI Agent</div>
                    <div class="welcome-card-desc">Asisten virtual yang bisa riset web (mencari harga kompetitor online) dan memproses coding Python.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
        with col2:
            if st.button("Sebelumnya", use_container_width=True):
                st.session_state.slide_index = 0
                st.rerun()
        with col3:
            if st.button("Selanjutnya", type="primary", use_container_width=True):
                st.session_state.slide_index = 2
                st.rerun()

    elif st.session_state.slide_index == 2:
        st.markdown(
            """
            <div style="margin-top: 20px; margin-bottom: 20px; text-align: center;">
                <h3 style="color: #f8fafc; font-weight: 600;">Langkah-Langkah Memulai</h3>
                <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">Cukup tiga langkah mudah untuk mengaktifkan asisten analitik pribadi Anda.</p>
            </div>
            <div class="welcome-card-grid">
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">1️⃣</div>
                    <div class="welcome-card-title">Upload Data</div>
                    <div class="welcome-card-desc">Unggah file ekspor pesanan mentah Anda ke dalam menu <b>Upload Data</b>.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">2️⃣</div>
                    <div class="welcome-card-title">Lihat Insight Data</div>
                    <div class="welcome-card-desc">Buka menu <b>Dashboard Forecasting</b> untuk melihat visualisasi grafik performa toko.</div>
                </div>
                <div class="welcome-option-card">
                    <div class="welcome-card-icon" style="background:none; box-shadow:none; font-size: 2.2rem; margin-bottom: 10px;">3️⃣</div>
                    <div class="welcome-card-title">Tanya Agent</div>
                    <div class="welcome-card-desc">Buka menu <b>Chatbot AI Agent</b> dan bertanyalah apapun kepada AI tentang bisnis Anda.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
        with col2:
            if st.button("Sebelumnya", use_container_width=True):
                st.session_state.slide_index = 1
                st.rerun()
        with col3:
            if st.button("Tanya Sekarang", type="primary", use_container_width=True):
                st.session_state.intro_completed = True
                st.rerun()

    st.stop()

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-icon"></div>
            <span class="logo-text">E-commerce AI</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    halaman = st.radio(
        "Navigasi",
        ["Upload Data", "Dashboard Forecasting", "Chatbot AI Agent"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("AI Agent Analitik E-commerce v3.0")


# ════════════════════════════════════════════════════════════
#  HALAMAN 1 : UPLOAD DATA
# ════════════════════════════════════════════════════════════
if halaman == "Upload Data":
    st.title("Upload Data Transaksi")
    st.info("Penjelasan Fitur: Di halaman ini Anda dapat mengunggah file laporan penjualan mentah (Excel/CSV) dari *Seller Centre* e-commerce Anda. Sistem kami akan secara cerdas membersihkan, memformat, dan menggabungkan data Anda sehingga siap untuk dianalisis oleh AI.", icon="💡")

    with st.expander("Format file yang diterima"):
        st.markdown("""
**Format yang didukung:** `.xlsx`, `.xls`, `.csv`

**Multi-file:** bisa upload beberapa bulan sekaligus, app akan
menggabungkan dan mengurutkan otomatis.

**App otomatis:**
- Filter hanya status **Selesai**
- Fix format angka ribuan
- Deduplikasi multi-item order
- Agregasi ke revenue harian
        """)

    # Upload multi-file: CSV dan XLSX
    uploaded_files = st.file_uploader(
        "Pilih file Shopee (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Bisa pilih banyak file sekaligus dengan Ctrl+klik"
    )

    if uploaded_files:
        all_raw = []
        failed  = []
        with st.spinner(f"Membaca {len(uploaded_files)} file..."):
            for f in uploaded_files:
                try:
                    df_raw = read_file(f)
                    df_raw['_source_file'] = f.name
                    all_raw.append(df_raw)
                except Exception as e:
                    failed.append(f"{f.name}: {e}")

        if failed:
            for msg in failed:
                st.warning(f"Gagal baca: {msg}")

        if all_raw:
            df_combined = pd.concat(all_raw, ignore_index=True)
            df_clean    = clean_raw(df_combined)

            if len(df_clean) == 0:
                st.error(
                    "Tidak ada transaksi 'Selesai' di file yang diupload. "
                    "Pastikan kolom 'Status Pesanan' ada dan berisi data valid."
                )
            else:
                st.session_state['df_clean']   = df_clean
                st.session_state['daily_hist'] = make_daily(df_clean)
                st.session_state['uploaded_file_names'] = [f.name for f in uploaded_files]

                n_files = len(uploaded_files)
                n_raw   = len(df_combined)
                n_clean = len(df_clean)
                daily   = st.session_state['daily_hist']
                n_days  = len(daily)

                st.success(
                    f"{n_files} file berhasil diproses: "
                    f"{n_raw:,} baris mentah → "
                    f"{n_clean:,} transaksi Selesai → "
                    f"{n_days} hari data"
                )

    # Tampilkan data jika sudah ada
    if 'df_clean' not in st.session_state:
        st.info("Belum ada data. Upload file di atas untuk memulai.")
        st.stop()

    df    = st.session_state['df_clean']
    daily = st.session_state['daily_hist']

    # Metrik
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi",  f"{len(df):,}")
    c2.metric("Total Revenue",    f"Rp {df['Total Pembayaran'].sum()/1e6:.1f}jt")
    c3.metric("Produk Unik",
              str(df['product_category'].nunique()) if 'product_category' in df.columns else "—")
    c4.metric("Rentang Tanggal",
              f"{daily['order_date'].min().strftime('%d %b %Y')} – "
              f"{daily['order_date'].max().strftime('%d %b %Y')}")

    # Preview
    st.subheader("Preview Data (setelah cleaning)")
    cols_show = [c for c in ['order_id','product_category','Status Pesanan',
                              'Waktu Pesanan Dibuat','Total Pembayaran',
                              'Jumlah','Metode Pembayaran','Provinsi']
                 if c in df.columns]
    st.dataframe(df[cols_show].head(20), use_container_width=True)

    # Tren revenue harian
    st.subheader("Tren Revenue Harian")
    fig = px.area(daily, x='order_date', y='total_revenue', 
                  labels={'order_date': 'Tanggal', 'total_revenue': 'Revenue (IDR)'})
    fig.update_traces(line_color='#00e5ff', fillcolor='rgba(0, 229, 255, 0.2)',
                      hovertemplate='<b>Tanggal</b>: %{x|%d %b %Y}<br><b>Revenue</b>: Rp %{y:,.0f}<extra></extra>')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 produk
    if 'product_category' in df.columns:
        st.subheader("Top 10 Produk Terlaris (revenue)")
        top = (df.groupby('product_category')['Total Pembayaran']
               .sum().sort_values(ascending=True).tail(10).reset_index())
        fig2 = px.bar(top, x='Total Pembayaran', y='product_category', orientation='h',
                      labels={'Total Pembayaran': 'Total Revenue (IDR)', 'product_category': 'Kategori Produk'})
        fig2.update_traces(marker_color='#facc15', 
                           hovertemplate='<b>Kategori</b>: %{y}<br><b>Revenue</b>: Rp %{x:,.0f}<extra></extra>')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  HALAMAN 2 : DASHBOARD FORECASTING
# ════════════════════════════════════════════════════════════
elif halaman == "Dashboard Forecasting":
    st.title("Dashboard Forecasting")
    st.info("**Penjelasan Fitur:**\nDi halaman ini Anda dapat melihat ringkasan performa penjualan Anda melalui grafik interaktif. Sistem juga menyajikan prediksi omzet (forecasting) untuk masa depan berdasarkan tren historis data penjualan Anda.", icon="💡")

    if 'daily_hist' not in st.session_state:
        st.warning("Upload data dulu di halaman **Upload Data**.")
        st.stop()

    try:
        model, scaler, meta = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.info("Pastikan `model_xgb_tuned.pkl` dan `feature_columns.json` ada di folder project.")
        st.stop()

    daily_hist  = st.session_state['daily_hist']
    n_days_hist = len(daily_hist)

    if n_days_hist < 30:
        st.error(
            f"Data hanya {n_days_hist} hari. "
            "Diperlukan minimal 30 hari agar lag_30 bisa dihitung. "
            "Upload lebih banyak file."
        )
        st.stop()

    # Info statistik data
    nonzero   = daily_hist.loc[daily_hist['total_revenue']>0, 'total_revenue']
    median_nz = float(nonzero.median()) if len(nonzero) > 0 else 0
    pct_zero  = (daily_hist['total_revenue']==0).mean() * 100

    st.info(
        f"Histori: **{daily_hist['order_date'].min().strftime('%d %b %Y')}** "
        f"s/d **{daily_hist['order_date'].max().strftime('%d %b %Y')}** "
        f"({n_days_hist} hari) | "
        f"Hari aktif: **{len(nonzero)}** | "
        f"Median revenue/hari aktif: **Rp {median_nz/1e3:.0f}rb**"
    )

    # Warning jika banyak hari tutup (misal April = Lebaran)
    if pct_zero > 20:
        st.warning(
            f"{pct_zero:.0f}% hari dalam data ini revenue = 0 (hari tutup/libur). "
            f"Lag features di-impute dengan median Rp {median_nz/1e3:.0f}rb "
            f"agar prediksi tetap realistis."
        )

    # Kontrol horizon
    st.sidebar.markdown("### Pengaturan Forecast")
    horizon = st.sidebar.selectbox(
        "Horizon prediksi",
        [30, 60, 90],
        format_func=lambda x: f"{x} hari ({x//30} bulan)"
    )

    # Jalankan forecast
    with st.spinner(f"Menghitung prediksi {horizon} hari ke depan..."):
        df_forecast = run_forecast(model, scaler, meta, daily_hist, horizon)

    # Metrik
    total_pred = df_forecast['predicted_revenue'].sum()
    avg_pred   = df_forecast['predicted_revenue'].mean()
    max_val    = df_forecast['predicted_revenue'].max()
    max_dt     = df_forecast.loc[df_forecast['predicted_revenue'].idxmax(), 'order_date']
    avg_hist   = daily_hist[daily_hist['total_revenue']>0]['total_revenue'].mean()
    delta_pct  = (avg_pred - avg_hist) / avg_hist * 100 if avg_hist > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Prediksi Revenue", f"Rp {total_pred:,.0f}",
              help=f"Akumulasi {horizon} hari ke depan")
    c2.metric("Rata-rata per Hari",
              f"Rp {avg_pred:,.0f}",
              delta=f"{delta_pct:+.1f}% vs historis")
    c3.metric("Puncak Prediksi",
              f"Rp {max_val:,.0f}",
              help=f"Tanggal: {max_dt.strftime('%d %b %Y')}")
    c4.metric("Model",                  "XGBoost Tuned",
              help="RMSE: ±Rp875rb | R²: 0.13 | MAPE: 58.92%")

    # Penjelasan sifat prediksi rolling forecast
    pred_std  = df_forecast['predicted_revenue'].std()
    pred_mean = df_forecast['predicted_revenue'].mean()
    hist_std  = daily_hist[daily_hist['total_revenue']>0]['total_revenue'].std()

    with st.expander("Cara membaca grafik prediksi ini", expanded=False):
        st.markdown(f"""
**Kenapa garis prediksi terlihat lebih stabil dari historis?**

Ini adalah perilaku normal **rolling forecast** berbasis lag features:

- Hari pertama prediksi menggunakan data aktual terakhir sebagai lag
- Hari berikutnya, lag diisi dengan hasil prediksi sebelumnya
- Akibatnya prediksi konverge ke nilai tengah (~Rp {pred_mean/1e3:.0f}rb/hari)
- Variasi tersisa berasal dari pola mingguan (hari kerja vs weekend)

**Std prediksi: Rp {pred_std/1e3:.0f}rb** vs **std historis: Rp {hist_std/1e3:.0f}rb**
→ Model tidak bisa prediksi spike mendadak (harbolnas, viral) karena R²=0.13

**Gunakan prediksi ini sebagai:** estimasi baseline revenue, bukan prediksi hari per hari yang presisi.
        """)

    # ── Panel 1: Grafik Historis (Plotly interaktif) ─────────────
    fig_hist = px.area(
        daily_hist,
        x='order_date',
        y='total_revenue',
        title=f'Data Historis Aktual ({n_days_hist} hari)',
        labels={'order_date': 'Tanggal', 'total_revenue': 'Revenue (IDR)'},
        color_discrete_sequence=['#00E5FF'],
    )
    fig_hist.update_traces(
        line=dict(width=2, color='#00E5FF'),
        fillcolor='rgba(0,229,255,0.12)',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Revenue: Rp %{y:,.0f}<extra></extra>'
    )
    # Tambahkan garis rata-rata historis
    fig_hist.add_hline(
        y=avg_hist,
        line_dash='dot', line_color='#FFD700', line_width=1.5,
        annotation_text=f'Rata-rata: Rp {avg_hist/1e3:.0f}rb',
        annotation_font_color='#FFD700',
        annotation_position='top right'
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.04)',
        font_color='#e2e8f0',
        title_font_size=14,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickformat='%d %b %Y',
            title=None,
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickformat=',',
            title='Revenue (IDR)',
            tickprefix='Rp ',
        ),
        showlegend=False,
        hovermode='x unified',
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Panel 2: Grafik Prediksi (Plotly interaktif) ──────────────
    import plotly.graph_objects as go
    fig_pred = go.Figure()

    # Area rentang kemungkinan ±42%
    fig_pred.add_trace(go.Scatter(
        x=pd.concat([df_forecast['order_date'], df_forecast['order_date'][::-1]]),
        y=pd.concat([df_forecast['predicted_revenue'] * 1.42,
                     (df_forecast['predicted_revenue'] * 0.58)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,215,0,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Rentang ±42% (MAPE)',
        hoverinfo='skip',
    ))

    # Garis rata-rata historis sebagai referensi
    fig_pred.add_hline(
        y=avg_hist,
        line_dash='dot', line_color='#00E5FF', line_width=1.2,
        annotation_text=f'Rata-rata historis: Rp {avg_hist/1e3:.0f}rb',
        annotation_font_color='#00E5FF',
        annotation_position='top left'
    )

    # Garis prediksi utama
    fig_pred.add_trace(go.Scatter(
        x=df_forecast['order_date'],
        y=df_forecast['predicted_revenue'],
        mode='lines',
        name='Prediksi baseline',
        line=dict(color='#FFD700', width=2.5, dash='dash'),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Prediksi: Rp %{y:,.0f}<extra></extra>',
    ))

    fig_pred.update_layout(
        title=f'Prediksi {horizon} Hari ke Depan',
        title_font_size=14,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.04)',
        font_color='#e2e8f0',
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickformat='%d %b',
            title='Tanggal',
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickformat=',',
            title='Revenue (IDR)',
            tickprefix='Rp ',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
        ),
        hovermode='x unified',
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Ringkasan prediksi per minggu
    st.subheader("Ringkasan Prediksi per Minggu")
    df_forecast['minggu'] = df_forecast['order_date'].dt.to_period('W')
    weekly = df_forecast.groupby('minggu').agg(
        total=('predicted_revenue','sum'),
        rata_rata=('predicted_revenue','mean'),
        n_hari=('predicted_revenue','count')
    ).reset_index()
    weekly['Periode'] = weekly['minggu'].apply(
        lambda w: f"{w.start_time.strftime('%d %b')} – {w.end_time.strftime('%d %b %Y')}")
    weekly['Total Prediksi'] = weekly['total'].apply(lambda x: f"Rp {x:,.0f}")
    weekly['Rata-rata/Hari'] = weekly['rata_rata'].apply(lambda x: f"Rp {x:,.0f}")
    weekly['Hari'] = weekly['n_hari']
    st.dataframe(weekly[['Periode','Hari','Rata-rata/Hari','Total Prediksi']],
                 use_container_width=True, hide_index=True)

    st.caption(
        f"**Catatan penting:** Prediksi ini adalah estimasi baseline (rata-rata yang diharapkan). "
        f"Revenue aktual bisa 42–58% lebih tinggi atau lebih rendah karena faktor tak terduga "
        f"(event viral, promosi mendadak, harbolnas). "
        f"Model XGBoost Tuned, RMSE ±Rp875rb/hari, R²=0.13."
    )

    # Tabel detail per hari
    with st.expander("Lihat tabel prediksi detail per hari"):
        df_show = df_forecast.copy()
        df_show['Hari']            = df_show['order_date'].dt.strftime('%A')
        df_show['Tanggal']         = df_show['order_date'].dt.strftime('%d %b %Y')
        df_show['Prediksi Revenue'] = df_show['predicted_revenue'].apply(lambda x: f"Rp {x:,.0f}")
        df_show['Batas Bawah']     = (df_show['predicted_revenue']*0.58).apply(lambda x: f"Rp {x:,.0f}")
        df_show['Batas Atas']      = (df_show['predicted_revenue']*1.42).apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(df_show[['Hari','Tanggal','Prediksi Revenue','Batas Bawah','Batas Atas']],
                     use_container_width=True, hide_index=True)

    # Tabel performa model
    st.subheader("Performa Model (evaluasi pada 30% test data)")
    perf = {
        'Model'     : ['XGBoost Tuned','RandomForest Default','RandomForest Tuned',
                       'XGBoost Default','LinearReg Ridge','LinearReg Default'],
        'RMSE (IDR)': ['875,804','890,311','894,852','977,411','1,516,198','1,662,944'],
        'R²'        : ['0.1326','0.1036','0.0944','-0.0804','-1.5998','-2.1274'],
        'MAE (IDR)' : ['629,015','632,631','639,573','777,577','1,333,407','1,479,758'],
        'MAPE'      : ['58.92%','59.26%','59.17%','80.50%','144.31%','159.43%'],
    }
    st.dataframe(
        pd.DataFrame(perf).style.apply(
            lambda x: ['background-color:#E1F5EE' if i==0 else ''
                       for i in range(len(x))], axis=0),
        use_container_width=True, hide_index=True
    )


# ════════════════════════════════════════════════════════════
#  HALAMAN 3 : CHATBOT AI AGENT
#  Arsitektur: LangChain ReAct (Reasoning + Acting)
#  Tools: Python REPL + Tavily Web Search
# ════════════════════════════════════════════════════════════
elif halaman == "Chatbot AI Agent":
    st.title("Chatbot AI Agent")
    st.info("**Penjelasan Fitur:**\nIni adalah asisten virtual pintar Anda. Anda bisa meminta AI untuk membuat grafik kustom, mencari harga kompetitor di internet, atau mendapatkan insight mendalam mengenai performa toko Anda. AI ini dilengkapi dengan arsitektur penalaran *ReAct* tingkat lanjut.", icon="💡")

    # ── Cek API Keys ──────────────────────────────────────────
    google_key = st.secrets.get("GOOGLE_API_KEY", "")
    tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    has_google = bool(google_key)
    has_tavily = bool(tavily_key)

    if not has_google:
        st.warning("**GOOGLE_API_KEY belum diset.** Tambahkan di `.streamlit/secrets.toml`:")
        st.code(
            'GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # wajib — https://aistudio.google.com/app/apikey (GRATIS)\n'
            'TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"  # opsional — https://tavily.com (GRATIS)',
            language="toml"
        )

    # ── Model selector (tampil hanya jika API key ada) ────────────
    GEMINI_MODELS = {
        "gemini-3.1-flash-lite"   : "Gemini 3.1 Flash Lite",
        "gemini-2.5-flash"   : "Gemini 2.5 Flash",
        "gemini-1.5-flash"   : "Gemini 1.5 Flash",
        "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8B",
        "gemini-2.0-flash"   : "Gemini 2.0 Flash",
    }
    selected_model = st.sidebar.selectbox(
        "Model Gemini",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda m: GEMINI_MODELS[m],
        help="Pilih model AI Anda."
    ) if has_google else "gemini-3.1-flash"

    # ── Siapkan data untuk agent ──────────────────────────────
    data_file_path   = "ecommerce_cleaned.csv"   # default: data demo
    data_source_info = "`ecommerce_cleaned.csv`"

    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        try:
            st.session_state['df_clean'].to_csv('user_data.csv', index=False)
            data_file_path   = "user_data.csv"
            # Tampilkan nama file asli yang diupload user
            file_names = st.session_state.get('uploaded_file_names', ['user_data.csv'])
            file_list = ', '.join(f'`{fn}`' for fn in file_names)
            data_source_info = f"Data upload Anda: {file_list}"
        except Exception as e:
            st.warning(f"Gagal menyimpan data upload ke CSV: {e}")

    st.info(f"**Sumber data aktif:** {data_source_info}")
    
    st.markdown("**Tips:** Jika agent error, klik **Reset Agent** di bawah chat.")

    # ── Contoh pertanyaan ─────────────────────────────────────
    with st.expander("Contoh pertanyaan yang bisa kamu tanya", expanded=False):
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("""
**Analisis Data :**
- *Tampilkan grafik revenue per bulan*
- *Produk apa yang paling laris? Buat chart*
- *Analisis penjualan berdasarkan provinsi*
- *Berapa rata-rata nilai transaksi per hari?*
- *Tren penjualan 3 bulan terakhir bagaimana?*
            """)
        with col_ex2:
            st.markdown("""
**Riset Pasar :**
- *Cari harga produk fashion wanita di Shopee*
- *Tren produk elektronik terlaris Tokopedia 2025*
- *Strategi promosi e-commerce yang efektif*
- *Kompetitor produk kecantikan di marketplace*
            """)

    st.divider()

    # ── Chat history init ─────────────────────────────────────
    if "messages" not in st.session_state:
        # Resolve the clean model name
        clean_model_name = GEMINI_MODELS.get(selected_model, selected_model)
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Halo! Saya **AI Agent Analitik E-commerce**\n\n"
                "**Kemampuan saya:**\n"
                "- Analisis data, buat grafik otomatis, hitung statistik\n"
                "- Riset harga kompetitor Shopee/Tokopedia, tren pasar\n\n"
                "Silakan tanya apa saja!"
            )
        }]

    # ── Render chat history ───────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image_bytes" in msg:
                st.image(msg["image_bytes"], use_container_width=True)

    # ── Chat input ────────────────────────────────────────────
    prompt_user = st.chat_input(
        "Tanya tentang data penjualan atau riset kompetitor..." if has_google
        else "Tambahkan GOOGLE_API_KEY untuk menggunakan chatbot"
    )

    if prompt_user:
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt_user})
        with st.chat_message("user"):
            st.markdown(prompt_user)

        # Proses jawaban
        with st.chat_message("assistant"):

            # ── Mode demo (tanpa API key) ─────────────────────
            if not has_google:
                resp = _demo_response(prompt_user)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})

            # ── Mode penuh (dengan API key) ───────────────────
            else:
                # Inisialisasi agent (cached)
                try:
                    agent_exec = _init_agent(
                        _google_api_key=google_key,
                        _tavily_api_key=tavily_key if has_tavily else "",
                        _model=selected_model
                    )
                except Exception as init_err:
                    err_msg = (
                        f"Gagal menginisialisasi AI Agent: `{init_err}`\n\n"
                        "**Kemungkinan penyebab:**\n"
                        "- GOOGLE_API_KEY tidak valid atau expired\n"
                        "- Paket LangChain belum terinstall lengkap\n\n"
                        "Coba jalankan: `pip install -r requirements.txt`"
                    )
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    st.stop()

                # ── Smart routing: deteksi tipe pertanyaan ─────────────────
                # Hindari agent salah pakai Python REPL untuk pertanyaan strategi
                q_lower = prompt_user.lower()

                # Keyword pertanyaan EKSTERNAL (butuh Web Search)
                WEB_SEARCH_KEYWORDS = [
                    "strategi", "cara ", "tips ", "promosi", "marketing",
                    "pemasaran", "kompetitor", "harga pasar", "tren terkini",
                    "shopee", "tokopedia", "lazada", "blibli", "marketplace",
                    "digital", "seo", "iklan", "konten", "media sosial",
                    "sosmed", "viral", "endorse", "influencer", "diskon efektif",
                    "cara berjualan", "cara meningkatkan", "bagaimana cara",
                    "apa yang harus", "rekomendasi toko", "harga kompetitor"
                ]

                # Keyword pertanyaan INTERNAL (butuh Python REPL / analisis data)
                DATA_KEYWORDS = [
                    "grafik", "chart", "plot", "visualisasi", "tampilkan data",
                    "berapa total", "berapa rata", "hitung", "statistik",
                    "dari data", "analisis data", "terlaris dari", "transaksi",
                    "revenue", "pendapatan", "omzet dari data"
                ]

                is_web_question  = any(kw in q_lower for kw in WEB_SEARCH_KEYWORDS)
                is_data_question = any(kw in q_lower for kw in DATA_KEYWORDS)

                # Tentukan routing hint
                if is_web_question and not is_data_question:
                    routing_hint = (
                        "[ROUTING HINT: Pertanyaan ini adalah tentang pengetahuan EKSTERNAL. "
                        "WAJIB gunakan tavily_search_results_json (Web Search) untuk menjawab. "
                        "JANGAN gunakan Python_REPL untuk pertanyaan ini.]"
                    )
                elif is_data_question and not is_web_question:
                    routing_hint = (
                        "[ROUTING HINT: Pertanyaan ini membutuhkan analisis data INTERNAL. "
                        "Gunakan Python_REPL untuk membaca dan menganalisis file CSV.]"
                    )
                else:
                    routing_hint = (
                        "[ROUTING HINT: Tentukan sendiri tool yang paling tepat berdasarkan "
                        "aturan routing di system prompt.]"
                    )

                # ── Susun history percakapan sebelumnya sebagai teks ──
                history_text = ""
                if len(st.session_state.messages) > 2: # Jika ada chat selain welcome message
                    history_text = "=== RIWAYAT PERCAKAPAN (KONTEKS) ===\n"
                    # Skip welcome msg (idx 0) dan prompt saat ini (terakhir)
                    for past_msg in st.session_state.messages[1:-1]:
                        role_name = "User" if past_msg["role"] == "user" else "AI"
                        # Batasi panjang history agar tidak over-token
                        msg_snippet = str(past_msg["content"]).strip()[:800]
                        if msg_snippet:
                            history_text += f"{role_name}: {msg_snippet}\n"
                    history_text += "====================================\n\n"

                # Susun input untuk agent — sertakan history + info data + routing hint
                agent_input = (
                    f"{history_text}"
                    f"{routing_hint}\n\n"
                    f"Konteks Data:\n"
                    f"- File data tersedia: '{data_file_path}' ({data_source_info})\n"
                    f"- Untuk load data: pd.read_csv('{data_file_path}')\n"
                    f"- Untuk grafik: simpan ke 'temp_plot.png'\n\n"
                    f"Pertanyaan user saat ini: {prompt_user}"
                )

                # Jalankan LangGraph ReAct agent
                try:
                    with st.spinner("Agent sedang berpikir (ReAct: Reasoning → Action → Observation)..."):
                        # LangGraph invoke format yang benar & aman:
                        result = agent_exec.invoke(
                            {"messages": [("human", agent_input)]}
                        )

                    # LangGraph result: dict dengan key "messages" berisi list pesan
                    # Pesan terakhir adalah jawaban final dari AI
                    all_msgs = result.get("messages", [])
                    answer = "Maaf, tidak dapat menghasilkan jawaban."
                    if all_msgs:
                        last_msg = all_msgs[-1]
                        content = getattr(last_msg, "content", str(last_msg))
                        # Terkadang Gemini mengembalikan list of dict blocks
                        if isinstance(content, list):
                            texts = [blk.get("text", "") for blk in content if isinstance(blk, dict) and "text" in blk]
                            answer = "\n".join(texts) if texts else str(content)
                        else:
                            answer = str(content)
                        
                        # Fallback jika model menolak menjawab (misal karena safety filter atau function call gagal)
                        if not answer.strip():
                            # Cek raw response metadata jika ada
                            raw_meta = getattr(last_msg, "response_metadata", {})
                            finish_reason = raw_meta.get("finish_reason", "UNKNOWN")
                            
                            if finish_reason == "MALFORMED_FUNCTION_CALL":
                                # AUTO-RETRY: Coba langsung tanpa tools (direct LLM call)
                                st.toast("MALFORMED_FUNCTION_CALL terdeteksi, mencoba mode langsung...", icon="🤖")
                                try:
                                    from langchain_google_genai import ChatGoogleGenerativeAI
                                    from langchain_core.messages import SystemMessage, HumanMessage
                                    direct_llm = ChatGoogleGenerativeAI(
                                        model=selected_model,
                                        google_api_key=google_key,
                                        temperature=0.3,
                                    )
                                    fallback_resp = direct_llm.invoke([
                                        SystemMessage(content=(
                                            "Kamu adalah AI Analyst E-commerce Indonesia. "
                                            "Jawab pertanyaan user secara informatif dalam Bahasa Indonesia. "
                                            "Jika pertanyaan butuh data CSV, beritahu user bahwa mode tool sedang bermasalah "
                                            "dan sarankan ganti model ke Gemini 1.5 Flash di sidebar."
                                        )),
                                        HumanMessage(content=prompt_user)
                                    ])
                                    fb_content = getattr(fallback_resp, "content", "")
                                    if isinstance(fb_content, list):
                                        fb_content = " ".join(b.get("text","") for b in fb_content if isinstance(b,dict))
                                    answer = (
                                        f"{fb_content}\n\n"
                                        "---\n"
                                        "Mode terbatas: Analisis data/grafik tidak tersedia karena MALFORMED_FUNCTION_CALL pada model ini. "
                                        f"Untuk analisis penuh, ganti model ke **Gemini 1.5 Flash** di sidebar."
                                    )
                                except Exception:
                                    answer = (
                                        f"Error: {finish_reason}\n\n"
                                        f"Model `{selected_model}` (Gemini 2.5 Flash) kadang mengalami bug function-call "
                                        f"saat dipasangkan dengan LangGraph.\n\n"
                                        "**Solusi cepat:** Ganti model ke **Gemini 1.5 Flash** di sidebar → klik **Reset Agent** → tanya lagi."
                                    )
                            else:
                                answer = (
                                    "**Agent mengembalikan teks kosong.**\n\n"
                                    f"Alasan penghentian (Finish Reason): `{finish_reason}`\n\n"
                                    "Coba sederhanakan pertanyaan atau ganti model di sidebar."
                                )

                    # ── Tampilkan reasoning steps dari messages (DISEMBUNYIKAN) ───
                    pass

                    # ── Cek & tampilkan grafik ─────────────────
                    image_bytes = None
                    if os.path.exists("temp_plot.png"):
                        with open("temp_plot.png", "rb") as imgf:
                            image_bytes = imgf.read()
                        st.image(image_bytes, use_container_width=True)
                        try:
                            os.remove("temp_plot.png")
                        except Exception:
                            pass

                    # ── Tampilkan jawaban final ────────────────
                    st.markdown(answer)

                    # ── Simpan ke history ──────────────────────
                    msg_entry = {"role": "assistant", "content": answer}
                    if image_bytes:
                        msg_entry["image_bytes"] = image_bytes
                    st.session_state.messages.append(msg_entry)

                except Exception as run_err:
                    err_detail = str(run_err)

                    # ── Deteksi error 429 RESOURCE_EXHAUSTED (quota habis) ────
                    if "429" in err_detail or "RESOURCE_EXHAUSTED" in err_detail or "quota" in err_detail.lower():
                         err_msg = (
                             "**Quota API Gemini Habis (429 RESOURCE_EXHAUSTED)**\n\n"
                             f"Model `{selected_model}` sudah mencapai batas free tier.\n\n"
                             "**Solusi (pilih salah satu):**\n"
                             "1. Tunggu beberapa menit lalu coba lagi (rate limit per menit)\n"
                             "2. 🤖 Ganti model di **sidebar kiri** ke `gemini-1.5-flash` atau `gemini-1.5-flash-8b`\n"
                             "3. Upgrade ke [Google AI Studio Pro](https://aistudio.google.com) untuk limit lebih tinggi\n\n"
                             "Setelah ganti model, klik **Reset Agent** lalu kirim pertanyaan lagi."
                         )
                         st.warning(err_msg)
                    else:
                         err_msg = (
                             f"**Error saat menjalankan agent:**\n\n`{err_detail[:300]}`\n\n"
                             "**Solusi:**\n"
                             "1. Klik tombol **Reset Agent** di bawah\n"
                             "2. Coba sederhanakan pertanyaan\n"
                             "3. Ganti model di sidebar jika error berulang"
                         )
                         st.error(err_msg)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg
                    })
                    # Reset cache agar agent dapat diinisialisasi ulang dengan model baru
                    _init_agent.clear()

    # ── Tombol aksi ───────────────────────────────────────────
    st.divider()
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("Bersihkan Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        if st.button("Reset Agent", use_container_width=True):
            _init_agent.clear()
            st.success("Agent cache direset. Kirim pertanyaan untuk inisialisasi ulang.")