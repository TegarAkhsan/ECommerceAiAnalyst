# ============================================================
# streamlit_app.py — AI Agent Analisis E-commerce (Gemini API & Clean UI)
# ============================================================
# 1. UI: Tema Putih & Biru Pastel (Minimalist & Profesional).
# 2. AI Agent: Menggunakan Google Gemini API (Kecepatan & Akurasi Tinggi).
# 3. Web Scraper: Mandiri tanpa Tavily API (Custom DuckDuckGo HTML).
# 4. Forecasting: Agregasi Mingguan (Weekly).
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib, json, os, warnings
from datetime import timedelta
import requests
from bs4 import BeautifulSoup
import urllib.parse

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
#  BAGIAN 1: KONFIGURASI UTAMA & UI STYLING (PUTIH & BIRU PASTEL)
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="E-commerce AI Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pemulihan otomatis saat halaman keluar dari Back-Forward Cache (bfcache)
import streamlit.components.v1 as components
components.html(
    """
    <script>
    window.addEventListener('pageshow', function(event) {
        if (event.persisted) {
            window.parent.location.reload();
        }
    });
    </script>
    """,
    height=0,
    width=0
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global typography - do NOT override icon spans */
    html, body, .stApp, .stMarkdown, p, label, input, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #0f172a !important;
    }

    /* Preserve Material Symbols */
    .material-symbols-rounded,
    .material-icons,
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-feature-settings: 'liga' !important;
        display: inline-block !important;
    }

    /* Overall Layout & Canvas */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 980px !important;
    }

    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Typography */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.35rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
    }

    /* Header subtitle component */
    .header-desc {
        color: #64748b !important;
        font-size: 0.92rem !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5 !important;
    }

    /* Minimalist GPT-style sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
        margin: 1rem 0 !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 12px 0;
    }
    .brand-avatar {
        width: 32px;
        height: 32px;
        background-color: #0f172a;
        color: #ffffff !important;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .brand-name {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a !important;
        line-height: 1.2;
    }
    .brand-role {
        font-size: 0.72rem;
        color: #64748b !important;
        line-height: 1.2;
    }

    .sidebar-section-title {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.06em !important;
        margin-top: 12px !important;
        margin-bottom: 6px !important;
        text-transform: uppercase !important;
    }

    /* Clean navigation menu (GPT list style) */
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin-bottom: 4px !important;
        color: #475569 !important;
        cursor: pointer !important;
        width: 100% !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        background-color: #edf2f7 !important;
        color: #0f172a !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* File Uploader Clean Light Styling (Fix black dropzone) */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        color: #334155 !important;
        padding: 24px 16px !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"]:hover,
    div[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    div[data-testid="stFileUploaderDropzone"] small {
        color: #64748b !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
    }

    /* Info card guide */
    .guide-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .guide-title {
        font-weight: 600;
        font-size: 0.82rem;
        color: #0f172a;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .guide-list {
        margin: 0;
        padding-left: 16px;
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.65;
    }

    /* Minimalist metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.45rem !important;
        font-weight: 600 !important;
    }

    /* Clean tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    th {
        background-color: #f8fafc !important;
        color: #334155 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
    td {
        background-color: #ffffff !important;
        color: #475569 !important;
        border-bottom: 1px solid #f1f5f9 !important;
        font-size: 0.85rem !important;
    }

    /* Conversational chat styling */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid #f1f5f9 !important;
        border-radius: 0 !important;
        padding: 16px 0 !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #fafbfc !important;
        border-radius: 10px !important;
        padding: 16px 18px !important;
    }

    /* Bulletproof GPT avatar badges (no raw ligature text leaks like 'smart_toy') */
    div[data-testid="stChatMessageAvatar"] {
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0f172a !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"] span {
        font-size: 0 !important;
        display: none !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"]::after {
        content: "AI";
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1 !important;
    }

    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
    }
    div[data-testid="stChatMessageAvatarUser"] span {
        font-size: 0 !important;
        display: none !important;
    }
    div[data-testid="stChatMessageAvatarUser"]::after {
        content: "U";
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
        line-height: 1 !important;
    }

    /* Always-visible Sidebar Toggle Buttons (Collapse & Expand) */
    div[data-testid="stSidebarHeader"],
    div[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarExpandButton"],
    button[data-testid="stSidebarExpandButton"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarExpandButton"] {
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 32px !important;
        height: 32px !important;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }

    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stSidebarExpandButton"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }

    button[data-testid="stSidebarCollapseButton"] span,
    button[data-testid="stSidebarExpandButton"] span {
        font-size: 0 !important;
        display: none !important;
    }

    button[data-testid="stSidebarCollapseButton"]::after {
        content: "«";
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        color: #334155 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    button[data-testid="stSidebarExpandButton"]::after {
        content: "»";
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        color: #334155 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #94a3b8 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    /* Neutral buttons */
    button[kind="secondary"], button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #334155 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover, button[kind="primary"]:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* Clean alerts */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #f8fafc !important;
        color: #334155 !important;
    }

    /* Sidebar footer */
    .sidebar-footer {
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
        margin-top: 24px;
    }
    .status-badge {
        font-size: 0.74rem;
        color: #059669;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
    }
    .version-text {
        font-size: 0.7rem;
        color: #94a3b8;
    }

    /* Data Analyst Insight Container & Cards */
    .analyst-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-top: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .analyst-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0369a1;
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 4px 10px;
        margin-bottom: 8px;
    }
    .analyst-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .analyst-subheading {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .analyst-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .analyst-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .analyst-card-title {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        margin-bottom: 6px;
    }
    .analyst-card-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .analyst-card-desc {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.45;
    }
    .analyst-section-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .analyst-p {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.65;
        margin-bottom: 12px;
    }
    .strategy-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3.5px solid #0284c7;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .strategy-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .strategy-body {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True
)

TRAINING_START = pd.Timestamp('2023-12-31')

@st.cache_resource
def load_artifacts():
    model  = joblib.load('best_model_weekly.pkl')
    scaler = joblib.load('scaler_weekly.pkl')
    with open('feature_columns_weekly.json') as f:
        meta = json.load(f)
    return model, scaler, meta


# ════════════════════════════════════════════════════════════
#  BAGIAN 2: FUNGSI PENGOLAHAN DATA & FORECASTING (WEEKLY FIXED)
# ════════════════════════════════════════════════════════════
from hijri_converter import Hijri, Gregorian
import datetime

def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    else:
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')

def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if 'Waktu Pengiriman Diatur' in df.columns and 'Waktu Pesanan Dibuat' not in df.columns:
        df = df.rename(columns={'Waktu Pengiriman Diatur': 'Waktu Pesanan Dibuat'})

    if 'Total Pembayaran' in df.columns:
        df['Total Pembayaran'] = df['Total Pembayaran'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df['Total Pembayaran'] = pd.to_numeric(df['Total Pembayaran'], errors='coerce')

    if 'Status Pesanan' in df.columns:
        df['Status Pesanan'] = df['Status Pesanan'].apply(lambda x: 'Selesai' if str(x).startswith('Pesanan diterima') else x)
        df = df[df['Status Pesanan'] == 'Selesai'].copy()

    if 'Total Pembayaran' in df.columns:
        df = df[df['Total Pembayaran'] > 0].copy()

    if 'Waktu Pesanan Dibuat' in df.columns:
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'], errors='coerce')
        df['order_date'] = df['Waktu Pesanan Dibuat'].dt.normalize()

    return df.reset_index(drop=True)

def make_weekly(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_order = (df_clean.drop_duplicates(subset='order_id', keep='first') if 'order_id' in df_clean.columns else df_clean.copy())
    weekly = df_order.groupby(pd.Grouper(key='order_date', freq='W'))['Total Pembayaran'].sum().reset_index()
    weekly = weekly.rename(columns={'order_date': 'order_week', 'Total Pembayaran': 'total_revenue'})
    
    full_range = pd.date_range(weekly['order_week'].min(), weekly['order_week'].max(), freq='W')
    weekly = weekly.set_index('order_week').reindex(full_range).reset_index().rename(columns={'index': 'order_week'})
    weekly['total_revenue'] = weekly['total_revenue'].fillna(0)
    return weekly.sort_values('order_week').reset_index(drop=True)

def build_features_weekly(df: pd.DataFrame, median_nonzero: float = None, training_start: pd.Timestamp = None) -> pd.DataFrame:
    df  = df.copy()
    df['order_week'] = pd.to_datetime(df['order_week'])
    
    # MURNI FITUR MINGGUAN (TIDAK ADA DAY ATAU DAYOFWEEK)
    df['year'] = df['order_week'].dt.year
    df['month'] = df['order_week'].dt.month
    df['week_of_year'] = df['order_week'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['order_week'].dt.quarter
    
    def get_ramadan_dates(year):
        hijri_year = Gregorian(year, 3, 1).to_hijri().year 
        ram_start = Hijri(hijri_year, 9, 1).to_gregorian()
        eid_day = Hijri(hijri_year, 10, 1).to_gregorian()
        return datetime.date(ram_start.year, ram_start.month, ram_start.day), datetime.date(eid_day.year, eid_day.month, eid_day.day)

    def check_harbolnas(week_end_date):
        dates_in_week = [week_end_date - datetime.timedelta(days=i) for i in range(7)]
        for d in dates_in_week:
            if d.month == 12 and d.day in [10, 11, 12]: return 1
        return 0

    is_ram_list, is_harb_list = [], []
    for dt in df['order_week'].dt.date:
        r_start, eid = get_ramadan_dates(dt.year)
        w_start = dt - datetime.timedelta(days=6)
        is_ram_list.append(1 if (w_start <= eid) and (dt >= r_start) else 0)
        is_harb_list.append(check_harbolnas(dt))
        
    df['is_ramadan'] = is_ram_list
    df['is_harbolnas'] = is_harb_list

    if 'is_forecast_placeholder' in df.columns:
        df['is_holiday_gap'] = ((df['total_revenue'] == 0) & (df['is_forecast_placeholder'] != 1)).astype(int)
    else:
        df['is_holiday_gap'] = (df['total_revenue'] == 0).astype(int)

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_year']  = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_year']  = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    ref_date = training_start if training_start is not None else df['order_week'].min()
    df['week_index'] = (df['order_week'] - ref_date).dt.days // 7

    rev_for_lag = df['total_revenue'].replace(0, median_nonzero) if (median_nonzero is not None and median_nonzero > 0) else df['total_revenue']

    # MURNI LAG MINGGUAN (Aman untuk data pendek berkat fillna(0))
    df['lag_1'] = rev_for_lag.shift(1).fillna(0)
    df['lag_4'] = rev_for_lag.shift(4).fillna(0)
    df['lag_12'] = rev_for_lag.shift(12).fillna(0)
    df['rolling_4w_mean']  = rev_for_lag.shift(1).rolling(4, min_periods=1).mean().fillna(0)
    df['rolling_12w_mean'] = rev_for_lag.shift(1).rolling(12, min_periods=1).mean().fillna(0)
    
    return df

def run_forecast_weekly(model, scaler, meta, weekly_hist: pd.DataFrame, n_weeks: int):
    df_roll    = weekly_hist.copy()
    preds      = []
    last_dt    = df_roll['order_week'].max()
    
    SCALE_COLS = meta['scale_cols']
    EXPECTED_FEATURES = meta['features'] # List fitur MINGGUAN eksak dari model

    nonzero_vals = df_roll.loc[df_roll['total_revenue'] > 0, 'total_revenue']
    median_nz = float(nonzero_vals.median()) if len(nonzero_vals) > 0 else 5_000_000.0

    for i in range(n_weeks):
        future_dt = last_dt + timedelta(weeks=i+1)
        placeholder = pd.DataFrame([{'order_week': future_dt, 'total_revenue': 0, 'is_forecast_placeholder': 1}])
        temp = pd.concat([df_roll, placeholder], ignore_index=True)
        
        # Buat fitur mingguan
        temp = build_features_weekly(temp, median_nonzero=median_nz, training_start=TRAINING_START)
        row  = temp.iloc[[-1]].copy()

        # -------------------------------------------------------------
        # PENGUNCI KOLOM (Mencegah Error 'feature_names mismatch')
        # 1. Pastikan semua kolom yang diminta model tersedia (isi 0 jika tidak ada)
        for col in EXPECTED_FEATURES:
            if col not in row.columns:
                row[col] = 0
        
        # 2. Isolasi HANYA kolom yang diminta model dengan URUTAN yang sama persis
        X = row[EXPECTED_FEATURES].copy()
        # -------------------------------------------------------------

        X_scaled = X.copy()
        scale_present = [c for c in SCALE_COLS if c in X_scaled.columns]
        if scale_present:
            X_scaled[scale_present] = scaler.transform(X[scale_present])

        # Eksekusi prediksi
        y_idr = max(0.0, float(np.expm1(model.predict(X_scaled)[0])))
        preds.append({'order_week': future_dt, 'predicted_revenue': y_idr})

        # Gabungkan hasil untuk putaran iterasi selanjutnya
        df_roll = pd.concat([df_roll, pd.DataFrame([{'order_week': future_dt, 'total_revenue': y_idr}])], ignore_index=True)
        if y_idr > 0:
            median_nz = float(df_roll.loc[df_roll['total_revenue'] > 0, 'total_revenue'].median())

    return pd.DataFrame(preds)

def generate_forecast_analyst_insights(weekly_hist: pd.DataFrame, df_forecast: pd.DataFrame, horizon: int) -> dict:
    """
    Menghasilkan analisis komprehensif layaknya seorang Data Analyst profesional:
    - Menghitung momentum tren, arah pergerakan, dan perbandingan terhadap riwayat historis
    - Mengidentifikasi titik kritis penjualan (Peak & Trough) serta estimasi kalender
    - Mengukur volatilitas dan stabilitas perkiraan pendapatan
    - Menyusun rekomendasi bisnis konkret (Stok/Inventaris, Pemasaran, dan Arus Kas)
    """
    n_hist = len(weekly_hist)
    avg_hist_all = float(weekly_hist['total_revenue'].mean()) if n_hist > 0 else 0.0
    
    recent_k = min(n_hist, max(4, horizon))
    avg_hist_recent = float(weekly_hist['total_revenue'].tail(recent_k).mean()) if n_hist > 0 else 0.0
    last_actual_rev = float(weekly_hist['total_revenue'].iloc[-1]) if n_hist > 0 else 0.0
    
    total_pred = float(df_forecast['predicted_revenue'].sum())
    avg_pred = float(df_forecast['predicted_revenue'].mean())
    min_pred = float(df_forecast['predicted_revenue'].min())
    max_pred = float(df_forecast['predicted_revenue'].max())
    
    peak_idx = df_forecast['predicted_revenue'].idxmax()
    peak_row = df_forecast.loc[peak_idx]
    peak_dt = peak_row['order_week']
    peak_rev = float(peak_row['predicted_revenue'])
    peak_dt_str = peak_dt.strftime('%d %b %Y')
    
    trough_idx = df_forecast['predicted_revenue'].idxmin()
    trough_row = df_forecast.loc[trough_idx]
    trough_dt = trough_row['order_week']
    trough_rev = float(trough_row['predicted_revenue'])
    trough_dt_str = trough_dt.strftime('%d %b %Y')
    
    growth_vs_recent_pct = ((avg_pred - avg_hist_recent) / (avg_hist_recent + 1e-9)) * 100 if avg_hist_recent > 0 else 0.0
    growth_vs_all_pct = ((avg_pred - avg_hist_all) / (avg_hist_all + 1e-9)) * 100 if avg_hist_all > 0 else 0.0
    growth_vs_last_pct = ((avg_pred - last_actual_rev) / (last_actual_rev + 1e-9)) * 100 if last_actual_rev > 0 else 0.0
    
    mid = max(1, len(df_forecast) // 2)
    h1_avg = float(df_forecast.iloc[:mid]['predicted_revenue'].mean())
    h2_avg = float(df_forecast.iloc[mid:]['predicted_revenue'].mean())
    slope_pct = ((h2_avg - h1_avg) / (h1_avg + 1e-9)) * 100
    
    if slope_pct > 3.0:
        trend_status = "Tren Menguat (Upward Trajectory)"
        trend_summary = f"Proyeksi menunjukkan laju penjualan meningkat {abs(slope_pct):.1f}% di paruh kedua horizon."
    elif slope_pct < -3.0:
        trend_status = "Tren Normalisasi (Gradual Correction)"
        trend_summary = f"Proyeksi melandai sekitar {abs(slope_pct):.1f}% di paruh kedua horizon waktu."
    else:
        trend_status = "Tren Konsisten (Steady Plateau)"
        trend_summary = f"Proyeksi bergerak stabil dalam batas normal (variasi antarfase hanya {abs(slope_pct):.1f}%)."

    peak_above_avg = ((peak_rev - avg_pred) / (avg_pred + 1e-9)) * 100
    trough_below_avg = ((avg_pred - trough_rev) / (avg_pred + 1e-9)) * 100
    
    std_pred = float(df_forecast['predicted_revenue'].std()) if len(df_forecast) > 1 else 0.0
    cv_pct = (std_pred / (avg_pred + 1e-9)) * 100
    
    if cv_pct < 12.0:
        volatility_status = "Volatilitas Rendah (Stabil & Terprediksi)"
        volatility_summary = f"Koefisien variasi {cv_pct:.1f}%, pergerakan pendapatan mingguan terpantau sangat stabil."
    elif cv_pct <= 25.0:
        volatility_status = "Volatilitas Moderat (Fluktuasi Wajar)"
        volatility_summary = f"Koefisien variasi {cv_pct:.1f}%, dinamika penjualan mencerminkan siklus transaksi ritel mingguan."
    else:
        volatility_status = "Volatilitas Tinggi (Fluktuasi Signifikan)"
        volatility_summary = f"Koefisien variasi {cv_pct:.1f}%, terdapat perbedaan omzet yang cukup tajam antarminggu."

    growth_direction = "tumbuh positif" if growth_vs_recent_pct >= 0 else "mengalami kontraksi"
    narrative_diagnosis = (
        f"Berdasarkan hasil analisis data historis dan peramalan model, rata-rata pendapatan mingguan toko untuk {horizon} minggu ke depan "
        f"diproyeksikan mencapai <strong>Rp {avg_pred:,.0f}</strong>. Angka ini mencerminkan dinamika {growth_direction} "
        f"sebesar <strong>{abs(growth_vs_recent_pct):.1f}%</strong> dibandingkan rata-rata penjualan riil {recent_k} minggu terakhir "
        f"(Rp {avg_hist_recent:,.0f}). Total akumulasi omzet yang dapat dicapai toko sepanjang horizon waktu ini diperkirakan berada di kisaran <strong>Rp {total_pred:,.0f}</strong>."
    )
    
    months_in_horizon = df_forecast['order_week'].dt.month.unique().tolist()
    seasonal_notes = []
    if any(m in [3, 4] for m in months_in_horizon):
        seasonal_notes.append("fase persiapan dan puncak musiman Ramadan / Idulfitri")
    if any(m in [11, 12] for m in months_in_horizon):
        seasonal_notes.append("momentum belanja Harbolnas dan promo akhir tahun (11.11 / 12.12)")
    
    seasonal_str = f" Periode ini turut dipengaruhi oleh {', '.join(seasonal_notes)}, yang berpotensi menjadi pendorong daya beli pelanggan." if seasonal_notes else ""
    
    narrative_pattern = (
        f"Analisis sebaran mingguan menunjukkan bahwa momen penjualan tertinggi diproyeksikan terjadi pada minggu <strong>{peak_dt_str}</strong> "
        f"dengan estimasi penerimaan mencapai <strong>Rp {peak_rev:,.0f}</strong> ({peak_above_avg:+.1f}% di atas rata-rata mingguan). "
        f"Sedangkan titik pendapatan terendah diproyeksikan terjadi pada minggu <strong>{trough_dt_str}</strong> di kisaran <strong>Rp {trough_rev:,.0f}</strong>.{seasonal_str} "
        f"Rentang selisih sebesar Rp {peak_rev - trough_rev:,.0f} memerlukan antisipasi kesiapan stok dan alokasi modal kerja yang adaptif."
    )
    
    reorder_dt = peak_dt - timedelta(days=10)
    reorder_dt_str = reorder_dt.strftime('%d %b %Y')
    
    rec_inventory = (
        f"Lakukan pemesanan ulang (restocking) barang dagang paling lambat sekitar tanggal <strong>{reorder_dt_str}</strong> "
        f"(7–10 hari sebelum minggu puncak {peak_dt_str}). Pastikan ketersediaan stok produk unggulan Anda ditingkatkan sekitar <strong>{min(50, max(15, int(peak_above_avg)))}%</strong> "
        f"untuk memenuhi potensi lonjakan pesanan dan mencegah hilangnya potensi penjualan (stockout). "
        f"Pada minggu {trough_dt_str}, tahan penumpukan stok baru untuk menjaga perputaran modal dan efisiensi ruang simpan."
    )
    
    rec_marketing = (
        f"Tingkatkan anggaran kampanye iklan toko 1 minggu menjelang <strong>{peak_dt_str}</strong> "
        f"untuk memaksimalkan tingkat konversi saat minat belanja konsumen berada di titik tertinggi. Sebaliknya, pada minggu dengan estimasi omzet lebih rendah "
        f"(sekitar {trough_dt_str}), terapkan program penawaran khusus seperti <em>bundle combo</em>, voucher toko berbatas waktu, atau penyesuaian harga khusus "
        f"guna memacu transaksi dan menjaga kontinuitas penjualan."
    )
    
    rec_cashflow = (
        f"Jadikan estimasi pendapatan terendah sebesar <strong>Rp {trough_rev:,.0f}</strong> sebagai tolok ukur batas aman likuiditas (<em>baseline safety buffer</em>) "
        f"dalam menyusun pengeluaran operasional mingguan. Siapkan cadangan kas operasional lebih dini sebelum tanggal {reorder_dt_str} "
        f"agar pengadaan barang untuk minggu puncak berjalan lancar tanpa mengganggu kesehatan arus kas toko."
    )

    return {
        'total_pred': total_pred,
        'avg_pred': avg_pred,
        'min_pred': min_pred,
        'max_pred': max_pred,
        'peak_dt_str': peak_dt_str,
        'peak_rev': peak_rev,
        'trough_dt_str': trough_dt_str,
        'trough_rev': trough_rev,
        'growth_vs_recent_pct': growth_vs_recent_pct,
        'growth_vs_all_pct': growth_vs_all_pct,
        'trend_status': trend_status,
        'trend_summary': trend_summary,
        'volatility_status': volatility_status,
        'volatility_summary': volatility_summary,
        'peak_above_avg': peak_above_avg,
        'narrative_diagnosis': narrative_diagnosis,
        'narrative_pattern': narrative_pattern,
        'rec_inventory': rec_inventory,
        'rec_marketing': rec_marketing,
        'rec_cashflow': rec_cashflow,
    }


def generate_historical_analyst_insights(df: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    total_revenue = float(weekly['total_revenue'].sum()) if len(weekly) > 0 else 0.0
    avg_weekly = float(weekly['total_revenue'].mean()) if len(weekly) > 0 else 0.0
    n_weeks = len(weekly)
    
    # Deteksi Titik Puncak dan Terendah
    if len(weekly) > 0:
        peak_idx = weekly['total_revenue'].idxmax()
        peak_row = weekly.loc[peak_idx]
        peak_date_str = peak_row['order_week'].strftime('%d %b %Y')
        peak_revenue = float(peak_row['total_revenue'])
        
        nonzero_w = weekly[weekly['total_revenue'] > 0]
        if len(nonzero_w) > 0:
            trough_idx = nonzero_w['total_revenue'].idxmin()
            trough_row = nonzero_w.loc[trough_idx]
        else:
            trough_idx = weekly['total_revenue'].idxmin()
            trough_row = weekly.loc[trough_idx]
            
        trough_date_str = trough_row['order_week'].strftime('%d %b %Y')
        trough_revenue = float(trough_row['total_revenue'])
    else:
        peak_date_str, trough_date_str = "—", "—"
        peak_revenue, trough_revenue = 0.0, 0.0

    peak_above_avg = ((peak_revenue - avg_weekly) / (avg_weekly + 1e-9)) * 100
    peak_share_pct = (peak_revenue / (total_revenue + 1e-9)) * 100

    # Total Pesanan & Average Order Value (AOV)
    total_orders = len(df['order_id'].unique()) if 'order_id' in df.columns else len(df)
    aov = total_revenue / (total_orders + 1e-9)

    # Volatilitas & Koefisien Variasi (CV)
    std_rev = float(weekly['total_revenue'].std()) if len(weekly) > 1 else 0.0
    cv_pct = (std_rev / (avg_weekly + 1e-9)) * 100

    if cv_pct > 35.0 or peak_above_avg > 75.0:
        volatility_status = "Pola Berfluktuasi Tinggi (Spike-Driven)"
        volatility_desc = f"Penjualan sangat terkonsentrasi pada minggu puncak ({peak_date_str}), yang menyumbang {peak_share_pct:.1f}% dari keseluruhan omzet toko."
    elif cv_pct > 18.0:
        volatility_status = "Pola Moderat (Dinamis)"
        volatility_desc = f"Penjualan mingguan mengalami variasi wajar dengan selisih puncak {peak_above_avg:+.1f}% di atas rata-rata."
    else:
        volatility_status = "Pola Stabil"
        volatility_desc = "Penjualan mingguan bergerak konsisten dengan fluktuasi rendah antarfase."

    # Penurunan Pasca-Puncak (Post-Peak Contraction)
    last_week_rev = float(weekly.iloc[-1]['total_revenue']) if len(weekly) > 0 else 0.0
    last_week_date = weekly.iloc[-1]['order_week'].strftime('%d %b %Y') if len(weekly) > 0 else "—"
    if peak_revenue > last_week_rev and peak_revenue > 0:
        drop_from_peak_pct = ((peak_revenue - last_week_rev) / peak_revenue) * 100
    else:
        drop_from_peak_pct = 0.0

    # Analisis Hari Transaksi Teramai
    top_day_name = "Senin"
    top_day_pct = 0.0
    top_day_count = 0
    if 'Waktu Pesanan Dibuat' in df.columns:
        valid_dates = pd.to_datetime(df['Waktu Pesanan Dibuat'], errors='coerce').dropna()
        if len(valid_dates) > 0:
            day_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
            counts = valid_dates.dt.dayofweek.map(day_map).value_counts()
            if len(counts) > 0:
                top_day_name = counts.index[0]
                top_day_count = int(counts.iloc[0])
                top_day_pct = (top_day_count / len(valid_dates)) * 100

    # Analisis Kategori / Produk Unggulan (Dinamis dari Kolom yang Tersedia)
    top_prod_name = "Produk Utama"
    top_prod_share = 100.0
    n_categories = 1
    
    prod_col_candidates = [c for c in ['product_category', 'Nama Produk', 'Kategori', 'nama_produk'] if c in df.columns]
    if prod_col_candidates:
        active_col = prod_col_candidates[0]
        prod_grp = df.groupby(active_col)['Total Pembayaran'].sum().sort_values(ascending=False)
        if len(prod_grp) > 0:
            top_prod_name = str(prod_grp.index[0])
            top_prod_share = (float(prod_grp.iloc[0]) / (total_revenue + 1e-9)) * 100
            n_categories = len(prod_grp)

    # Analisis Logistik (Dinamis)
    top_courier = "Reguler / Standar"
    top_courier_pct = 0.0
    if 'Opsi Pengiriman' in df.columns:
        c_counts = df['Opsi Pengiriman'].value_counts()
        if len(c_counts) > 0:
            top_courier = str(c_counts.index[0]).split('-')[0].strip()
            top_courier_pct = (float(c_counts.iloc[0]) / (len(df) + 1e-9)) * 100

    # Narasi Diagnosa Konteks Kalender & Musiman (100% Dinamis dari Tanggal Puncak)
    seasonal_context = ""
    if len(weekly) > 0 and peak_above_avg > 35.0:
        peak_dt_val = peak_row['order_week']
        peak_month = peak_dt_val.month
        is_peak_ramadan = False
        try:
            r_start, eid = get_ramadan_dates(peak_dt_val.year)
            w_start = peak_dt_val.date() - datetime.timedelta(days=6)
            if (w_start <= eid) and (peak_dt_val.date() >= r_start):
                is_peak_ramadan = True
        except Exception:
            pass

        if is_peak_ramadan:
            seasonal_context = " Lonjakan ini bertepatan dengan siklus musiman Ramadan dan Idulfitri, di mana intensitas belanja konsumen pada marketplace lazim mengalami eskalasi tinggi."
        elif peak_month in [11, 12]:
            seasonal_context = " Lonjakan ini selaras dengan momentum festival belanja kuartal akhir (kampanye Harbolnas 11.11 dan 12.12)."
        elif peak_month in [6, 7]:
            seasonal_context = " Lonjakan ini bertepatan dengan momentum libur tengah tahun dan persiapan tahun ajaran baru."
        else:
            seasonal_context = f" Lonjakan pada bulan {peak_dt_val.strftime('%B')} ini mengindikasikan keberhasilan kampanye promosi toko, program diskon berkala (seperti payday atau tanggal kembar), atau peningkatan eksposur produk pada minggu tersebut."

    narrative_diag = (
        f"Sepanjang periode <strong>{n_weeks} minggu</strong>, toko membukukan total pendapatan riil sebesar <strong>Rp {total_revenue:,.0f}</strong> "
        f"dari <strong>{total_orders:,} transaksi selesai</strong>, dengan rata-rata penjualan <strong>Rp {avg_weekly:,.0f} per minggu</strong> "
        f"dan nilai keranjang rata-rata (Average Order Value) sebesar <strong>Rp {aov:,.0f} per transaksi</strong>. "
        f"Titik omzet tertinggi tercatat pada minggu <strong>{peak_date_str}</strong> sebesar <strong>Rp {peak_revenue:,.0f}</strong> "
        f"({peak_above_avg:+.1f}% di atas baseline rata-rata mingguan). Satu minggu puncak ini menyumbang <strong>{peak_share_pct:.1f}%</strong> "
        f"dari total pendapatan toko.{seasonal_context}"
    )

    if drop_from_peak_pct > 15.0:
        drop_detail = f"mengalami kontraksi sebesar <strong>{drop_from_peak_pct:.1f}%</strong> ke angka <strong>Rp {last_week_rev:,.0f}</strong>"
    elif drop_from_peak_pct > 0:
        drop_detail = f"mengalami penyesuaian wajar sebesar <strong>{drop_from_peak_pct:.1f}%</strong> ke angka <strong>Rp {last_week_rev:,.0f}</strong>"
    else:
        drop_detail = f"tetap bertahan pada level optimalnya di kisaran <strong>Rp {last_week_rev:,.0f}</strong>"

    narrative_drop = (
        f"Pasca-minggu puncak, penjualan pada periode minggu terakhir ({last_week_date}) {drop_detail}. "
        f"Dinamika ini mencerminkan elastisitas permintaan toko terhadap momentum promosi, "
        f"sehingga toko memerlukan strategi stabilisasi agar perputaran kas di luar momen puncak tetap terjaga konsisten."
    )

    # Rekomendasi Aksi Data Analyst (Target AOV & Strategi Dihitung Dinamis dari Data)
    target_aov_low = aov * 1.25
    target_aov_high = aov * 1.50

    rec_retention = (
        f"Dari total <strong>{total_orders:,} pesanan</strong> yang berhasil dihimpun, minggu puncak ({peak_date_str}) merepresentasikan konsentrasi pelanggan terbesar ({peak_share_pct:.1f}% omzet). "
        f"Segera luncurkan strategi retensi dengan mengirimkan voucher pembelian kedua (repeat-order voucher) dengan masa berlaku terbatas, "
        f"guna mengonversi pembeli momentum menjadi pelanggan loyal dan mempertahankan omzet toko di minggu-minggu berikutnya."
    )

    if n_categories > 1:
        bundling_strategy = f"kombinasikan produk kategori <strong>{top_prod_name}</strong> dengan produk kategori sekunder dalam bentuk <em>bundle promo</em>"
    else:
        bundling_strategy = f"hadirkan opsi paket kuantitas bertingkat (misal: beli 2 atau 3 unit lebih hemat) pada kategori <strong>{top_prod_name}</strong>"

    rec_bundling = (
        f"Kategori <strong>{top_prod_name}</strong> merupakan kontributor omzet terbesar ({top_prod_share:.1f}%). "
        f"Dengan Average Order Value (AOV) riil saat ini sebesar <strong>Rp {aov:,.0f}</strong>, {bundling_strategy} "
        f"untuk menaikkan nilai belanja rata-rata per transaksi sebesar 25%–50% ke kisaran <strong>Rp {target_aov_low:,.0f} – Rp {target_aov_high:,.0f}</strong>."
    )

    rec_timing = (
        f"Data transaksi menunjukkan bahwa hari <strong>{top_day_name}</strong> mencatat volume pemesanan tertinggi "
        f"({top_day_count} pesanan atau {top_day_pct:.1f}% dari keseluruhan transaksi). "
        f"Fokuskan anggaran iklan toko, penayangan siaran langsung (live streaming), dan rilis kupon diskon pada hari {top_day_name} "
        f"guna mengoptimalkan rasio konversi. Pertahankan pula opsi logistik <strong>{top_courier}</strong> "
        f"yang menjadi preferensi pengiriman terbanyak ({top_courier_pct:.1f}% pesanan)."
    )

    return {
        'total_revenue': total_revenue,
        'avg_weekly': avg_weekly,
        'n_weeks': n_weeks,
        'total_orders': total_orders,
        'aov': aov,
        'peak_date_str': peak_date_str,
        'peak_revenue': peak_revenue,
        'trough_date_str': trough_date_str,
        'trough_revenue': trough_revenue,
        'peak_above_avg': peak_above_avg,
        'peak_share_pct': peak_share_pct,
        'volatility_status': volatility_status,
        'volatility_desc': volatility_desc,
        'last_week_rev': last_week_rev,
        'last_week_date': last_week_date,
        'drop_from_peak_pct': drop_from_peak_pct,
        'top_day_name': top_day_name,
        'top_day_pct': top_day_pct,
        'top_day_count': top_day_count,
        'top_prod_name': top_prod_name,
        'top_prod_share': top_prod_share,
        'top_courier': top_courier,
        'narrative_diag': narrative_diag,
        'narrative_drop': narrative_drop,
        'rec_retention': rec_retention,
        'rec_bundling': rec_bundling,
        'rec_timing': rec_timing,
    }


# ════════════════════════════════════════════════════════════
#  BAGIAN 3: AI AGENT LOGIC (DIIMPOR DARI agent_core.py)
# ════════════════════════════════════════════════════════════
from agent_core import (
    custom_marketplace_scraper,
    python_repl,
    get_agent_tools,
    build_agent,
    demo_response as _demo_response,
    extract_text_content as _extract_text_content,
    parse_ai_error_details,
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
)

def render_error_diagnostics(diag: dict):
    """Menampilkan kotak diagnosa kegagalan koneksi AI yang rapi, profesional, dan solutif."""
    html = f"""
    <div style="background-color: #ffffff; border: 1px solid #fee2e2; border-left: 4px solid {diag['badge_color']}; border-radius: 8px; padding: 16px 18px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{diag['title']}</span>
            <span style="font-size: 0.72rem; font-weight: 600; color: {diag['badge_color']}; background: {diag['badge_bg']}; padding: 2px 8px; border-radius: 999px;">{diag['badge']}</span>
        </div>
        <div style="font-size: 0.84rem; color: #334155; margin-bottom: 8px; line-height: 1.55;">
            <strong>Penyebab:</strong> {diag['cause']}
        </div>
        <div style="font-size: 0.84rem; color: #0f172a; margin-bottom: 12px; line-height: 1.55; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <strong>Solusi yang Disarankan:</strong> {diag['solution']}
        </div>
        <details style="font-size: 0.76rem; color: #64748b; cursor: pointer;">
            <summary style="font-weight: 500; color: #64748b;">Rincian Pesan Teknis</summary>
            <pre style="background: #f1f5f9; padding: 10px; border-radius: 6px; margin-top: 6px; overflow-x: auto; white-space: pre-wrap; font-family: monospace; font-size: 0.72rem; color: #334155;">{diag['technical']}</pre>
        </details>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  BAGIAN 4: STREAMLIT CACHED AGENT INITIALIZATION
# ════════════════════════════════════════════════════════════

@st.cache_resource
def _init_agent(_google_api_key: str, _model: str = "gemini-3.5-flash-lite"):
    """
    Wrapper Streamlit cache_resource yang memanggil build_agent dari agent_core.py.
    Mempertahankan kompatibilitas penuh dengan pemanggilan UI dan fungsi .clear().
    """
    return build_agent(google_api_key=_google_api_key, model=_model)



# ════════════════════════════════════════════════════════════
#  BAGIAN 5: ROUTING HALAMAN STREAMLIT
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-avatar">A</div>
            <div>
                <div class="brand-name">AI Analyst</div>
                <div class="brand-role">E-commerce Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-section-title">MENU UTAMA</div>', unsafe_allow_html=True)
    menu_options = ["Upload Data", "Prediksi Penjualan", "Chatbot AI Agent", "Pengaturan"]
    if "nav_menu" not in st.session_state:
        st.session_state["nav_menu"] = "Upload Data"

    current_idx = menu_options.index(st.session_state["nav_menu"]) if st.session_state["nav_menu"] in menu_options else 0
    halaman = st.radio(
        "Menu Utama",
        menu_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    st.session_state["nav_menu"] = halaman
    
    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="status-badge"><span class="status-dot"></span> Sistem Aktif</div>
            <div class="version-text">Versi 4.1 · Gemini Powered</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── HALAMAN 1: UPLOAD DATA ────────────────────────────────────
if halaman == "Upload Data":
    st.markdown('<h1>Upload Data Transaksi</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-desc">Unggah laporan pesanan toko (format Excel atau CSV) untuk memproses histori transaksi, mengekstrak tren penjualan, dan mengaktifkan AI Analyst.</div>', unsafe_allow_html=True)

    col_upload, col_info = st.columns([1.5, 1], gap="large")
    with col_upload:
        uploaded_files = st.file_uploader(
            "Pilih file laporan pesanan",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Mendukung file ekspor Seller Centre berformat .xlsx, .xls, dan .csv"
        )
    with col_info:
        st.markdown(
            (
                '<div class="guide-card">'
                '<div class="guide-title">Panduan Berkas</div>'
                '<ul class="guide-list">'
                '<li><strong>Format:</strong> CSV, XLSX, atau XLS hasil ekspor Seller Centre.</li>'
                '<li><strong>Kolom Kunci:</strong> Waktu Pesanan Dibuat, Total Pembayaran, Status Pesanan.</li>'
                '<li><strong>Pembersihan:</strong> Sistem otomatis menyaring pesanan selesai dan menangani multi-item order.</li>'
                '</ul>'
                '</div>'
            ),
            unsafe_allow_html=True
        )

    if uploaded_files:
        all_raw, failed = [], []
        with st.spinner("Memproses file..."):
            for f in uploaded_files:
                try:
                    df_raw = read_file(f)
                    all_raw.append(df_raw)
                except Exception as e:
                    failed.append(f"{f.name}: {e}")

        if all_raw:
            df_combined = pd.concat(all_raw, ignore_index=True)
            df_clean    = clean_raw(df_combined)

            if len(df_clean) > 0:
                st.session_state['df_clean']   = df_clean
                st.session_state['weekly_hist'] = make_weekly(df_clean)
                st.success(f"Berhasil memproses {len(df_clean):,} transaksi pesanan selesai.")
            else:
                st.error("Tidak ada transaksi berstatus 'Selesai' yang ditemukan.")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
        weekly = st.session_state['weekly_hist']

        h_insights = generate_historical_analyst_insights(df, weekly)

        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Omzet", f"Rp {h_insights['total_revenue']:,.0f}")
        k2.metric("Total Pesanan", f"{h_insights['total_orders']:,}")
        k3.metric("Rata-rata Mingguan", f"Rp {h_insights['avg_weekly']:,.0f}")
        k4.metric("Average Order Value", f"Rp {h_insights['aov']:,.0f}", help="Nilai transaksi rata-rata tiap pesanan pelanggan")
        k5.metric("Puncak Penjualan", f"Rp {h_insights['peak_revenue']:,.0f}", delta=h_insights['peak_date_str'], delta_color="off")

        st.subheader("Tren Penjualan Mingguan & Baseline")
        import plotly.graph_objects as go
        fig_trend = go.Figure()

        # Area chart penjualan aktual
        fig_trend.add_trace(go.Scatter(
            x=weekly['order_week'],
            y=weekly['total_revenue'],
            mode='lines+markers',
            name='Omzet Aktual',
            line=dict(color='#0f172a', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(15, 23, 42, 0.06)',
            marker=dict(size=6, color='#0f172a'),
            hovertemplate='<b>Minggu %{x|%d %b %Y}</b><br>Omzet: Rp %{y:,.0f}<extra></extra>'
        ))

        # Garis Baseline Rata-rata
        fig_trend.add_trace(go.Scatter(
            x=weekly['order_week'],
            y=[h_insights['avg_weekly']] * len(weekly),
            mode='lines',
            name=f"Rata-rata (Rp {h_insights['avg_weekly']:,.0f})",
            line=dict(color='#dc2626', width=1.5, dash='dot'),
            hovertemplate=f"Rata-rata: Rp {h_insights['avg_weekly']:,.0f}<extra></extra>"
        ))

        # Anotasi Titik Puncak
        if len(weekly) > 0 and h_insights['peak_revenue'] > 0:
            peak_point_x = weekly.loc[weekly['total_revenue'].idxmax()]['order_week']
            fig_trend.add_annotation(
                x=peak_point_x,
                y=h_insights['peak_revenue'],
                text=f"Puncak: Rp {h_insights['peak_revenue']:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='#0284c7',
                ax=0,
                ay=-35,
                bgcolor='#ffffff',
                bordercolor='#0284c7',
                borderwidth=1,
                borderpad=4,
                font=dict(family='Inter, sans-serif', size=11, color='#0f172a')
            )

        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#475569',
            font_family='Inter, sans-serif',
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor='#f1f5f9', title=None, tickformat='%d %b %Y'),
            yaxis=dict(gridcolor='#f1f5f9', title=None, tickprefix='Rp ', tickformat=','),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── PANEL INSIGHT DATA ANALYST ──────────────────────────────
        analyst_hist_html = (
            f'<div class="analyst-box">'
            f'<div class="analyst-badge">Data Analyst Business Insights & Diagnostics</div>'
            f'<div class="analyst-heading">Diagnosa Kinerja & Pola Penjualan Historis</div>'
            f'<div class="analyst-subheading">Interpretasi komprehensif terhadap dinamika transaksi toko, pembedahan faktor lonjakan omzet, dan rekomendasi aksi strategis.</div>'
            f'<div class="analyst-grid">'
            f'<div class="analyst-card">'
            f'<div class="analyst-card-title">Dinamika Puncak Penjualan</div>'
            f'<div class="analyst-card-value">{h_insights["peak_date_str"]}</div>'
            f'<div class="analyst-card-desc">Omzet mencapai <strong>Rp {h_insights["peak_revenue"]:,.0f}</strong> ({h_insights["peak_above_avg"]:+.1f}% di atas rata-rata), menyumbang {h_insights["peak_share_pct"]:.1f}% dari keseluruhan omzet toko.</div>'
            f'</div>'
            f'<div class="analyst-card">'
            f'<div class="analyst-card-title">Stabilitas & Konsentrasi Omzet</div>'
            f'<div class="analyst-card-value">{h_insights["volatility_status"]}</div>'
            f'<div class="analyst-card-desc">{h_insights["volatility_desc"]}</div>'
            f'</div>'
            f'<div class="analyst-card">'
            f'<div class="analyst-card-title">Profil Transaksi & Pelanggan</div>'
            f'<div class="analyst-card-value">AOV Rp {h_insights["aov"]:,.0f} · Hari Teramai: {h_insights["top_day_name"]}</div>'
            f'<div class="analyst-card-desc">Hari {h_insights["top_day_name"]} menyumbang {h_insights["top_day_count"]} pesanan ({h_insights["top_day_pct"]:.1f}%). Ekspedisi utama: {h_insights["top_courier"]}.</div>'
            f'</div>'
            f'</div>'
            f'<div class="analyst-section-title">1. Diagnosa Pola & Dinamika Lonjakan Omzet</div>'
            f'<p class="analyst-p">{h_insights["narrative_diag"]}</p>'
            f'<p class="analyst-p">{h_insights["narrative_drop"]}</p>'
            f'<div class="analyst-section-title">2. Rekomendasi Aksi Strategis Bisnis Toko</div>'
            f'<div class="strategy-card">'
            f'<div class="strategy-title">Retensi Pembeli Pasca-Puncak (Post-Peak Retention)</div>'
            f'<div class="strategy-body">{h_insights["rec_retention"]}</div>'
            f'</div>'
            f'<div class="strategy-card">'
            f'<div class="strategy-title">Peningkatan Nilai Keranjang Belanja (AOV Enhancement & Bundling)</div>'
            f'<div class="strategy-body">{h_insights["rec_bundling"]}</div>'
            f'</div>'
            f'<div class="strategy-card">'
            f'<div class="strategy-title">Optimalisasi Hari Kampanye & Logistik (Timing & Channel Strategy)</div>'
            f'<div class="strategy-body">{h_insights["rec_timing"]}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(analyst_hist_html, unsafe_allow_html=True)

        # Konsultasi AI Real-time Opsional jika API Key tersedia
        google_api_key = str(st.session_state.get("google_api_key", "")).strip()
        if google_api_key:
            with st.expander("Konsultasi Tambahan dari Google Gemini AI", expanded=False):
                st.caption("AI Gemini akan membaca ringkasan metrik data toko Anda secara langsung dan memberikan analisis kualitatif real-time.")
                if st.button("Hasilkan Analisis AI Real-time", key="btn_ai_upload_insight"):
                    with st.spinner("AI Gemini sedang menganalisis data toko Anda..."):
                        try:
                            from langchain_google_genai import ChatGoogleGenerativeAI
                            curr_upload_model = st.session_state.get("gemini_model", "gemini-3.5-flash-lite")
                            llm_direct = ChatGoogleGenerativeAI(
                                model=curr_upload_model,
                                google_api_key=google_api_key,
                                temperature=0.2
                            )
                            ai_prompt = (
                                f"Kamu adalah Senior E-commerce Data Analyst. Berikan evaluasi strategis bisnis yang ringkas, tajam, dan aplikatif berdasarkan metrik performa toko berikut:\n"
                                f"- Total Omzet: Rp {h_insights['total_revenue']:,.0f}\n"
                                f"- Total Transaksi Selesai: {h_insights['total_orders']:,} pesanan\n"
                                f"- Nilai Keranjang Rata-rata (AOV): Rp {h_insights['aov']:,.0f}\n"
                                f"- Rata-rata Mingguan: Rp {h_insights['avg_weekly']:,.0f}\n"
                                f"- Minggu Puncak Penjualan: {h_insights['peak_date_str']} (Rp {h_insights['peak_revenue']:,.0f}, kontribusi {h_insights['peak_share_pct']:.1f}%)\n"
                                f"- Hari Tersibuk: {h_insights['top_day_name']} ({h_insights['top_day_count']} pesanan)\n"
                                f"- Produk Utama: {h_insights['top_prod_name']} ({h_insights['top_prod_share']:.1f}% omzet)\n"
                                f"- Opsi Pengiriman Favorit: {h_insights['top_courier']}\n\n"
                                f"Berikan 3 poin evaluasi profesional: 1) Diagnosa bottleneck atau peluang toko, 2) Taktik ekspansi omzet di hari biasa, 3) Rekomendasi aksi promosi."
                            )
                            ai_resp = llm_direct.invoke(ai_prompt)
                            ai_text = _extract_text_content(ai_resp.content)
                            st.markdown(
                                f"""
                                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #0284c7; border-radius: 8px; padding: 16px 18px; margin-top: 12px;">
                                    <div style="font-weight: 600; color: #0f172a; font-size: 0.92rem; margin-bottom: 8px;">Konsultasi Khusus AI Gemini ({curr_upload_model})</div>
                                    <div style="font-size: 0.85rem; color: #334155; line-height: 1.65;">{ai_text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        except Exception as ai_e:
                            curr_upload_model = st.session_state.get("gemini_model", "gemini-3.5-flash-lite")
                            diag = parse_ai_error_details(ai_e, curr_upload_model)
                            render_error_diagnostics(diag)

        # ── TABS PENDALAMAN DATA ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        tab_day, tab_geo, tab_raw = st.tabs([
            "Pola Hari Transaksi",
            "Sebaran Wilayah & Logistik",
            "Sampel Data Bersih"
        ])

        with tab_day:
            col_t1, col_t2 = st.columns([1.8, 1])
            with col_t1:
                if 'Waktu Pesanan Dibuat' in df.columns:
                    valid_dates = pd.to_datetime(df['Waktu Pesanan Dibuat'], errors='coerce').dropna()
                    day_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                    day_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                    day_counts = valid_dates.dt.dayofweek.map(day_map).value_counts().reindex(day_order, fill_value=0).reset_index()
                    day_counts.columns = ['Hari', 'Jumlah Pesanan']

                    fig_day = px.bar(
                        day_counts,
                        x='Hari',
                        y='Jumlah Pesanan',
                        text='Jumlah Pesanan',
                        color='Jumlah Pesanan',
                        color_continuous_scale=['#cbd5e1', '#0284c7']
                    )
                    fig_day.update_traces(
                        textposition='outside',
                        hovertemplate='<b>%{x}</b>: %{y} pesanan<extra></extra>'
                    )
                    fig_day.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#475569',
                        font_family='Inter, sans-serif',
                        margin=dict(l=10, r=10, t=15, b=10),
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor='#f1f5f9', title=None),
                        yaxis=dict(gridcolor='#f1f5f9', title=None)
                    )
                    st.plotly_chart(fig_day, use_container_width=True)
                else:
                    st.info("Informasi waktu transaksi tidak tersedia untuk analisis harian.")

            with col_t2:
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:10px;">
                        <div style="font-weight:600; font-size:0.88rem; color:#0f172a; margin-bottom:6px;">Insight Waktu Belanja</div>
                        <div style="font-size:0.82rem; color:#475569; line-height:1.6;">
                            Hari <strong>{h_insights['top_day_name']}</strong> merupakan hari paling produktif dengan <strong>{h_insights['top_day_count']} transaksi</strong> ({h_insights['top_day_pct']:.1f}% total pesanan).
                            <br><br>
                            <strong>Rekomendasi Operasional:</strong>
                            <ul style="margin:6px 0 0 16px; padding:0;">
                                <li>Jadwalkan peluncuran voucher toko pada awal pekan.</li>
                                <li>Pastikan tim pengemasan dan ketersediaan stok siap optimal di hari {h_insights['top_day_name']}.</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_geo:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("**Top 5 Provinsi Pembeli**")
                if 'Provinsi' in df.columns:
                    top_prov = df['Provinsi'].value_counts().head(5).reset_index()
                    top_prov.columns = ['Provinsi', 'Jumlah Pesanan']
                    fig_prov = px.bar(
                        top_prov,
                        y='Provinsi',
                        x='Jumlah Pesanan',
                        orientation='h',
                        text='Jumlah Pesanan',
                        color_discrete_sequence=['#0f172a']
                    )
                    fig_prov.update_traces(textposition='outside')
                    fig_prov.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#475569',
                        font_family='Inter, sans-serif',
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(autorange='reversed', title=None),
                        xaxis=dict(gridcolor='#f1f5f9', title=None)
                    )
                    st.plotly_chart(fig_prov, use_container_width=True)
                else:
                    st.info("Data provinsi tidak tersedia.")

            with col_g2:
                st.markdown("**Opsi Pengiriman Terpopuler**")
                if 'Opsi Pengiriman' in df.columns:
                    top_ship = df['Opsi Pengiriman'].value_counts().head(5).reset_index()
                    top_ship.columns = ['Opsi Pengiriman', 'Jumlah Pesanan']
                    fig_ship = px.bar(
                        top_ship,
                        y='Opsi Pengiriman',
                        x='Jumlah Pesanan',
                        orientation='h',
                        text='Jumlah Pesanan',
                        color_discrete_sequence=['#0284c7']
                    )
                    fig_ship.update_traces(textposition='outside')
                    fig_ship.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#475569',
                        font_family='Inter, sans-serif',
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(autorange='reversed', title=None),
                        xaxis=dict(gridcolor='#f1f5f9', title=None)
                    )
                    st.plotly_chart(fig_ship, use_container_width=True)
                else:
                    st.info("Data opsi pengiriman tidak tersedia.")

        with tab_raw:
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Unduh Data Bersih (.csv)",
                data=csv_data,
                file_name="data_penjualan_bersih.csv",
                mime="text/csv"
            )

# ── HALAMAN 2: PREDIKSI PENJUALAN ──────────────────────────
elif halaman == "Prediksi Penjualan":
    st.markdown('<h1>Prediksi Penjualan Mingguan</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-desc">Estimasi pendapatan toko untuk beberapa minggu ke depan guna membantu perencanaan stok barang, target omzet, dan strategi bisnis Anda.</div>', unsafe_allow_html=True)

    if 'weekly_hist' not in st.session_state:
        st.markdown(
            (
                '<div class="guide-card" style="margin-top: 1rem;">'
                '<div class="guide-title">Data Penjualan Belum Tersedia</div>'
                '<p style="font-size: 0.9rem; color: #475569; margin-bottom: 12px; line-height: 1.6;">'
                'Untuk melihat estimasi proyeksi penjualan toko Anda, sistem membutuhkan data transaksi sebelumnya.'
                '</p>'
                '<ul class="guide-list" style="margin-bottom: 8px;">'
                '<li>Buka menu <strong>Upload Data</strong> di panel sebelah kiri.</li>'
                '<li>Unggah berkas laporan pesanan toko (format Excel atau CSV).</li>'
                '<li>Kembali ke menu ini untuk melihat grafik tren dan estimasi pendapatan mingguan.</li>'
                '</ul>'
                '</div>'
            ),
            unsafe_allow_html=True
        )
        st.stop()

    try:
        model, scaler, meta = load_artifacts()
    except Exception:
        st.error("Model machine learning mingguan belum ditemukan di direktori project.")
        st.stop()

    weekly_hist = st.session_state['weekly_hist']
    
    # --- PERBAIKAN VALIDASI: Cukup 2 minggu data untuk memicu Prediksi ---
    if len(weekly_hist) < 2:
        st.error("Data transaksi terlalu sedikit. Diperlukan minimal 2 minggu data historis agar AI dapat melihat tren dasar.")
        st.stop()
    elif len(weekly_hist) < 12:
        st.warning("**Volume Data Terbatas:** Anda mengunggah data kurang dari 12 minggu (3 Bulan). AI tetap akan memproyeksikan penjualan berdasarkan tren jangka pendek Anda, namun akurasi untuk membaca pola musiman jangka panjang mungkin terbatas.")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        horizon = st.selectbox(
            "Pilih Horizon Peramalan:",
            [4, 8, 12],
            format_func=lambda w: f"{w} Minggu ({w//4} Bulan ke depan)"
        )

    with st.spinner("Menghitung proyeksi peramalan..."):
        df_forecast = run_forecast_weekly(model, scaler, meta, weekly_hist, horizon)

    # Analisis Komprehensif Data Analyst
    insights = generate_forecast_analyst_insights(weekly_hist, df_forecast, horizon)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Prediksi Revenue", f"Rp {insights['total_pred']:,.0f}", help=f"Akumulasi proyeksi selama {horizon} minggu ke depan")
    c2.metric("Rata-rata Mingguan", f"Rp {insights['avg_pred']:,.0f}", delta=f"{insights['growth_vs_recent_pct']:+.1f}% vs historis")
    c3.metric("Estimasi Puncak", f"Rp {insights['peak_rev']:,.0f}", delta=insights['peak_dt_str'], delta_color="off")
    c4.metric("Estimasi Terendah", f"Rp {insights['trough_rev']:,.0f}", delta=insights['trough_dt_str'], delta_color="off")

    import plotly.graph_objects as go
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(
        x=weekly_hist['order_week'],
        y=weekly_hist['total_revenue'],
        mode='lines+markers',
        name='Aktual',
        line=dict(color='#0f172a', width=2),
        marker=dict(size=4, color='#0f172a'),
        hovertemplate='<b>Aktual:</b> Rp %{y:,.0f}<extra></extra>'
    ))
    fig_combined.add_trace(go.Scatter(
        x=df_forecast['order_week'],
        y=df_forecast['predicted_revenue'],
        mode='lines+markers',
        name='Prediksi Model',
        line=dict(color='#0284c7', width=2.5, dash='dash'),
        marker=dict(size=5, color='#0284c7'),
        hovertemplate='<b>Prediksi:</b> Rp %{y:,.0f}<extra></extra>'
    ))
    fig_combined.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#475569',
        font_family='Inter, sans-serif',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor='#f1f5f9', title=None, tickformat='%d %b %Y'),
        yaxis=dict(gridcolor='#f1f5f9', title=None, tickprefix='Rp ', tickformat=','),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    # ── PANEL INSIGHT DATA ANALYST ──────────────────────────────
    analyst_html = (
        f'<div class="analyst-box">'
        f'<div class="analyst-badge">Data Analyst Insight & Intelligence</div>'
        f'<div class="analyst-heading">Analisis & Panduan Keputusan Bisnis</div>'
        f'<div class="analyst-subheading">Interpretasi komprehensif terhadap pola proyeksi penjualan, perbandingan tren historis, dan panduan keputusan bisnis.</div>'
        f'<div class="analyst-grid">'
        f'<div class="analyst-card">'
        f'<div class="analyst-card-title">Arah & Momentum Tren</div>'
        f'<div class="analyst-card-value">{insights["trend_status"]}</div>'
        f'<div class="analyst-card-desc">{insights["trend_summary"]}</div>'
        f'</div>'
        f'<div class="analyst-card">'
        f'<div class="analyst-card-title">Jendela Puncak Penjualan</div>'
        f'<div class="analyst-card-value">{insights["peak_dt_str"]}</div>'
        f'<div class="analyst-card-desc">Estimasi omzet mencapai <strong>Rp {insights["peak_rev"]:,.0f}</strong> ({insights["peak_above_avg"]:+.1f}% di atas rata-rata).</div>'
        f'</div>'
        f'<div class="analyst-card">'
        f'<div class="analyst-card-title">Stabilitas & Volatilitas</div>'
        f'<div class="analyst-card-value">{insights["volatility_status"]}</div>'
        f'<div class="analyst-card-desc">{insights["volatility_summary"]}</div>'
        f'</div>'
        f'</div>'
        f'<div class="analyst-section-title">1. Diagnosa Pola & Dinamika Penjualan</div>'
        f'<p class="analyst-p">{insights["narrative_diagnosis"]}</p>'
        f'<p class="analyst-p">{insights["narrative_pattern"]}</p>'
        f'<div class="analyst-section-title">2. Rekomendasi Aksi Strategis Toko</div>'
        f'<div class="strategy-card">'
        f'<div class="strategy-title">Rantai Pasok & Manajemen Inventaris</div>'
        f'<div class="strategy-body">{insights["rec_inventory"]}</div>'
        f'</div>'
        f'<div class="strategy-card">'
        f'<div class="strategy-title">Pemasaran & Kampanye Promosi</div>'
        f'<div class="strategy-body">{insights["rec_marketing"]}</div>'
        f'</div>'
        f'<div class="strategy-card">'
        f'<div class="strategy-title">Pengendalian Arus Kas & Keuangan</div>'
        f'<div class="strategy-body">{insights["rec_cashflow"]}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(analyst_html, unsafe_allow_html=True)

    with st.expander("Rincian Tabel Prediksi Mingguan"):
        df_display = df_forecast.copy()
        df_display['Periode Minggu'] = df_display['order_week'].dt.strftime('%d %b %Y')
        df_display['Prediksi Pendapatan'] = df_display['predicted_revenue'].apply(lambda v: f"Rp {v:,.0f}")
        st.dataframe(df_display[['Periode Minggu', 'Prediksi Pendapatan']], use_container_width=True, hide_index=True)

# ── HALAMAN 3: CHATBOT AI AGENT (GEMINI) ─────────────────────────
elif halaman == "Chatbot AI Agent":
    # Inisialisasi state konfigurasi AI jika belum ada
    if "google_api_key" not in st.session_state:
        st.session_state["google_api_key"] = st.secrets.get("GOOGLE_API_KEY", "")
    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = "gemini-3.5-flash-lite"

    google_key = str(st.session_state.get("google_api_key", "")).strip()
    has_google = bool(google_key)
    selected_model = st.session_state.get("gemini_model", "gemini-3.5-flash-lite")
    model_options = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.5-flash"
    ]
    current_model_idx = model_options.index(selected_model) if selected_model in model_options else 0

    col_hdr, col_top_reset = st.columns([3.6, 1.4], vertical_alignment="center")
    with col_hdr:
        st.markdown('<h1>AI Analyst Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<div class="header-desc">Konsultasikan performa penjualan toko atau lakukan riset harga pasar daring secara interaktif.</div>', unsafe_allow_html=True)
    with col_top_reset:
        if st.button("Mulai Obrolan Baru", use_container_width=True, help="Bersihkan riwayat obrolan dan mulai sesi baru"):
            st.session_state.messages = []
            st.session_state["last_ai_status"] = "ok"
            _init_agent.clear()
            st.rerun()

    # Sidebar konfigurasi AI (Status informatif)
    st.sidebar.markdown('<div class="sidebar-section-title">STATUS ENGINE AI</div>', unsafe_allow_html=True)
    if has_google:
        status_color = "#16a34a" if st.session_state.get("last_ai_status", "ok") == "ok" else "#dc2626"
        status_bg = "#dcfce7" if st.session_state.get("last_ai_status", "ok") == "ok" else "#fee2e2"
        status_text = "● Terkoneksi" if st.session_state.get("last_ai_status", "ok") == "ok" else "● Terkendala"
        st.sidebar.markdown(
            f"""
            <div style="background-color: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.76rem; font-weight: 600; color: #0f172a;">Gemini Engine</span>
                    <span style="font-size: 0.68rem; color: {status_color}; font-weight: 600; background: {status_bg}; padding: 2px 7px; border-radius: 999px;">{status_text}</span>
                </div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">Model: <b style="color: #0f172a;">{selected_model}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.sidebar.button("Reset Percakapan", use_container_width=True, key="btn_sidebar_reset"):
            st.session_state.messages = []
            st.session_state["last_ai_status"] = "ok"
            _init_agent.clear()
            st.rerun()
    else:
        st.sidebar.markdown(
            """
            <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.76rem; font-weight: 600; color: #92400e;">Gemini Engine</span>
                    <span style="font-size: 0.68rem; color: #d97706; font-weight: 600; background: #fef3c7; padding: 2px 7px; border-radius: 999px;">● Belum Diset</span>
                </div>
                <div style="font-size: 0.72rem; color: #b45309; margin-top: 4px;">Atur di menu Pengaturan</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Banner info jika API Key belum diset
    if not has_google:
        col_warn, col_to_setting = st.columns([3.2, 1], vertical_alignment="center")
        with col_warn:
            st.warning("**Google API Key belum diset.** Buka halaman **Pengaturan** untuk memasukkan API Key agar asisten AI dapat memproses analisis data Anda.")
        with col_to_setting:
            if st.button("Buka Pengaturan", use_container_width=True, type="primary"):
                st.session_state["nav_menu"] = "Pengaturan"
                st.rerun()

    # Save Data for Agent
    data_file_path = "ecommerce_cleaned.csv"
    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        try:
            st.session_state['df_clean'].to_csv('user_data.csv', index=False)
            data_file_path = "user_data.csv"
        except Exception:
            pass

    # Inisialisasi Riwayat Percakapan
    if "messages" not in st.session_state or not st.session_state.messages or "scraping harga" in st.session_state.messages[0].get("content", ""):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Halo, saya AI Analyst toko Anda. Saya siap membantu memeriksa ringkasan transaksi toko, menganalisis produk terlaris, pola omzet mingguan, ataupun mencari harga pasar kompetitor di internet."
        }]

    # Ide pertanyaan awal sederhana jika percakapan masih baru
    preset_chosen = None
    if len(st.session_state.messages) <= 1:
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 600; color: #475569; margin-bottom: 8px;">
                Ide Pertanyaan Awal yang Dapat Dicoba:
            </div>
            """,
            unsafe_allow_html=True
        )
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if st.button("Total Penjualan", use_container_width=True, help="Ringkasan omzet dan transaksi toko"):
                preset_chosen = "Berapa total penjualan dan jumlah transaksi yang sudah berhasil di toko saya?"
        with col_q2:
            if st.button("5 Produk Terlaris", use_container_width=True, help="Daftar produk dengan penjualan terbanyak"):
                preset_chosen = "Sebutkan 5 produk yang paling laris dan berapa kontribusinya terhadap omzet toko."
        with col_q3:
            if st.button("Hari Belanja Teramai", use_container_width=True, help="Hari dengan volume pesanan tertinggi"):
                preset_chosen = "Di hari apa toko saya menerima pesanan paling banyak dan bagaimana sarannya?"
        with col_q4:
            if st.button("Cek Harga Pasar", use_container_width=True, help="Riset harga pembanding di marketplace"):
                preset_chosen = "Cari kisaran harga pasaran celengan atau produk sejenis di Shopee dan Tokopedia."

    # Tampilkan percakapan
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image_bytes" in msg:
                st.image(msg["image_bytes"], use_container_width=True)

    # Input pengguna
    prompt_user = st.chat_input("Ketik pertanyaan Anda...")
    active_prompt = preset_chosen if preset_chosen else prompt_user

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            if not has_google:
                resp = _demo_response(active_prompt)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                agent_exec = None
                try:
                    agent_exec = _init_agent(_google_api_key=google_key, _model=selected_model)
                except Exception as init_err:
                    st.session_state["last_ai_status"] = "error"
                    diag = parse_ai_error_details(init_err, selected_model)
                    render_error_diagnostics(diag)
                    if st.button("Ubah Model di Pengaturan", key="btn_goto_sett_init"):
                        st.session_state["nav_menu"] = "Pengaturan"
                        st.rerun()
                    st.stop()

                agent_input = (
                    f"Data transaksi toko tersimpan pada berkas: '{data_file_path}'.\n"
                    f"Pertanyaan Pemilik Toko: {active_prompt}\n\n"
                    f"PANDUAN KOMUNIKASI: Jangan pernah menyebutkan nama berkas fisik teknis ('{data_file_path}' atau lainnya) di dalam teks jawaban. "
                    f"Sebut selalu sebagai 'Data Penjualan Toko Anda' atau 'Laporan Transaksi Toko'."
                )

                try:
                    with st.status("AI Analyst sedang menganalisis...", expanded=True) as status_box:
                        st.write("1. Mengidentifikasi kebutuhan analisis bisnis...")
                        
                        is_market_query = any(k in active_prompt.lower() for k in ["harga", "pasar", "kompetitor", "pesaing", "shopee", "tokopedia", "riset", "luar", "online"])
                        if is_market_query:
                            st.write("2. Menghimpun data pembanding harga produk dari marketplace daring...")
                        else:
                            st.write("2. Membaca dan membedah ringkasan data transaksi toko...")
                            
                        result = agent_exec.invoke({"messages": [("human", agent_input)]})
                        
                        st.write("3. Merumuskan wawasan bisnis dan rekomendasi keputusan...")
                        status_box.update(label="Analisis Selesai", state="complete", expanded=False)
                    
                    all_msgs = result.get("messages", [])
                    raw_content = getattr(all_msgs[-1], "content", "") if all_msgs else "Tidak ada respons yang dihasilkan."
                    answer = _extract_text_content(raw_content)

                    image_bytes = None
                    if os.path.exists("temp_plot.png"):
                        with open("temp_plot.png", "rb") as imgf:
                            image_bytes = imgf.read()
                        st.image(image_bytes, use_container_width=True)
                        os.remove("temp_plot.png")

                    st.markdown(answer)
                    st.session_state["last_ai_status"] = "ok"
                    
                    msg_entry = {"role": "assistant", "content": answer}
                    if image_bytes:
                        msg_entry["image_bytes"] = image_bytes
                    st.session_state.messages.append(msg_entry)

                except Exception as e:
                    st.session_state["last_ai_status"] = "error"
                    diag = parse_ai_error_details(e, selected_model)
                    render_error_diagnostics(diag)
                    col_act1, _ = st.columns([1.5, 2.5])
                    with col_act1:
                        if st.button("Pilih Model Lain di Pengaturan", key="btn_switch_model_err"):
                            st.session_state["nav_menu"] = "Pengaturan"
                            st.rerun()

# ── HALAMAN 4: PENGATURAN SISTEM & AI ─────────────────────────────
elif halaman == "Pengaturan":
    st.markdown('<h1>Pengaturan Sistem & Kredensial AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-desc">Kelola konfigurasi API Key Google Gemini, preferensi model kecerdasan buatan, dan parameter integrasi sistem toko Anda.</div>', unsafe_allow_html=True)

    if "google_api_key" not in st.session_state:
        st.session_state["google_api_key"] = st.secrets.get("GOOGLE_API_KEY", "")
    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = "gemini-3.5-flash-lite"

    col_form, col_info = st.columns([1.6, 1], gap="large")

    with col_form:
        st.markdown("### Kredensial Google Gemini")
        st.caption("Kunci API ini menjadi mesin pemikir utama untuk membedah data transaksi internal UMKM dan melakukan riset kompetitor di web.")
        
        current_api_key = st.session_state.get("google_api_key", "")
        input_api_key = st.text_input(
            "Google API Key",
            type="password",
            value=current_api_key,
            placeholder="Tempel Google API Key Anda di sini...",
            help="Dapatkan kunci API gratis di Google AI Studio (aistudio.google.com)."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Preferensi Model AI")
        st.caption("Pilih model LLM Google Gemini yang digunakan oleh ReAct Agent (disesuaikan dengan alokasi kuota gratis Google AI Studio):")

        model_list = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.5-flash"
        ]
        model_desc = {
            "gemini-3.5-flash-lite": "**Gemini 3.5 Flash Lite (Direkomendasikan)** — Kuota gratis tertinggi: 500 permintaan/hari (RPD) & 15 RPM. Kecepatan respon kilat, sangat hemat kuota, dan optimal untuk analisis data toko serta eksekusi Python REPL.",
            "gemini-3.1-flash-lite": "**Gemini 3.1 Flash Lite** — Kuota gratis 500 permintaan/hari (RPD) & 15 RPM. Opsi alternatif stabil berkecepatan tinggi dengan alokasi harian besar.",
            "gemini-3.8-flash": "**Gemini 3.8 Flash** — Kuota gratis 20 permintaan/hari (RPD) & 5 RPM. Model generasi terkini dengan kemampuan penalaran logika mendalam untuk evaluasi strategi e-commerce lanjutan.",
            "gemini-3.5-flash": "**Gemini 3.5 Flash** — Kuota gratis 20 permintaan/hari (RPD) & 5 RPM. Keseimbangan solid antara kapasitas penalaran analitis dan kecepatan inferensi."
        }

        current_selected = st.session_state.get("gemini_model", "gemini-3.5-flash-lite")
        sel_idx = model_list.index(current_selected) if current_selected in model_list else 0
        input_model = st.selectbox(
            "Pilih Varian Model",
            model_list,
            index=sel_idx
        )
        st.info(model_desc.get(input_model, ""))

        # Uji koneksi interaktif
        st.markdown("<br>", unsafe_allow_html=True)
        col_save, col_test, col_del, col_chat = st.columns([1.2, 1.2, 1, 1.2])
        
        with col_save:
            if st.button("Simpan Pengaturan", type="primary", use_container_width=True):
                st.session_state["google_api_key"] = input_api_key.strip()
                st.session_state["gemini_model"] = input_model
                st.session_state["last_ai_status"] = "ok"
                _init_agent.clear()
                st.success("Pengaturan berhasil disimpan dan langsung aktif.")
                st.rerun()

        with col_test:
            if st.button("Uji Koneksi", use_container_width=True):
                test_key = input_api_key.strip()
                if not test_key:
                    st.warning("Masukkan Google API Key terlebih dahulu untuk melakukan pengujian.")
                else:
                    with st.spinner(f"Menguji koneksi ke {input_model}..."):
                        try:
                            from langchain_google_genai import ChatGoogleGenerativeAI
                            test_llm = ChatGoogleGenerativeAI(
                                model=input_model,
                                google_api_key=test_key,
                                temperature=0.1
                            )
                            test_res = test_llm.invoke("Halo, verifikasi koneksi sistem e-commerce.")
                            st.session_state["last_ai_status"] = "ok"
                            st.success(f"Koneksi Berhasil! Model '{input_model}' aktif dan siap digunakan.")
                        except Exception as test_e:
                            st.session_state["last_ai_status"] = "error"
                            diag = parse_ai_error_details(test_e, input_model)
                            render_error_diagnostics(diag)

        with col_del:
            if st.button("Hapus Kunci", use_container_width=True):
                st.session_state["google_api_key"] = ""
                st.session_state["last_ai_status"] = "ok"
                _init_agent.clear()
                st.warning("Kunci API telah dihapus.")
                st.rerun()

        with col_chat:
            if st.button("Ke Chatbot AI", use_container_width=True):
                st.session_state["nav_menu"] = "Chatbot AI Agent"
                st.rerun()

    with col_info:
        st.markdown("### Status Integrasi")
        has_key_saved = bool(st.session_state.get("google_api_key", "").strip())
        active_m = st.session_state.get('gemini_model', 'gemini-3.5-flash-lite')
        
        # Kuota info
        quota_info = "500 Permintaan / Hari (15 RPM)" if "lite" in active_m else "20 Permintaan / Hari (5 RPM)"
        
        if has_key_saved:
            st.markdown(
                f"""
                <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #15803d;">Sistem AI Terhubung</div>
                    <div style="font-size: 0.78rem; color: #166534; margin-top: 4px;">Kunci API terdeteksi dan terhubung ke engine Google Gemini.</div>
                    <div style="font-size: 0.74rem; color: #4b5563; margin-top: 8px;">Model Aktif: <code style="background: #e2e8f0; padding: 1px 5px; border-radius: 4px;">{active_m}</code></div>
                    <div style="font-size: 0.74rem; color: #15803d; margin-top: 4px; font-weight: 600;">Alokasi Kuota: {quota_info}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #b45309;">Kunci Belum Terpasang</div>
                    <div style="font-size: 0.78rem; color: #92400e; margin-top: 4px;">Masukkan Google API Key Anda untuk mengaktifkan seluruh fitur Chatbot AI Agent.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Ringkasan Kuota Gratis Model (Google AI)")
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; font-size: 0.76rem; color: #334155; line-height: 1.55; margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #e2e8f0; font-weight: 600;">
                        <th style="text-align: left; padding: 4px 0;">Model</th>
                        <th style="text-align: center; padding: 4px 0;">RPM</th>
                        <th style="text-align: right; padding: 4px 0;">RPD (Harian)</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9; background: #f0fdf4;">
                        <td style="padding: 5px 0; font-weight: 600; color: #166534;">3.5 Flash Lite</td>
                        <td style="text-align: center; color: #166534;">15</td>
                        <td style="text-align: right; font-weight: 600; color: #166534;">500 RPD</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 5px 0;">3.1 Flash Lite</td>
                        <td style="text-align: center;">15</td>
                        <td style="text-align: right; font-weight: 600;">500 RPD</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 5px 0;">3.8 Flash</td>
                        <td style="text-align: center;">5</td>
                        <td style="text-align: right;">20 RPD</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;">3.5 Flash</td>
                        <td style="text-align: center;">5</td>
                        <td style="text-align: right;">20 RPD</td>
                    </tr>
                </table>
                <div style="font-size: 0.70rem; color: #64748b; margin-top: 8px;">
                    *RPD = Requests Per Day (Permintaan per Hari). RPM = Requests Per Minute.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Panduan Mendapatkan Kunci")
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; font-size: 0.82rem; line-height: 1.6; color: #475569;">
                1. Buka situs <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #0284c7; font-weight: 600;">Google AI Studio</a>.<br>
                2. Masuk menggunakan akun Google Anda.<br>
                3. Klik tombol <b>"Create API Key"</b>.<br>
                4. Salin kode API Key yang muncul.<br>
                5. Tempelkan ke kolom form sebelah kiri dan klik <b>Simpan Pengaturan</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Keamanan dan Privasi Sesi")
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; font-size: 0.78rem; color: #64748b; line-height: 1.5;">
                Kunci API yang Anda masukkan hanya disimpan pada memori sesi peramban Anda (Streamlit Session State) dan tidak pernah dibagikan atau dipublikasikan.
            </div>
            """,
            unsafe_allow_html=True
        )