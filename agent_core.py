# ============================================================
# agent_core.py — Inti Logika AI Agent E-commerce (Bebas Streamlit)
# ============================================================
# Modul ini berisi:
# 1. Custom Tools (Web Scraper Pasar Bebas-Tavily & Python REPL)
# 2. System Prompt & Persona AI Analyst E-commerce
# 3. Fungsi Pembangun Agent Graph (build_agent) menggunakan LangGraph & Gemini
# 4. Utilitas ekstraksi respon & diagnosa kesalahan (error diagnostics)
#
# PENTING: File ini tidak bergantung pada Streamlit (zero Streamlit dependencies),
# sehingga dapat diuji secara headless, di-import oleh harness evaluasi (eval_agentic_ai.py),
# maupun digunakan dalam UI Streamlit (streamlit_app.py).
#
# CATATAN PERBAIKAN (lihat komentar "# [FIX]" di bawah untuk detail):
# 1. forecast_tool: docstring tidak lagi menyebut nama algoritma spesifik (XGBoost),
#    karena model terbaik yang di-load dari best_model_weekly.pkl bisa berubah
#    (saat ini Ridge Regression Tuned) tanpa perlu update dokumentasi ini lagi.
# 2. forecast_tool: is_ramadan dan is_harbolnas tidak lagi di-hardcode ke 0 untuk
#    minggu-minggu prediksi masa depan — sekarang dihitung dinamis per tanggal,
#    sama seperti perhitungan pada tahap training (memakai hijri-converter).
# ============================================================

import os
import re
import ast
import sys
import shutil
import tempfile
import subprocess
import datetime
import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any, Union

import matplotlib
matplotlib.use("Agg")

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI MODEL & PROMPT SISTEM
# ════════════════════════════════════════════════════════════

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SUPPORTED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash"
]

SYSTEM_PROMPT = """Kamu adalah AI Analyst E-commerce profesional untuk toko online UMKM di Indonesia.

=== ATURAN PENGGUNAAN INSTRUMEN ===
1. Jika pengguna bertanya mengenai performa toko historis, omzet, transaksi, tren penjualan, atau grafik -> Gunakan instrumen `Python_REPL` untuk menganalisis data pada file internal 'user_data.csv'.
2. Jika pengguna meminta prediksi, peramalan (forecasting), atau proyeksi pendapatan/omzet penjualan masa depan -> Gunakan instrumen `forecast_tool`.
3. Jika pengguna bertanya mengenai harga pasaran luar, kompetitor, tren e-commerce di Shopee/Tokopedia -> Gunakan instrumen `custom_marketplace_scraper`.
4. SAAT MERANGKUM HASIL RISET PASAR: Wajib cantumkan tautan URL sumber (Tautan Rujukan) secara jelas untuk setiap klaim harga atau produk yang kamu sebutkan.
5. ATURAN FAIL-CLOSED (WAJIB, TANPA PENGECUALIAN): Jika `custom_marketplace_scraper` mengembalikan hasil kosong atau pesan bahwa tidak ada produk/URL yang ditemukan, kamu DILARANG KERAS memberikan kisaran harga, estimasi angka, atau perkiraan pasar dari pengetahuan umum/internal sebagai gantinya, dalam bentuk apapun. Sebagai gantinya, kamu WAJIB menyatakan secara eksplisit dan jujur kepada pengguna bahwa sumber data pasar untuk kueri tersebut sedang tidak tersedia, lalu sarankan pengguna mencoba kata kunci pencarian yang lebih spesifik. Jangan mengisi kekosongan data dengan tebakan angka apapun, walau terdengar masuk akal.

=== PANDUAN PENTING KOMUNIKASI PENGGUNA (BAHASA BISNIS) ===
- JANGAN PERNAH menyebut nama berkas teknis internal (seperti 'user_data.csv', 'ecommerce_cleaned.csv', 'temp_plot.png') atau modul teknis (seperti 'Python_REPL', 'scraper', kode error 'HTTP 202').
- Gunakan selalu peristilahan bisnis yang ramah dan alami:
  * Alih-alih 'user_data.csv' -> sebut sebagai 'Data Penjualan Toko Anda' atau 'Laporan Transaksi Toko'.
  * Alih-alih 'Python_REPL' -> sebut sebagai 'Sistem Analitik Data Toko'.
  * Alih-alih 'custom_marketplace_scraper' -> sebut sebagai 'Sistem Riset Pasar Marketplace'.
- Berikan respon yang informatif, analitis, mudah dipahami pemilik toko, dan langsung dapat ditindaklanjuti untuk meningkatkan omzet."""


# ════════════════════════════════════════════════════════════
# 1B. SANDBOX KEAMANAN UNTUK PYTHON_REPL
# ════════════════════════════════════════════════════════════
# Mitigasi risiko eksekusi kode arbitrer oleh agen AI. Terdiri dari tiga lapis:
#   (a) Whitelist statis modul & builtin yang boleh dipanggil (AST parsing,
#       dijalankan SEBELUM kode dieksekusi sama sekali),
#   (b) Eksekusi pada proses Python terpisah (subprocess) dengan CPU time
#       limit, memory limit, dan wall-clock timeout,
#   (c) Dataset hanya-baca (read-only) yang disalin ke direktori kerja
#       sementara (sandbox dir), sehingga proses child tidak dapat menulis,
#       menghapus, atau mengubah data toko asli maupun berkas sistem lain.
# Catatan: pendekatan ini adalah sandboxing tingkat proses (process-level),
# bukan container/VM terisolasi penuh (mis. Docker/gVisor); pembatasan
# jaringan hanya dicapai secara tidak langsung melalui whitelist modul
# (requests/socket/urllib tidak termasuk modul yang diizinkan).

SANDBOX_ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "math", "statistics",
    "datetime", "json", "re", "collections", "itertools",
}

SANDBOX_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests",
    "urllib", "http", "ftplib", "pathlib", "importlib",
    "open", "eval", "exec", "compile", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "breakpoint", "memoryview",
}


class SandboxSecurityError(Exception):
    """Dilempar ketika kode yang diajukan agen melanggar kebijakan whitelist sandbox."""
    pass


def _sandbox_validate_code(code: str) -> None:
    """
    Analisis statis (AST) terhadap kode sebelum dieksekusi.
    Menolak: import modul di luar whitelist, pemanggilan builtin berbahaya
    (open/eval/exec/__import__/dst), serta akses atribut dunder yang lazim
    dipakai untuk sandbox-escape (mis. ().__class__.__bases__).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxSecurityError(f"Kode mengandung galat sintaksis: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                mod_roots = [n.name.split(".")[0] for n in node.names]
            else:
                mod_roots = [node.module.split(".")[0]] if node.module else []
            for m in mod_roots:
                if m not in SANDBOX_ALLOWED_MODULES:
                    raise SandboxSecurityError(
                        f"Import modul '{m}' tidak diizinkan oleh kebijakan sandbox. "
                        f"Modul yang diizinkan: {sorted(SANDBOX_ALLOWED_MODULES)}"
                    )

        if isinstance(node, ast.Name) and node.id in SANDBOX_FORBIDDEN_NAMES:
            raise SandboxSecurityError(
                f"Pemanggilan '{node.id}' tidak diizinkan oleh kebijakan sandbox."
            )

        if isinstance(node, ast.Attribute):
            if node.attr in SANDBOX_FORBIDDEN_NAMES:
                raise SandboxSecurityError(
                    f"Akses atribut '{node.attr}' tidak diizinkan oleh kebijakan sandbox."
                )
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SandboxSecurityError(
                    "Akses atribut dunder tidak diizinkan (potensi sandbox-escape)."
                )


def _sandbox_preexec_limits():
    """
    Diterapkan pada proses child (hanya berlaku pada sistem POSIX/Linux/macOS).
    Membatasi CPU time, memori, dan jumlah proses agar satu eksekusi kode
    tidak dapat membebani atau membekukan server aplikasi.
    """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))                     # 10 detik CPU time
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))  # 512 MB memori
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))                   # cegah fork-bomb
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))                  # batasi file descriptor
    except Exception:
        # Windows tidak memiliki modul 'resource' — timeout wall-clock pada
        # subprocess.run tetap berlaku sebagai lapisan mitigasi utama.
        pass


def _sandbox_execute(code: str, workdir: str, timeout_sec: int = 10) -> str:
    """
    Menjalankan kode pada proses Python child yang terisolasi:
    - cwd diarahkan ke direktori sandbox sementara (bukan direktori aplikasi),
    - flag -I (isolated mode): mengabaikan PYTHONPATH/PYTHONSTARTUP dan user
      site-packages, mengurangi permukaan serangan melalui environment,
    - wall-clock timeout via subprocess.run(timeout=...),
    - CPU/memory/proc limit via preexec_fn (POSIX) sebagai lapisan kedua.
    """
    script_path = os.path.join(workdir, "_sandbox_script.py")
    wrapped_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapped_code)

    preexec = _sandbox_preexec_limits if os.name == "posix" else None

    try:
        proc = subprocess.run(
            [sys.executable, "-I", script_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return (
            f"EKSEKUSI DIHENTIKAN OLEH SANDBOX: kode melebihi batas waktu "
            f"{timeout_sec} detik dan telah dihentikan paksa untuk melindungi sistem."
        )
    except Exception as e:
        return f"Sandbox gagal menjalankan proses eksekusi: {e}"

    if proc.returncode != 0:
        err_tail = proc.stderr.strip()[-1500:]
        return f"Kode mengalami galat saat dieksekusi pada sandbox:\n{err_tail}"

    output = proc.stdout.strip()
    return output if output else (
        "(Kode berhasil dieksekusi tanpa output tercetak. "
        "Gunakan print(...) pada kode agar hasilnya dapat ditampilkan.)"
    )


def run_sandboxed_python(code: str, dataset_filename: str = "user_data.csv", timeout_sec: int = 10) -> str:
    """
    Titik masuk utama sandbox: validasi kode -> siapkan direktori sandbox ->
    salin dataset sebagai read-only -> eksekusi terisolasi -> bersihkan.
    """
    try:
        _sandbox_validate_code(code)
    except SandboxSecurityError as e:
        return f"KODE DITOLAK OLEH KEBIJAKAN SANDBOX KEAMANAN: {e}"

    workdir = tempfile.mkdtemp(prefix="agent_sandbox_")
    try:
        src_data = dataset_filename if os.path.exists(dataset_filename) else "ecommerce_cleaned.csv"
        if os.path.exists(src_data):
            dst_data = os.path.join(workdir, "user_data.csv")
            shutil.copy2(src_data, dst_data)
            try:
                os.chmod(dst_data, 0o444)  # read-only: mencegah kode menimpa data asli
            except Exception:
                pass

        output = _sandbox_execute(code, workdir, timeout_sec=timeout_sec)

        # Bawa keluar hasil visualisasi (jika ada) agar tetap bisa ditampilkan di UI,
        # tanpa mengekspos direktori sandbox itu sendiri ke luar.
        plot_src = os.path.join(workdir, "temp_plot.png")
        if os.path.exists(plot_src):
            shutil.copy2(plot_src, "temp_plot.png")

        return output
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════
# 2. CUSTOM TOOLS
# ════════════════════════════════════════════════════════════

@tool
def custom_marketplace_scraper(query: str) -> str:
    """
    INSTRUMEN RISET PASAR DARING (MARKETPLACE): 
    Gunakan alat ini untuk mencari harga pasar, perbandingan harga kompetitor, tren e-commerce, dan produk serupa di marketplace (Shopee, Tokopedia).
    Input berupa string pencarian spesifik (contoh: 'harga celengan shopee tokopedia').
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    }
    extracted_data = []

    # 1. Jalur Utama: Mesin Pencari E-Commerce Yahoo (Stabil, ramah query marketplace ID)
    try:
        y_url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
        resp = requests.get(y_url, headers=headers, timeout=8)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for div in soup.find_all('div', class_='algo')[:5]:
                a = div.find('a')
                h3 = div.find('h3')
                comp = div.find('div', class_='compText') or div.find('p')
                if a and h3:
                    title = h3.text.strip()
                    raw_url = a.get('href', '')
                    clean_url = raw_url
                    m = re.search(r'/RU=(.*?)/RK=', raw_url)
                    if m:
                        clean_url = urllib.parse.unquote(m.group(1))
                    snippet = comp.text.strip() if comp else ''
                    if title and clean_url:
                        extracted_data.append(f"- **{title}**\n  *Ringkasan & Kisaran Harga*: {snippet}\n  *Tautan Rujukan*: {clean_url}\n")
    except Exception:
        pass

    # 2. Jalur Cadangan: DuckDuckGo HTML
    if not extracted_data:
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            resp_ddg = requests.get(ddg_url, headers=headers, timeout=6)
            if resp_ddg.status_code == 200:
                soup = BeautifulSoup(resp_ddg.text, 'html.parser')
                for res in soup.find_all('div', class_='result__body')[:4]:
                    t_tag = res.find('a', class_='result__url')
                    s_tag = res.find('a', class_='result__snippet')
                    if t_tag and s_tag:
                        raw_link = t_tag.get('href', '')
                        clean_link = raw_link
                        if "uddg=" in raw_link:
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_link).query)
                            clean_link = parsed.get('uddg', [raw_link])[0]
                        extracted_data.append(f"- **{t_tag.text.strip()}**\n  *Ringkasan*: {s_tag.text.strip()}\n  *Tautan Rujukan*: {clean_link}\n")
        except Exception:
            pass

    if not extracted_data:
        return (
            "[SUMBER_TIDAK_TERSEDIA] Hasil riset pasar daring saat ini TIDAK menemukan produk maupun URL sumber apapun untuk kueri tersebut. "
            "INSTRUKSI WAJIB: Jangan memberikan estimasi harga, kisaran harga, atau angka perkiraan apapun dari pengetahuan umum sebagai pengganti. "
            "Sampaikan kepada pengguna bahwa sumber data pasar untuk kueri ini sedang tidak tersedia, dan sarankan kata kunci pencarian yang lebih spesifik."
        )

    return "HASIL RISET PASAR DARING (MARKETPLACE):\n\n" + "\n".join(extracted_data)


# Alias web_scraper untuk kompatibilitas evaluasi
web_scraper = custom_marketplace_scraper


# [FIX #2] Helper kalender dipindahkan ke module-level agar bisa dipakai forecast_tool
# untuk menghitung is_ramadan/is_harbolnas secara dinamis per tanggal prediksi,
# alih-alih di-hardcode ke 0 seperti sebelumnya. Logika ini identik dengan yang
# dipakai pada tahap training (Tahap 2 feature engineering).
def _get_ramadan_dates(year: int):
    from hijri_converter import Hijri, Gregorian
    hijri_year = Gregorian(year, 3, 1).to_hijri().year
    ram_start = Hijri(hijri_year, 9, 1).to_gregorian()
    eid_day = Hijri(hijri_year, 10, 1).to_gregorian()
    return (
        datetime.date(ram_start.year, ram_start.month, ram_start.day),
        datetime.date(eid_day.year, eid_day.month, eid_day.day),
    )


def _compute_is_ramadan(week_date: datetime.date) -> int:
    r_start, eid = _get_ramadan_dates(week_date.year)
    week_end = week_date + datetime.timedelta(days=6)
    if (r_start <= week_date <= eid) or (r_start <= week_end <= eid) or (week_date <= r_start <= week_end):
        return 1
    return 0


def _compute_is_harbolnas(week_date: datetime.date) -> int:
    dates_in_week = [week_date + datetime.timedelta(days=i) for i in range(7)]
    for d in dates_in_week:
        if d.month == 12 and 10 <= d.day <= 18:
            return 1
    return 0


@tool
def forecast_tool(n_weeks: int = 4) -> str:
    """
    INSTRUMEN PREDIKSI PENDAPATAN TOKO (SALES FORECASTING):
    Gunakan alat ini ketika pengguna meminta prediksi, peramalan (forecasting), atau proyeksi omzet atau pendapatan penjualan toko untuk minggu-minggu mendatang (1 hingga 12 minggu ke depan).
    Alat ini menjalankan model machine learning terlatih untuk memprediksi omzet toko secara mingguan.
    Input: jumlah minggu yang ingin diprediksi (integer, contoh: 1, 2, 4).
    """
    # [FIX #1] Docstring di atas sengaja tidak menyebut nama algoritma spesifik
    # (sebelumnya "XGBoost terlatih"), karena model yang di-load dari
    # best_model_weekly.pkl mengikuti model terbaik hasil evaluasi Tahap 3
    # (saat ini Ridge Regression Tuned) dan bisa berubah di masa depan tanpa
    # perlu mengubah dokumentasi tool ini lagi.
    import joblib, json
    import numpy as np
    import pandas as pd
    from datetime import timedelta

    try:
        n_weeks = int(n_weeks) if n_weeks else 4
    except Exception:
        n_weeks = 4
    n_weeks = max(1, min(12, n_weeks))

    # Cek ketersediaan file model ML
    if not (os.path.exists("best_model_weekly.pkl") and os.path.exists("scaler_weekly.pkl") and os.path.exists("feature_columns_weekly.json")):
        return (
            f"Model machine learning forecasting belum tersedia di direktori sistem. "
            f"Proyeksi estimasi tren rata-rata mingguan: Rp 15.000.000 per minggu untuk {n_weeks} minggu ke depan."
        )

    try:
        model = joblib.load("best_model_weekly.pkl")
        scaler = joblib.load("scaler_weekly.pkl")
        with open("feature_columns_weekly.json", "r") as f:
            meta = json.load(f)

        data_src = "user_data.csv" if os.path.exists("user_data.csv") else "ecommerce_cleaned.csv"
        df_raw = pd.read_csv(data_src)

        # Konversi tanggal
        date_col = "order_date" if "order_date" in df_raw.columns else "Waktu Pesanan Dibuat"
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[date_col])

        rev_col = "Total Pembayaran" if "Total Pembayaran" in df_raw.columns else ("revenue_per_item" if "revenue_per_item" in df_raw.columns else df_raw.columns[-1])
        weekly_hist = df_raw.groupby(pd.Grouper(key=date_col, freq="W"))[rev_col].sum().reset_index()
        weekly_hist = weekly_hist.rename(columns={date_col: "order_week", rev_col: "total_revenue"}).sort_values("order_week").reset_index(drop=True)

        if len(weekly_hist) == 0:
            return "Data histori penjualan belum cukup untuk menjalankan peramalan mingguan."

        # Jalankan rolling forecast
        last_dt = weekly_hist["order_week"].max()
        scale_cols = meta.get("scale_cols", [])
        expected_features = meta.get("features", [])

        df_roll = weekly_hist.copy()
        preds = []
        training_start = pd.Timestamp("2023-12-31")

        for i in range(n_weeks):
            future_dt = last_dt + timedelta(weeks=i + 1)
            placeholder = pd.DataFrame([{"order_week": future_dt, "total_revenue": 0, "is_forecast_placeholder": 1}])
            temp = pd.concat([df_roll, placeholder], ignore_index=True)
            temp["order_week"] = pd.to_datetime(temp["order_week"])

            # Fitur kalender & lag
            temp["year"] = temp["order_week"].dt.year
            temp["month"] = temp["order_week"].dt.month
            temp["week_of_year"] = temp["order_week"].dt.isocalendar().week.astype(int)
            temp["quarter"] = temp["order_week"].dt.quarter
            temp["sin_month"] = np.sin(2 * np.pi * temp["month"] / 12)
            temp["cos_month"] = np.cos(2 * np.pi * temp["month"] / 12)
            temp["sin_year"] = np.sin(2 * np.pi * temp["week_of_year"] / 52)
            temp["cos_year"] = np.cos(2 * np.pi * temp["week_of_year"] / 52)
            temp["week_index"] = (temp["order_week"] - training_start).dt.days // 7

            rev_lag = temp["total_revenue"].replace(0, 5000000.0)
            temp["lag_1"] = rev_lag.shift(1).fillna(0)
            temp["lag_4"] = rev_lag.shift(4).fillna(0)
            temp["lag_12"] = rev_lag.shift(12).fillna(0)
            temp["rolling_4w_mean"] = rev_lag.shift(1).rolling(4, min_periods=1).mean().fillna(0)
            temp["rolling_12w_mean"] = rev_lag.shift(1).rolling(12, min_periods=1).mean().fillna(0)

            # [FIX #2] is_ramadan dan is_harbolnas dihitung dinamis per tanggal,
            # BUKAN di-hardcode ke 0 seperti versi sebelumnya. Ini penting karena
            # prediksi yang jatuh pada periode Ramadan/Desember sebelumnya selalu
            # dianggap non-Ramadan/non-Harbolnas oleh model, sehingga bias hasil
            # prediksi pada periode musiman tersebut.
            temp["is_ramadan"] = temp["order_week"].apply(lambda x: _compute_is_ramadan(x.date()))
            temp["is_harbolnas"] = temp["order_week"].apply(lambda x: _compute_is_harbolnas(x.date()))
            temp["is_holiday_gap"] = 0

            row = temp.iloc[[-1]].copy()
            for col in expected_features:
                if col not in row.columns:
                    row[col] = 0

            X = row[expected_features].copy()
            scale_present = [c for c in scale_cols if c in X.columns]
            if scale_present:
                X[scale_present] = scaler.transform(X[scale_present])

            y_idr = max(0.0, float(np.expm1(model.predict(X)[0])))
            preds.append({"order_week": future_dt, "predicted_revenue": y_idr})
            df_roll = pd.concat([df_roll, pd.DataFrame([{"order_week": future_dt, "total_revenue": y_idr}])], ignore_index=True)

        lines = [f"HASIL PREDIKSI OMZET PENJUALAN MINGGUAN ({n_weeks} MINGGU KE DEPAN):"]
        total_pred = 0
        for p in preds:
            dt_str = p["order_week"].strftime("%d %b %Y")
            lines.append(f"- Pekan berakhir {dt_str}: Rp {p['predicted_revenue']:,.0f}")
            total_pred += p["predicted_revenue"]
        lines.append(f"\nTotal Akumulasi Proyeksi Omzet: Rp {total_pred:,.0f}")
        lines.append(f"Rata-rata Mingguan: Rp {total_pred / n_weeks:,.0f}")
        return "\n".join(lines)

    except Exception as e:
        return f"Proses forecasting mengalami kendala: {str(e)}"


@tool("Python_REPL")
def python_repl(code: str) -> str:
    """
    INSTRUMEN ANALITIK TRANSAKSI INTERNAL TOKO (SANDBOXED):
    Kamu adalah Data Analyst. Jalankan kode Python murni dengan Pandas untuk menganalisis
    data transaksi toko pada file 'user_data.csv'.
    1. Muat data: df = pd.read_csv('user_data.csv')
    2. Hitung statistik dan metrik penjualan toko sesuai pertanyaan.
    3. JIKA USER MINTA GRAFIK: Buat dengan matplotlib, dan simpan dengan:
       `plt.savefig('temp_plot.png', dpi=120, bbox_inches='tight'); plt.close()`.
    4. PENTING: Jangan pernah menyebutkan nama berkas fisik 'user_data.csv' atau 'Python_REPL'
       dalam narasi akhir kepada pengguna.

    CATATAN KEAMANAN (ditegakkan secara teknis, bukan hanya instruksi):
    Kode dieksekusi pada proses terisolasi (sandbox) yang terpisah dari aplikasi utama.
    Hanya modul pandas, numpy, matplotlib, math, statistics, datetime, json, re, collections,
    dan itertools yang diizinkan. Builtin berisiko (open, eval, exec, __import__, os, sys,
    subprocess, socket, requests) diblokir sebelum kode dijalankan. Dataset disediakan dalam
    mode hanya-baca (read-only) pada salinan sementara sehingga data toko asli tidak dapat
    diubah atau dihapus. Eksekusi dibatasi maksimal 10 detik CPU time dan 512MB memori;
    kode yang melebihi batas akan dihentikan paksa secara otomatis.
    """
    return run_sandboxed_python(code, dataset_filename="user_data.csv", timeout_sec=10)


def get_agent_tools() -> list:
    """Mengembalikan daftar tool aktif yang digunakan oleh agent."""
    return [python_repl, forecast_tool, custom_marketplace_scraper]



# ════════════════════════════════════════════════════════════
# 3. UTILITAS FORMATING & DIAGNOSTIK ERROR
# ════════════════════════════════════════════════════════════

def demo_response(prompt: str = "") -> str:
    """Fallback response jika GOOGLE_API_KEY belum diset."""
    return (
        "**API Key Google Gemini belum diset.**\n\n"
        "Silakan buka halaman **Pengaturan** pada menu sebelah kiri (Sidebar) untuk memasukkan Google API Key Anda.\n\n"
        "Dapatkan gratis di [Google AI Studio](https://aistudio.google.com/app/apikey)."
    )

# Alias backward compatibility
_demo_response = demo_response


def extract_text_content(content: Any) -> str:
    """Mengekstrak teks murni dari response LLM (baik string maupun list of parts)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts) if texts else str(content)
    return str(content)

# Alias backward compatibility
_extract_text_content = extract_text_content


def parse_ai_error_details(err: Exception, model_name: str) -> dict:
    """
    Mendiagnosa akar penyebab kegagalan koneksi atau inferensi model Gemini AI
    dan merumuskan panduan solusi solutif bagi pengguna.
    """
    err_str = str(err)
    err_lower = err_str.lower()

    # 1. Deteksi 404 NOT_FOUND (Model tidak ada / deprecated)
    if "404" in err_str or "not_found" in err_lower or "not found" in err_lower or "no longer available" in err_lower:
        return {
            "title": "Model AI Tidak Ditemukan atau Tidak Didukung (HTTP 404)",
            "badge": "Model Tidak Tersedia",
            "badge_color": "#dc2626",
            "badge_bg": "#fee2e2",
            "cause": f"Model <strong>{model_name}</strong> saat ini tidak tersedia di akun Google AI Studio Anda atau telah digantikan versinya oleh Google.",
            "solution": "Buka menu <strong>Pengaturan</strong> dan pilih model yang aktif dengan kuota gratis tinggi: <strong>gemini-3.5-flash-lite</strong> (500 permintaan/hari) atau <strong>gemini-3.1-flash-lite</strong>.",
            "technical": err_str
        }

    # 2. Deteksi 429 Quota Exceeded / Rate Limit
    if "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
        return {
            "title": "Batas Kuota Permintaan Tercapai (HTTP 429)",
            "badge": "Limit Kuota",
            "badge_color": "#ea580c",
            "badge_bg": "#ffedd5",
            "cause": f"Akun Google API Anda telah mencapai limit laju panggilan per menit (RPM) atau kuota harian (RPD) untuk model <strong>{model_name}</strong>.",
            "solution": "Tunggu 1–2 menit sebelum mengirim perintah baru. Jika kuota harian telah habis, beralihlah ke model <strong>gemini-3.5-flash-lite</strong> (alokasi 500 permintaan/hari) di menu Pengaturan.",
            "technical": err_str
        }

    # 3. Deteksi API Key Invalid / Unauthorized
    if "400" in err_str or "401" in err_str or "403" in err_str or "api_key_invalid" in err_lower or "api key not valid" in err_lower or "permission_denied" in err_lower:
        return {
            "title": "Google API Key Tidak Valid atau Ditolak (HTTP 401/403)",
            "badge": "Autentikasi Gagal",
            "badge_color": "#dc2626",
            "badge_bg": "#fee2e2",
            "cause": "Kunci Google API yang dimasukkan tidak valid, belum diaktifkan, atau tidak memiliki izin akses ke Google AI Studio.",
            "solution": "Periksa dan masukkan kembali kunci API yang valid pada halaman <strong>Pengaturan</strong>. Pastikan tidak ada spasi atau karakter yang terpotong.",
            "technical": err_str
        }

    # 4. Deteksi Network / Timeout
    if "timeout" in err_lower or "connection" in err_lower or "connecterror" in err_lower or "econnrefused" in err_lower:
        return {
            "title": "Gangguan Koneksi ke Server Google (Network Timeout)",
            "badge": "Koneksi Terputus",
            "badge_color": "#475569",
            "badge_bg": "#f1f5f9",
            "cause": "Aplikasi gagal tersambung ke endpoint Google AI Studio (generativelanguage.googleapis.com).",
            "solution": "Periksa koneksi internet Anda atau pastikan jaringan tidak memblokir akses ke domain API Google.",
            "technical": err_str
        }

    # 5. Fallback Error Umum
    return {
        "title": "Kendala Pemrosesan Permintaan AI",
        "badge": "Kendala Sistem",
        "badge_color": "#b91c1c",
        "badge_bg": "#fef2f2",
        "cause": f"Terjadi kendala saat memproses kueri dengan model <strong>{model_name}</strong>.",
        "solution": "Coba reset percakapan atau pilih model lain melalui menu Pengaturan. Pastikan format pertanyaan jelas.",
        "technical": err_str
    }


# ════════════════════════════════════════════════════════════
# 4. FUNGSI PEMBANGUN AGENT (BUILD AGENT)
# ════════════════════════════════════════════════════════════

def build_agent(
    google_api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
    tools: Optional[list] = None
):
    """
    Inisialisasi LangGraph ReAct Agent dengan Google Gemini API.
    Bebas dari Streamlit runtime, sehingga dapat digunakan di UI, script headless, maupun testing.

    Parameters:
    - google_api_key: Kunci API Google Gemini. Jika None, akan mencari dari os.environ['GOOGLE_API_KEY'].
    - model: Nama model Gemini (default: gemini-3.5-flash-lite).
    - temperature: Tingkat kreativitas model (default: 0.1 untuk stabilitas analitis).
    - system_prompt: Prompt persona kustom (default: SYSTEM_PROMPT).
    - tools: Daftar tools yang ingin disediakan (default: [python_repl, forecast_tool, custom_marketplace_scraper]).

    Returns:
    - LangGraph CompiledGraph (ReAct agent)
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key or not str(api_key).strip():
        raise ValueError(
            "Google API Key tidak ditemukan. Berikan parameter google_api_key "
            "atau atur environment variable 'GOOGLE_API_KEY'."
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=str(api_key).strip(),
        temperature=temperature,
    )

    agent_tools = tools if tools is not None else get_agent_tools()
    prompt = system_prompt or SYSTEM_PROMPT

    agent_graph = create_react_agent(
        model=llm,
        tools=agent_tools,
        prompt=prompt,
    )
    return agent_graph


def run_agent_query(
    agent_graph,
    user_query: str,
    data_file_path: str = "user_data.csv"
) -> Dict[str, Any]:
    """
    Menjalankan kueri pengguna pada agent graph dan mengembalikan jawaban terstruktur.

    Returns:
    - Dict berisi:
      - 'answer': Teks jawaban akhir dari agent
      - 'image_bytes': Bytes grafik jika agent menghasilkan temp_plot.png (lalu file dihapus otomatis)
      - 'messages': Semua pesan hasil eksekusi graph
    """
    agent_input = (
        f"Data transaksi toko tersimpan pada berkas: '{data_file_path}'.\n"
        f"Pertanyaan Pemilik Toko: {user_query}\n\n"
        f"PANDUAN KOMUNIKASI: Jangan pernah menyebutkan nama berkas fisik teknis ('{data_file_path}' atau lainnya) di dalam teks jawaban. "
        f"Sebut selalu sebagai 'Data Penjualan Toko Anda' atau 'Laporan Transaksi Toko'."
    )

    result = agent_graph.invoke({"messages": [("human", agent_input)]})
    all_msgs = result.get("messages", [])
    raw_content = getattr(all_msgs[-1], "content", "") if all_msgs else "Tidak ada respons yang dihasilkan."
    answer = extract_text_content(raw_content)

    image_bytes = None
    if os.path.exists("temp_plot.png"):
        try:
            with open("temp_plot.png", "rb") as imgf:
                image_bytes = imgf.read()
            os.remove("temp_plot.png")
        except Exception:
            pass

    return {
        "answer": answer,
        "image_bytes": image_bytes,
        "messages": all_msgs
    }