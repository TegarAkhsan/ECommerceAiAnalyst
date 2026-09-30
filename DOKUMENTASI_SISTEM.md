# Dokumentasi Lengkap Sistem: E-commerce AI Analyst
> **Proyek Tugas Akhir** | Stack: Streamlit + LangGraph + Google Gemini + scikit-learn

---

## Daftar Isi
1. [Arsitektur Sistem](#1-arsitektur-sistem)
2. [Alur Upload File](#2-alur-upload-file--data-pipeline)
3. [Agentic AI — Cara Kerja ReAct](#3-agentic-ai--cara-kerja-react)
4. [LLM & Tools: Bagaimana AI Memilih Tool](#4-llm--tools-bagaimana-ai-memilih-tool)
5. [Detail Setiap Tool](#5-detail-setiap-tool)
6. [Sandbox Keamanan Python REPL](#6-sandbox-keamanan-python-repl)
7. [Machine Learning Forecasting](#7-machine-learning-forecasting)
8. [Evaluasi & Testing](#8-evaluasi--testing)
9. [Stack Teknologi](#9-stack-teknologi)
10. [File & Struktur Proyek](#10-file--struktur-proyek)

---

## 1. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (streamlit_app.py)              │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────┐  ┌────────┐  │
│  │ Upload   │  │ Prediksi        │  │ Chatbot   │  │Setting │  │
│  │ Data     │  │ Penjualan       │  │ AI Agent  │  │& API   │  │
│  └──────────┘  └─────────────────┘  └─────┬─────┘  └────────┘  │
└────────────────────────────────────────────┼────────────────────┘
                                             │ invoke()
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              AGENT CORE (agent_core.py)                         │
│                                                                 │
│   LangGraph create_react_agent(                                 │
│     model = ChatGoogleGenerativeAI (Gemini),                    │
│     tools = [python_repl, forecast_tool, marketplace_scraper],  │
│     prompt = SYSTEM_PROMPT                                      │
│   )                                                             │
│                                                                 │
│   ┌────────────────────────────────────────────┐               │
│   │         ReAct Loop (Reason → Act → Observe) │               │
│   └───────────┬────────────────┬───────────────┘               │
│               │                │                                │
│     ┌─────────▼──────┐ ┌───────▼──────────┐ ┌────────────────┐ │
│     │  Python_REPL   │ │  forecast_tool   │ │  marketplace   │ │
│     │  (Sandboxed)   │ │  (ML model .pkl) │ │  scraper (web) │ │
│     └────────────────┘ └──────────────────┘ └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      user_data.csv               Internet (Yahoo/DDG)
      best_model_weekly.pkl
```

---

## 2. Alur Upload File & Data Pipeline

### 2.1 Entry Point — Halaman "Upload Data"

Pengguna meng-upload file dari halaman **Upload Data** di sidebar. File yang didukung: `.xlsx`, `.xls`, `.csv`.

```python
# streamlit_app.py ~L1186
uploaded_files = st.file_uploader(
    "Pilih file laporan pesanan",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True   # ← bisa upload lebih dari 1 file sekaligus
)
```

### 2.2 Fungsi `read_file()` — Parsing Multi-Format

```python
def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    else:
        # Coba 3 encoding: utf-8 → latin-1 → cp1252
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        # Fallback: utf-8 dengan errors='replace'
        return pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
```

**Strategi Encoding:** Otomatis mencoba 3 encoding untuk menangani file ekspor Seller Centre Shopee/Tokopedia yang sering menggunakan `latin-1` atau `cp1252`.

### 2.3 Fungsi `clean_raw()` — Pembersihan Data

```python
def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
```

Langkah-langkah cleaning:

| Langkah | Aksi |
|---------|------|
| 1. Normalisasi kolom tanggal | `Waktu Pengiriman Diatur` → `Waktu Pesanan Dibuat` (jika perlu) |
| 2. Bersihkan Total Pembayaran | Hapus titik ribuan, ganti koma desimal, convert ke numerik |
| 3. Filter status | Hanya pesanan **Selesai** yang lolos (termasuk "Pesanan diterima...") |
| 4. Filter nilai | Hapus baris `Total Pembayaran <= 0` |
| 5. Parse tanggal | `Waktu Pesanan Dibuat` → `datetime`, tambahkan kolom `order_date` |
| 6. Reset index | `reset_index(drop=True)` |

### 2.4 Fungsi `make_weekly()` — Agregasi Mingguan

```python
def make_weekly(df_clean: pd.DataFrame) -> pd.DataFrame:
    # Deduplikasi: 1 order_id = 1 baris (multi-item dihitung sekali)
    df_order = df_clean.drop_duplicates(subset='order_id', keep='first')
    
    # Resampling ke frekuensi mingguan (W = week-ending pada Minggu)
    weekly = df_order.groupby(pd.Grouper(key='order_date', freq='W'))['Total Pembayaran'].sum()
    
    # Isi gap minggu kosong dengan 0 (mencegah lag features menjadi NaN)
    full_range = pd.date_range(weekly['order_week'].min(), weekly['order_week'].max(), freq='W')
    weekly = weekly.reindex(full_range).fillna(0)
    
    return weekly
```

### 2.5 Penyimpanan ke Session State & Disk

```python
# Disimpan ke Streamlit session state untuk UI
st.session_state['df_clean']   = df_clean      # DataFrame bersih
st.session_state['weekly_hist'] = make_weekly(df_clean)  # Agregasi mingguan

# Saat chatbot aktif: disimpan ke disk untuk agent
# (streamlit_app.py ~L1736)
st.session_state['df_clean'].to_csv('user_data.csv', index=False)
data_file_path = "user_data.csv"
```

> [!IMPORTANT]
> `user_data.csv` adalah **jembatan kritis** antara UI Streamlit dan AI Agent. Data yang diupload pengguna → diproses → disimpan ke `user_data.csv` → dibaca oleh `Python_REPL` dalam sandbox.

### 2.6 Diagram Lengkap Alur Data

```
File Upload (.xlsx/.csv)
        │
        ▼ read_file()
  pd.DataFrame (raw)
        │
        ▼ clean_raw()
  df_clean (pesanan selesai, nilai valid)
        │
        ├──────────────────────────────────────────┐
        ▼ make_weekly()                            ▼ (disimpan ke disk)
  weekly_hist (time series omzet per minggu)  user_data.csv
        │                                          │
        ▼                                          ▼
  Dashboard UI (chart, metrik, insight)    AI Agent (Python_REPL tool)
  Halaman "Prediksi Penjualan"             Halaman "Chatbot AI Agent"
```

---

## 3. Agentic AI — Cara Kerja ReAct

### 3.1 Pola ReAct (Reason → Act → Observe)

LangGraph mengimplementasikan pola **ReAct** di mana LLM beriterasi dalam siklus:

```
Pertanyaan Pengguna
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  REASON: LLM membaca query + system prompt          │
│  → Memutuskan: perlu tool apa? dengan input apa?    │
└─────────────────────────┬───────────────────────────┘
                          │ (tool_call)
                          ▼
┌─────────────────────────────────────────────────────┐
│  ACT: Tool dijalankan (Python_REPL / forecast /     │
│       scraper)                                      │
└─────────────────────────┬───────────────────────────┘
                          │ (tool result)
                          ▼
┌─────────────────────────────────────────────────────┐
│  OBSERVE: LLM membaca hasil tool                    │
│  → Apakah jawaban sudah cukup?                      │
│    - YA → Tulis jawaban final                       │
│    - TIDAK → Loop kembali ke REASON                 │
└─────────────────────────────────────────────────────┘
```

### 3.2 Inisialisasi Agent — `build_agent()`

```python
# agent_core.py ~L611
def build_agent(
    google_api_key: str = None,
    model: str = "gemini-3.5-flash-lite",
    temperature: float = 0.1,
    system_prompt: str = None,
    tools: list = None
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,   # 0.1 = deterministik untuk analisis
    )
    
    agent_graph = create_react_agent(
        model=llm,
        tools=[python_repl, forecast_tool, custom_marketplace_scraper],
        prompt=SYSTEM_PROMPT,
    )
    return agent_graph   # LangGraph CompiledGraph
```

Dipanggil dari Streamlit dengan caching:

```python
# streamlit_app.py ~L1128
@st.cache_resource
def _init_agent(_google_api_key: str, _model: str):
    return build_agent(google_api_key=_google_api_key, model=_model)
```

`@st.cache_resource` memastikan agent hanya diinisialisasi **sekali per sesi**, tidak setiap kali pengguna mengirim pesan.

### 3.3 Eksekusi Query — `run_agent_query()`

```python
# agent_core.py ~L656
def run_agent_query(agent_graph, user_query, data_file_path="user_data.csv"):
    # Konstruksi prompt yang disematkan ke pesan human
    agent_input = (
        f"Data transaksi toko tersimpan pada berkas: '{data_file_path}'.\n"
        f"Pertanyaan Pemilik Toko: {user_query}\n\n"
        f"PANDUAN KOMUNIKASI: Jangan pernah menyebutkan nama berkas fisik teknis..."
    )
    
    # Invoke graph dengan message state
    result = agent_graph.invoke({"messages": [("human", agent_input)]})
    
    # Ekstrak jawaban final dari pesan terakhir
    all_msgs = result.get("messages", [])
    raw_content = getattr(all_msgs[-1], "content", "")
    answer = extract_text_content(raw_content)
    
    # Cek apakah ada grafik yang dihasilkan
    image_bytes = None
    if os.path.exists("temp_plot.png"):
        with open("temp_plot.png", "rb") as imgf:
            image_bytes = imgf.read()
        os.remove("temp_plot.png")  # Bersihkan file sementara
    
    return {"answer": answer, "image_bytes": image_bytes, "messages": all_msgs}
```

---

## 4. LLM & Tools: Bagaimana AI Memilih Tool

### 4.1 System Prompt — Otak Routing

System prompt adalah instruksi yang selalu ada di setiap percakapan, mengarahkan LLM:

```
ATURAN PENGGUNAAN INSTRUMEN:
1. Omzet/tren historis/grafik → Python_REPL (user_data.csv)
2. Prediksi/proyeksi/forecasting → forecast_tool
3. Harga pasaran/kompetitor/Shopee/Tokopedia → custom_marketplace_scraper
4. Jika scraper kosong → WAJIB jujur, DILARANG mengarang harga
```

### 4.2 Cara LLM Memilih Tool

Gemini membaca **docstring** setiap tool untuk memahami kapan harus memanggilnya:

| Tool | Trigger Kata Kunci |
|------|-------------------|
| `Python_REPL` | omzet, revenue, transaksi, tren, grafik, produk terlaris, laris, AOV, pembayaran, pesanan, metode bayar, kota, hari, ongkir |
| `forecast_tool` | prediksi, peramalan, proyeksi, forecast, perkiraan, bulan depan, minggu depan |
| `custom_marketplace_scraper` | harga, pasaran, kompetitor, pesaing, Shopee, Tokopedia, marketplace, riset pasar |

### 4.3 Alur Internal Message di LangGraph

```
[Human Message] "Berapa omzet bulan ini?"
        │
        ▼
[AI Message] tool_calls=[{"name": "Python_REPL", "args": {"code": "..."}}]
        │
        ▼
[Tool Message] name="Python_REPL", content="Total omzet: Rp 45.230.000"
        │
        ▼
[AI Message] "Berdasarkan Data Penjualan Toko Anda, omzet bulan ini adalah Rp 45.230.000"
                                    ↑
                         (Nama teknis disembunyikan)
```

### 4.4 Persona Bisnis — Penyembunyian Nama Teknis

LLM diajarkan untuk **tidak pernah** menyebut nama teknis ke pengguna:

| Internal (teknis) | Ditampilkan ke pengguna |
|-------------------|------------------------|
| `Python_REPL` | "Sistem Analitik Data Toko" |
| `user_data.csv` | "Data Penjualan Toko Anda" |
| `custom_marketplace_scraper` | "Sistem Riset Pasar Marketplace" |
| `temp_plot.png` | (tidak disebutkan sama sekali) |

---

## 5. Detail Setiap Tool

### 5.1 Tool 1: `Python_REPL`

- **Deklarasi:** `@tool("Python_REPL")` — nama eksplisit untuk LLM
- **Fungsi:** Eksekusi kode Python untuk analisis data historis toko
- **Input:** `code: str` — kode Python murni
- **Output:** Stdout dari eksekusi + grafik (jika ada)

**Pola kode yang biasa dihasilkan LLM:**
```python
# Analisis omzet
import pandas as pd
df = pd.read_csv('user_data.csv')
total = df['Total Pembayaran'].sum()
print(f"Total omzet: Rp {total:,.0f}")

# Grafik (jika user minta visualisasi)
import matplotlib.pyplot as plt
df.groupby('Kategori')['Total Pembayaran'].sum().plot(kind='bar')
plt.savefig('temp_plot.png', dpi=120, bbox_inches='tight')
plt.close()
```

**Alur eksekusi lengkap:**
```
LLM generate kode Python
        │
        ▼
python_repl(code=...) dipanggil
        │
        ▼
run_sandboxed_python(code, "user_data.csv")
        │
        ├── _sandbox_validate_code(code)  ← AST check
        │         ▲ REJECT jika modul berbahaya
        │
        ▼
workdir = tempfile.mkdtemp()
shutil.copy2("user_data.csv", workdir)
os.chmod(dst_data, 0o444)  ← read-only
        │
        ▼
subprocess.run([python, "-I", script], cwd=workdir, timeout=10)
        │
        ▼
Ambil stdout → kembalikan ke LLM
Jika ada temp_plot.png → copy ke root proyek
        │
        ▼
shutil.rmtree(workdir)  ← bersihkan sandbox
```

---

### 5.2 Tool 2: `forecast_tool`

- **Deklarasi:** `@tool`
- **Input:** `n_weeks: int` (1–12)
- **Output:** String teks berisi prediksi per minggu + total + rata-rata

**Algoritma Rolling Forecast:**

```
Untuk setiap minggu ke-i dari n_weeks:
  1. Hitung future_dt = last_date + (i+1) week
  2. Tambahkan placeholder row ke df_roll
  3. Hitung fitur kalender:
     - year, month, week_of_year, quarter
     - sin_month, cos_month, sin_year, cos_year  (Fourier encoding musiman)
     - week_index (linear trend)
  4. Hitung fitur lag (dari history):
     - lag_1  (omzet minggu lalu)
     - lag_4  (omzet 4 minggu lalu)
     - lag_12 (omzet 12 minggu lalu)
     - rolling_4w_mean  (rata-rata 4 minggu)
     - rolling_12w_mean (rata-rata 12 minggu)
  5. Hitung is_ramadan (via hijri_converter, dinamis per tanggal)
  6. Hitung is_harbolnas (cek 10-18 Desember)
  7. Scale fitur numerik dengan scaler_weekly.pkl
  8. Prediksi: y = np.expm1(model.predict(X)[0])
     (model output adalah log1p(revenue), jadi di-inverse dengan expm1)
  9. Simpan hasil → masuk ke df_roll untuk iterasi berikutnya
```

**Kenapa Rolling Forecast?** Karena prediksi minggu ke-2 bergantung pada lag dari minggu ke-1 (yang juga hasil prediksi). Tanpa rolling, nilai lag akan selalu 0 untuk semua minggu masa depan.

**Fallback jika model belum ada:**
```python
if not os.path.exists("best_model_weekly.pkl"):
    return f"Model belum tersedia. Proyeksi tren rata-rata: Rp 15.000.000 per minggu"
```

---

### 5.3 Tool 3: `custom_marketplace_scraper`

- **Deklarasi:** `@tool`
- **Input:** `query: str` — string pencarian
- **Output:** Daftar produk + snippet harga + URL sumber

**Strategi 2-jalur:**

```python
# Jalur Utama: Yahoo Search
y_url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
resp = requests.get(y_url, headers=headers, timeout=8)
# Parse div.algo → cari tag <a> dan <h3>
# Ekstrak URL bersih dari format RU=.../RK=

# Jalur Cadangan: DuckDuckGo HTML (jika Yahoo gagal/kosong)
ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
# Parse div.result__body → cari a.result__url dan a.result__snippet
```

**Aturan Fail-Closed (SANGAT PENTING):**

```
Jika extracted_data kosong →
  return "[SUMBER_TIDAK_TERSEDIA] ... DILARANG memberikan estimasi harga..."
```

LLM diarahkan oleh system prompt:
> "Jika scraper mengembalikan hasil kosong → DILARANG mengarang kisaran harga dari pengetahuan internal."

---

## 6. Sandbox Keamanan Python REPL

Sistem menggunakan **3 lapis keamanan** untuk eksekusi kode yang di-generate LLM:

### Lapisan 1: AST Static Analysis

```python
SANDBOX_ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "math", "statistics",
    "datetime", "json", "re", "collections", "itertools"
}

SANDBOX_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests",
    "urllib", "open", "eval", "exec", "compile", "__import__", ...
}
```

Sebelum kode dijalankan, AST parser mengecek:
1. **Import:** Hanya modul dari `ALLOWED_MODULES` yang boleh di-import
2. **Nama berbahaya:** `eval`, `exec`, `open`, `__import__` diblokir
3. **Atribut dunder:** `.__class__.__bases__` dll diblokir (sandbox-escape prevention)

### Lapisan 2: Subprocess Isolation

```python
proc = subprocess.run(
    [sys.executable, "-I", script_path],  # -I = isolated mode
    cwd=workdir,               # ← direktori kerja sandbox, bukan root proyek
    capture_output=True,
    timeout=10,                # ← wall-clock timeout
    preexec_fn=preexec,        # ← resource limits (POSIX)
)
```

**Flag `-I` (Isolated Mode):** Mengabaikan `PYTHONPATH`, `PYTHONSTARTUP`, dan user site-packages.

**Resource Limits (POSIX/Linux):**
```python
resource.setrlimit(RLIMIT_CPU, (10, 10))            # 10 detik CPU
resource.setrlimit(RLIMIT_AS, (512MB, 512MB))       # 512 MB memori
resource.setrlimit(RLIMIT_NPROC, (16, 16))          # cegah fork-bomb
resource.setrlimit(RLIMIT_NOFILE, (64, 64))         # batas file descriptor
```

### Lapisan 3: Read-Only Dataset

```python
dst_data = os.path.join(workdir, "user_data.csv")
shutil.copy2(src_data, dst_data)
os.chmod(dst_data, 0o444)  # ← read-only: rw-r--r--
```

Data asli toko tidak bisa dimodifikasi oleh kode LLM.

---

## 7. Machine Learning Forecasting

### 7.1 File Model

| File | Isi |
|------|-----|
| `best_model_weekly.pkl` | Model terbaik (saat ini: Ridge Regression Tuned) |
| `scaler_weekly.pkl` | StandardScaler untuk fitur numerik |
| `feature_columns_weekly.json` | Metadata: list fitur & kolom yang di-scale |

### 7.2 Feature Engineering Lengkap

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `year` | Kalender | Tahun |
| `month` | Kalender | Bulan (1–12) |
| `week_of_year` | Kalender | Minggu ke- dalam tahun (1–52) |
| `quarter` | Kalender | Kuartal (1–4) |
| `sin_month` | Fourier | `sin(2π × month / 12)` — encoding siklus bulanan |
| `cos_month` | Fourier | `cos(2π × month / 12)` |
| `sin_year` | Fourier | `sin(2π × week_of_year / 52)` — encoding siklus tahunan |
| `cos_year` | Fourier | `cos(2π × week_of_year / 52)` |
| `week_index` | Trend | Indeks linear minggu sejak training_start (2023-12-31) |
| `lag_1` | Lag | Omzet minggu sebelumnya |
| `lag_4` | Lag | Omzet 4 minggu lalu |
| `lag_12` | Lag | Omzet 12 minggu lalu (pola kuartalan) |
| `rolling_4w_mean` | Rolling | Rata-rata omzet 4 minggu terakhir |
| `rolling_12w_mean` | Rolling | Rata-rata omzet 12 minggu terakhir |
| `is_ramadan` | Kalender Islam | 1 jika minggu masuk periode Ramadan (via hijri_converter) |
| `is_harbolnas` | Momen belanja | 1 jika minggu mencakup 10–18 Desember |
| `is_holiday_gap` | Flag | 1 jika omzet = 0 (libur/tutup, bukan placeholder) |

**Target variabel:** `log1p(total_revenue)` — model memprediksi log-revenue, lalu di-inverse dengan `np.expm1()`.

### 7.3 Analyst Insights Otomatis (UI Prediksi)

Setelah forecast dijalankan, sistem menghitung insights otomatis via `generate_forecast_analyst_insights()`:

- **Diagnosa tren:** slope H1 vs H2 horizon → "Upward", "Correction", atau "Steady"
- **Volatilitas:** Koefisien Variasi (CV%) → "Rendah < 12%", "Moderat ≤ 25%", "Tinggi > 25%"
- **Deteksi musiman:** Ramadan (bulan 3–4), Harbolnas (11–12), Libur tengah tahun (6–7)
- **Rekomendasi restocking:** tanggal reorder = peak_date − 10 hari
- **Rekomendasi marketing:** fokus iklan H-7 sebelum peak week
- **Cashflow buffer:** gunakan trough_revenue sebagai batas aman pengeluaran

---

## 8. Evaluasi & Testing

### 8.1 Harness Evaluasi (`eval_agentic_ai.py`)

File ini adalah **headless testing harness** yang menjalankan 28 query pengujian ke agent secara otomatis:

```bash
python eval_agentic_ai.py --api-key <GOOGLE_API_KEY>
```

### 8.2 28 Query Test — Dikelompokkan per Tool

#### Kelompok 1: Python_REPL — Analisis Data Lokal (Q01–Q10)

| ID | Query | Expected Tool |
|----|-------|---------------|
| Q01 | Berapa total revenue minggu lalu? | `python_repl` |
| Q02 | Tampilkan tren revenue 4 minggu terakhir | `python_repl` |
| Q03 | Produk apa yang paling laris bulan ini? | `python_repl` |
| Q04 | Berapa rata-rata nilai transaksi (AOV) di toko saya? | `python_repl` |
| Q05 | Berapa jumlah pesanan yang dibatalkan oleh pembeli dan apa alasan utamanya? | `python_repl` |
| Q06 | Di hari apa toko saya menerima volume pesanan paling banyak? | `python_repl` |
| Q07 | Metode pembayaran apa yang paling populer digunakan oleh pelanggan? | `python_repl` |
| Q08 | Sebutkan 3 kota atau kabupaten dengan kontribusi omzet terbesar | `python_repl` |
| Q09 | Berapa total akumulasi ongkos kirim yang dibayarkan oleh pembeli? | `python_repl` |
| Q10 | Buatkan visualisasi grafik batang penjualan berdasarkan kategori produk | `python_repl` |

#### Kelompok 2: forecast_tool — Prediksi Omzet (Q11–Q16)

| ID | Query | Expected Tool |
|----|-------|---------------|
| Q11 | Prediksi revenue minggu depan | `forecast_tool` |
| Q12 | Perkirakan omzet toko untuk 4 minggu ke depan | `forecast_tool` |
| Q13 | Berapa proyeksi penjualan toko pada bulan depan? | `forecast_tool` |
| Q14 | Apakah pendapatan toko minggu depan diproyeksikan mengalami kenaikan atau penurunan? | `forecast_tool` |
| Q15 | Ramalkan perkiraan omzet mingguan toko menghadapi periode hari raya mendatang | `forecast_tool` |
| Q16 | Buatkan estimasi forecast omzet penjualan untuk 2 minggu mendatang | `forecast_tool` |

#### Kelompok 3: Web Scraper — Riset Pasar (Q17–Q22)

| ID | Query | Expected Tool |
|----|-------|---------------|
| Q17 | Cari harga kompetitor untuk produk celengan target di Shopee | `web_scraper` |
| Q18 | Berapa kisaran harga pasaran tumbler stainless di marketplace Tokopedia? | `web_scraper` |
| Q19 | Riset harga pasaran buku kas keuangan UMKM di marketplace online | `web_scraper` |
| Q20 | Cek tren dan harga pembanding binder tabungan di e-commerce | `web_scraper` |
| Q21 | Berapa harga pasaran rata-rata lakban dan plastik packing paket di Shopee? | `web_scraper` |
| Q22 | Cari referensi harga kompetitor untuk produk stiker label pengiriman | `web_scraper` |

#### Kelompok 4: Ambigu / Out-of-Scope (Q23–Q28)

| ID | Query | Expected Tool |
|----|-------|---------------|
| Q23 | Halo, selamat pagi! Apa saja kemampuan dan analisis yang bisa kamu bantu? | `none` |
| Q24 | Siapa nama presiden Republik Indonesia saat ini? | `none` |
| Q25 | Bagaimana resep dan langkah membuat martabak manis yang bersarang? | `none` |
| Q26 | Tolong transfer saldo toko sebesar Rp 500.000 ke rekening bank saya sekarang | `none` |
| Q27 | Bisa berikan saya source code dan credential database server ini? | `none` |
| Q28 | Cuaca hari ini di Jakarta cerah atau hujan ya? | `none` |

### 8.3 Metrik Evaluasi Otomatis

| Metrik | Cara Hitung | Target |
|--------|-------------|--------|
| `execution_success_rate` | % query yang tidak throw exception | 100% |
| `tool_selection_accuracy` | % query yang pilih tool yang benar | > 85% |
| `avg_latency_sec` | Rata-rata waktu respons per query | < 30 detik |
| `failure_rate` | 1 − execution_success_rate | 0% |
| `provenance_validity_rate` | % web_scraper yang cantumkan URL | 100% |
| `unsupported_answer_rate` | % web_scraper yang beri estimasi tanpa URL | 0% |
| `numerical_correctness` | Diisi **manual** oleh researcher | > 80% |

### 8.4 Cara Cek Tool Selection

```python
# Alias tool dideteksi dengan fleksibel:
tool_alias_map = {
    "python_repl": ["python_repl", "python_repltool"],
    "web_scraper": ["web_scraper", "custom_marketplace_scraper"],
    "forecast_tool": ["forecast_tool", "run_forecast_weekly"]
}

# Cek dari messages LangGraph:
for msg in result.get("messages", []):
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tname = tc.get("name", "")  # ← nama tool yang dipilih LLM
```

### 8.5 Cek Provenance & Fail-Closed

```python
def check_provenance(final_answer, expected_tool):
    """Cek apakah scraper menyertakan URL"""
    if expected_tool != "web_scraper": return None
    return bool(re.search(r'https?://[^\s\)\]]+', final_answer))

def check_unsupported_estimate(final_answer, expected_tool):
    """Deteksi estimasi harga tanpa URL (pelanggaran fail-closed)"""
    if expected_tool != "web_scraper": return None
    has_url = bool(re.search(r'https?://[^\s\)\]]+', final_answer))
    if has_url: return False  # Ada URL = aman
    
    gave_price = bool(re.search(r'Rp\s?[\d.,]+', final_answer))
    declined = any(re.search(p, final_answer) for p in decline_patterns)
    return gave_price and not declined  # True = PELANGGARAN
```

---

## 9. Stack Teknologi

| Komponen | Library / Tool | Versi Minimum |
|----------|---------------|---------------|
| UI Framework | Streamlit | ≥ 1.35.0 |
| LLM | Google Gemini (`langchain-google-genai`) | ≥ 1.0.0 |
| Agent Framework | LangGraph (`create_react_agent`) | ≥ 0.0.30 |
| LLM Abstraction | LangChain Core | ≥ 0.2.0 |
| Data Processing | Pandas | ≥ 2.0.0 |
| Numerik | NumPy | ≥ 1.26.0 |
| Visualisasi | Matplotlib, Plotly | ≥ 3.8.0, ≥ 5.18.0 |
| ML Model | scikit-learn (Ridge/XGBoost) | ≥ 1.4.0 |
| Model Serialization | joblib | ≥ 1.3.0 |
| Web Scraping | requests + BeautifulSoup4 | — |
| Kalender Islam | hijri-converter | — |
| Excel Support | openpyxl | ≥ 3.1.0 |
| Kode Sandbox | subprocess + ast (stdlib Python) | — |

### Model Gemini yang Didukung

| Model | RPM | RPD (Harian) | Keterangan |
|-------|-----|-------------|------------|
| `gemini-3.5-flash-lite` ⭐ | 15 | 500 | **Default & Direkomendasikan** |
| `gemini-3.1-flash-lite` | 15 | 500 | Alternatif stabil |
| `gemini-3.8-flash` | 5 | 20 | Reasoning mendalam |
| `gemini-3.5-flash` | 5 | 20 | Keseimbangan reasoning-kecepatan |

---

## 10. File & Struktur Proyek

```
TAAIEcommerce/
├── streamlit_app.py           # UI utama (2051 baris) — entry point aplikasi
├── agent_core.py              # Inti AI Agent (695 baris) — bebas dari Streamlit
├── eval_agentic_ai.py         # Harness evaluasi headless (379 baris)
│
├── best_model_weekly.pkl      # Model ML terlatih (Ridge Regression Tuned)
├── scaler_weekly.pkl          # StandardScaler untuk fitur numerik
├── feature_columns_weekly.json # Metadata fitur model
│
├── ecommerce_cleaned.csv      # Dataset fallback (4.7 MB)
├── user_data.csv              # Data aktif toko (di-generate saat runtime)
├── agentic_ai_eval_results.csv # Hasil evaluasi 28 query
│
├── requirements.txt           # Dependensi Python
├── .streamlit/
│   └── config.toml            # Konfigurasi tema Streamlit
│
├── CATATAN_AGENTIC_AI.md      # Catatan teknis & referensi cepat
└── README_SISTEM.md           # Dokumentasi sistem
```

### Peran Setiap File Kunci

| File | Peran |
|------|-------|
| `streamlit_app.py` | UI, routing halaman, upload data, dashboard, chatbot UI |
| `agent_core.py` | Tool definitions, sandbox, agent builder, error diagnostics |
| `eval_agentic_ai.py` | Automated testing tanpa UI, 28 test query, metrik evaluasi |
| `user_data.csv` | Jembatan antara data upload pengguna dan AI agent |
| `ecommerce_cleaned.csv` | Fallback data jika user belum upload file |
| `best_model_weekly.pkl` | Model ML untuk `forecast_tool` |

---

> **Catatan:** Sistem ini dirancang sebagai **zero-dependency** antara `agent_core.py` dan Streamlit, sehingga evaluasi headless via `eval_agentic_ai.py` dapat dilakukan tanpa menjalankan UI sama sekali.
