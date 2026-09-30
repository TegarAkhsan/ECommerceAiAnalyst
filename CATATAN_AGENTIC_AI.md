# Catatan: Agentic AI Tools — TAAIEcommerce

File ini adalah catatan teknis & referensi cepat untuk memahami cara kerja
AI Agent dalam sistem ini. Dibuat dari analisis kode agent_core.py & streamlit_app.py.

---

## Arsitektur Agent

Sistem menggunakan **LangGraph ReAct Agent** dengan **Google Gemini** sebagai LLM.
Pola kerja: **Reason - Act - Observe** (berulang sampai jawaban siap).

```
Pengguna -> Streamlit UI -> ReAct Agent (Gemini)
                              |
               +--------------+--------------+
               v              v              v
          Python_REPL   forecast_tool   custom_marketplace_scraper
          (data toko)   (prediksi ML)   (riset pasar web)
```

---

## 3 Tools Aktif

### 1. Python_REPL — Analitik Data Historis Toko
- **Fungsi**: Analisis data transaksi historis dengan kode Python (Pandas)
- **Sumber data**: user_data.csv (read-only, di-copy ke sandbox)
- **Output**: Teks statistik + grafik (temp_plot.png) jika diminta
- **Keamanan**: Sandbox 3-lapis — AST check > subprocess terisolasi > dataset read-only
- **Modul diizinkan**: pandas, numpy, matplotlib, math, statistics, datetime, json, re, collections, itertools
- **Dipanggil ketika**: User tanya omzet, tren, produk terlaris, grafik penjualan

### 2. forecast_tool — Prediksi Omzet ML
- **Fungsi**: Prediksi omzet mingguan ke depan (1-12 minggu)
- **Model**: best_model_weekly.pkl (saat ini Ridge Regression Tuned)
- **Sumber data basis**: user_data.csv -> dipakai untuk fitur lag & rolling mean
- **Fitur kalender dinamis**:
  - is_ramadan -> dihitung via hijri_converter per tanggal prediksi
  - is_harbolnas -> cek apakah tanggal masuk 10-18 Desember
- **Metode**: Rolling forecast — hasil prediksi minggu ke-N jadi input lag untuk minggu ke-(N+1)
- **Dipanggil ketika**: User minta prediksi / proyeksi / forecasting omzet masa depan

### 3. custom_marketplace_scraper — Riset Pasar Web
- **Fungsi**: Scraping harga & produk kompetitor di Shopee/Tokopedia
- **Jalur utama**: Yahoo Search (ramah query marketplace Indonesia)
- **Jalur cadangan**: DuckDuckGo HTML (otomatis jika Yahoo gagal)
- **Output**: Daftar produk + snippet harga + URL sumber
- **Aturan Fail-Closed**: Jika tidak ada hasil -> agent WAJIB jujur, DILARANG mengarang harga
- **Dipanggil ketika**: User tanya harga pasaran, kompetitor, tren produk di marketplace

---

## Sumber Data Agent

```python
# streamlit_app.py (baris 1735-1739)
data_file_path = "ecommerce_cleaned.csv"       # <- fallback default
if 'df_clean' in st.session_state and ...:
    df_clean.to_csv('user_data.csv', index=False)
    data_file_path = "user_data.csv"           # <- data sudah diproses di UI
```

PENTING: user_data.csv BUKAN hasil upload langsung di chatbot.
Ini adalah data yang sudah diproses dari halaman Data di Streamlit,
lalu disimpan otomatis ke session state.

| File                          | Peran                                          |
|-------------------------------|------------------------------------------------|
| user_data.csv                 | Data aktif yang digunakan agent                |
| ecommerce_cleaned.csv         | Fallback jika user_data.csv belum tersedia     |
| best_model_weekly.pkl         | Model ML untuk forecast_tool                   |
| scaler_weekly.pkl             | Scaler fitur numerik untuk model               |
| feature_columns_weekly.json   | Metadata fitur & kolom yang dipakai model      |

---

## Apa yang Bisa Ditanya ke Agent?

| Pertanyaan                              | Tool                        | Sumber                       |
|-----------------------------------------|-----------------------------|------------------------------|
| Total omzet / statistik transaksi       | Python_REPL                 | user_data.csv                |
| Produk terlaris / tren harian & mingguan| Python_REPL                 | user_data.csv                |
| Grafik / visualisasi data toko          | Python_REPL                 | user_data.csv -> temp_plot.png |
| Prediksi omzet N minggu ke depan       | forecast_tool               | user_data.csv + model ML     |
| Detail hasil forecast (sesi yang sama)  | (LLM pakai konteks)         | Dari chat history            |
| Harga kompetitor di Shopee/Tokopedia    | custom_marketplace_scraper  | Internet                     |

---

## Batas Memori Sesi

Agent menggunakan in-memory conversation history (LangGraph message state).
Tidak ada persistensi antar sesi — setiap refresh halaman = konteks baru.

| Skenario                                              | Perilaku                                    |
|-------------------------------------------------------|---------------------------------------------|
| Tanya forecast > tanya detail hasilnya (sesi sama)   | LLM ingat, tidak perlu run tool lagi        |
| Refresh halaman > tanya hasil forecast tadi          | Hilang, agent run forecast_tool ulang        |
| Tanya data historis > tanya grafik (sesi sama)       | Agent lanjut dengan konteks yang ada         |

---

## Routing Logic (System Prompt)

| Kondisi Pertanyaan                        | Tool yang Dipilih             |
|-------------------------------------------|-------------------------------|
| Omzet historis, transaksi, tren, grafik   | Python_REPL                   |
| Prediksi / forecasting / proyeksi         | forecast_tool                 |
| Harga pasar, kompetitor, Shopee/Tokopedia | custom_marketplace_scraper    |

Persona bisnis — nama teknis disembunyikan dari pengguna:

| Nama Teknis                 | Ditampilkan ke User Sebagai    |
|-----------------------------|--------------------------------|
| Python_REPL                 | Sistem Analitik Data Toko      |
| user_data.csv               | Data Penjualan Toko Anda       |
| custom_marketplace_scraper  | Sistem Riset Pasar Marketplace |

---

## Stack Teknologi

| Komponen       | Teknologi                         |
|----------------|-----------------------------------|
| LLM            | Google Gemini (langchain_google_genai) |
| Agent Framework| LangGraph create_react_agent      |
| UI             | Streamlit                         |
| ML Model       | scikit-learn via joblib           |
| Web Scraping   | requests + BeautifulSoup          |
| Sandbox        | subprocess + ast (AST parsing)    |
| Kalender Islam | hijri_converter                   |

---

Dibuat: 2026-09-29 | Referensi: agent_core.py, streamlit_app.py