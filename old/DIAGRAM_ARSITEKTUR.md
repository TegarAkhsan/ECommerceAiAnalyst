# 🏗️ DIAGRAM ARSITEKTUR SISTEM — E-commerce AI Analyst
> **Tugas Akhir** | Stack: Streamlit + LangGraph + Google Gemini + scikit-learn
>
> **Panduan:** File ini berisi semua diagram dalam format Mermaid. Masukkan konten tiap blok ```mermaid ... ``` ke tool seperti Mermaid.live, draw.io, atau AI (ChatGPT/Claude) untuk generate flowchart-nya.

---

## DAFTAR ISI

1. [Arsitektur Sistem (System Architecture)](#1-arsitektur-sistem-berlapis-layer-architecture)
2. [Flowchart Alur Upload & Data Pipeline](#2-flowchart-alur-upload--data-pipeline)
3. [Flowchart Agentic AI — ReAct Loop](#3-flowchart-agentic-ai--react-loop)
4. [Flowchart Tool Routing (Tool Selection)](#4-flowchart-tool-routing--tool-selection)
5. [Flowchart Python REPL Sandbox](#5-flowchart-python-repl-sandbox--keamanan)
6. [Flowchart Forecasting (ML Prediction)](#6-flowchart-forecasting--rolling-forecast-ml)
7. [Flowchart Web Scraper Marketplace](#7-flowchart-web-scraper-marketplace)
8. [Use Case Diagram](#8-use-case-diagram)
9. [Business Process Flow (Alur Proses Bisnis)](#9-business-process-flow-alur-proses-bisnis)
10. [Entity Relationship & Data Flow Diagram](#10-entity-relationship--data-flow-diagram)
11. [Sequence Diagram — Skenario Lengkap](#11-sequence-diagram--skenario-lengkap)
12. [State Machine Diagram — Session State](#12-state-machine-diagram--session-state)
13. [Component Diagram](#13-component-diagram)
14. [Deployment Diagram](#14-deployment-diagram)

---

## 1. Arsitektur Sistem Berlapis (Layer Architecture)

> **Prompt ke AI:** "Buatkan diagram arsitektur berlapis (layered architecture) berdasarkan spesifikasi berikut"

```mermaid
graph TB
    subgraph PRESENTATION["PRESENTATION LAYER — Antarmuka Pengguna"]
        A1["📁 Upload Data\n(Multi-file .xlsx/.csv)"]
        A2["📊 Dashboard Prediksi\n(Forecasting UI)"]
        A3["🤖 Chatbot AI Agent\n(Chat Interface)"]
        A4["⚙️ Pengaturan\n(API Key & Model)"]
    end

    subgraph APPLICATION["APPLICATION LAYER — Logika Aplikasi"]
        B1["streamlit_app.py\nOrchestrator / Controller"]
        B2["Routing & Navigation\n(Sidebar Menu)"]
        B3["Session State Manager\n(st.session_state)"]
        B4["Visualization Engine\n(Plotly + Matplotlib)"]
    end

    subgraph BIZLOGIC["BUSINESS LOGIC LAYER — Proses Utama"]
        C1["Data Pipeline\nread_file() → clean_raw() → make_weekly()"]
        C2["Forecast Engine\nRolling Forecast + Feature Engineering"]
        C3["Agent Orchestrator\nbuild_agent() → run_agent_query()"]
        C4["Insight Generator\ngenerate_forecast_analyst_insights()"]
    end

    subgraph AILOGIC["AI / INTELLIGENCE LAYER — Kecerdasan Buatan"]
        D1["LangGraph ReAct Agent\ncreate_react_agent()"]
        D2["Google Gemini LLM\nChatGoogleGenerativeAI"]
        D3["System Prompt\n(Tool Routing Instructions)"]
        D4["Tool Registry\n[python_repl, forecast_tool, scraper]"]
    end

    subgraph TOOLS["TOOL LAYER — Instrumen AI"]
        E1["🔬 Python_REPL\nSandboxed Execution"]
        E2["📈 forecast_tool\nML Model Runner"]
        E3["🌐 marketplace_scraper\nWeb Search"]
    end

    subgraph DATA["DATA LAYER — Penyimpanan Data"]
        F1["user_data.csv\nData Aktif Toko"]
        F2["ecommerce_cleaned.csv\nFallback Dataset"]
        F3["best_model_weekly.pkl\nTrained ML Model"]
        F4["scaler_weekly.pkl\nFeature Scaler"]
        F5["feature_columns_weekly.json\nModel Metadata"]
    end

    subgraph EXTERNAL["EXTERNAL SERVICES"]
        G1["Google Gemini API\ngenerativelanguage.googleapis.com"]
        G2["Yahoo Search\nsearch.yahoo.com"]
        G3["DuckDuckGo HTML\nhtml.duckduckgo.com"]
    end

    A1 & A2 & A3 & A4 --> B1
    B1 --> B2 & B3 & B4
    B1 --> C1 & C2 & C3 & C4
    C1 --> F1
    C2 --> F3 & F4 & F5
    C3 --> D1
    D1 --> D2 & D3 & D4
    D4 --> E1 & E2 & E3
    E1 --> F1 & F2
    E2 --> F3 & F4 & F5
    E3 --> G2 & G3
    D2 --> G1

    style A1 fill:#dbeafe,stroke:#3b82f6
    style A2 fill:#dbeafe,stroke:#3b82f6
    style A3 fill:#dbeafe,stroke:#3b82f6
    style A4 fill:#dbeafe,stroke:#3b82f6
    style D1 fill:#fef3c7,stroke:#f59e0b
    style D2 fill:#fef3c7,stroke:#f59e0b
    style E1 fill:#dcfce7,stroke:#22c55e
    style E2 fill:#dcfce7,stroke:#22c55e
    style E3 fill:#dcfce7,stroke:#22c55e
```

---

## 2. Flowchart Alur Upload & Data Pipeline

> **Prompt ke AI:** "Buatkan flowchart detail alur proses upload data dan data pipeline dari input file hingga penyimpanan"

```mermaid
flowchart TD
    START(["🚀 Pengguna Buka Halaman Upload Data"]) --> A

    A["📂 Pilih File\n(.xlsx / .xls / .csv)\nbisa multi-file"] --> B{File dipilih?}
    B -- Tidak --> WAIT(["⏳ Tunggu Input"])
    B -- Ya --> C

    C["🔍 read_file()\nDeteksi format file"] --> D{Format?}
    D -- ".xlsx / .xls" --> E["pd.read_excel()\nMembaca file Excel"]
    D -- ".csv" --> F["Coba encoding:\n1. utf-8\n2. latin-1\n3. cp1252"]
    F --> G{Berhasil?}
    G -- Ya --> H["pd.read_csv()"]
    G -- Tidak semua --> I["Fallback: utf-8\nerrors='replace'"]
    I --> H
    E --> J["DataFrame Mentah / Raw"]
    H --> J

    J --> K{Multi-file?}
    K -- Ya --> L["pd.concat()\nGabung semua DataFrame"]
    K -- Tidak --> M
    L --> M

    M["🧹 clean_raw()\nPembersihan Data"] --> M1
    M1["1. Normalisasi kolom tanggal"] --> M2
    M2["2. Bersihkan Total Pembayaran\nHapus titik, koma ke titik, to numeric"] --> M3
    M3["3. Filter Status: Hanya 'Selesai'"] --> M4
    M4["4. Filter Nilai: Hapus baris <= 0"] --> M5
    M5["5. Parse Tanggal ke datetime"] --> M6
    M6["6. reset_index(drop=True)"] --> N

    N["DataFrame Bersih / df_clean"] --> O

    O["📅 make_weekly()\nAgregasi Mingguan"] --> O1
    O1["Deduplikasi order_id\n1 order = 1 baris"] --> O2
    O2["Groupby order_date, freq='W'\nSum Total Pembayaran per minggu"] --> O3
    O3["Reindex full date range\nIsi gap minggu kosong = 0"] --> P

    P["weekly_hist: Time Series Omzet Mingguan"] --> Q

    N --> R["💾 Simpan ke Session State\nst.session_state['df_clean']\nst.session_state['weekly_hist']"]
    P --> R

    R --> S["💾 Export ke Disk\nuser_data.csv\n(Jembatan ke AI Agent)"]

    S --> T{Upload berhasil?}
    T -- Ya --> U["✅ Tampilkan Metrik Ringkasan\nTotal Transaksi, Revenue,\nJumlah Produk, Rentang Tanggal"]
    T -- Error --> V["❌ Tampilkan Pesan Error"]

    U --> W["📊 Tampilkan Visualisasi\nTren Revenue + Top 10 Produk"]
    W --> END(["✅ Data Siap Digunakan"])

    style START fill:#22c55e,color:#fff
    style END fill:#3b82f6,color:#fff
    style V fill:#ef4444,color:#fff
```

---

## 3. Flowchart Agentic AI — ReAct Loop

> **Prompt ke AI:** "Buatkan flowchart arsitektur ReAct (Reasoning + Acting) agent dengan loop iteratif"

```mermaid
flowchart TD
    USER(["👤 Pengguna Kirim Pertanyaan\nContoh: 'Berapa omzet bulan ini?'"]) --> A

    A["📝 Konstruksi Agent Input\nTambahkan konteks path file data\ndan panduan komunikasi bisnis"] --> B

    B["🤖 LangGraph invoke()\nagent_graph.invoke({'messages': []})"] --> C

    C{{"🧠 REASON\nGemini LLM membaca:\nPertanyaan + System Prompt\n+ Riwayat messages\n+ Docstring setiap tool"}} --> D

    D{Perlu tool?} -- Ya --> E
    D -- Tidak perlu tool --> K

    E{{"Pilih Tool & Parameter:\nNama tool + argumen input"}} --> F

    F{Tool dipanggil} --> F1 & F2 & F3

    F1["🔬 Python_REPL\neksekusi kode Pandas\ndi sandbox terisolasi"] --> G
    F2["📈 forecast_tool\njalankan ML model\nRolling Forecast"] --> G
    F3["🌐 marketplace_scraper\nscraping Yahoo/DDG\nharga kompetitor"] --> G

    G["📨 Tool Message\nHasil dikembalikan ke LLM\nsebagai ToolMessage"] --> H

    H{{"👁️ OBSERVE\nLLM membaca hasil tool:\nApakah jawaban sudah cukup?"}} --> I

    I{Jawaban cukup?} -- Tidak, butuh tool lagi --> C
    I -- Ya --> K

    K["✍️ Tulis Jawaban Final\nDalam bahasa bisnis\nSembunyikan nama teknis"] --> L

    L{Ada grafik?\ntemp_plot.png?} -- Ya --> M
    L -- Tidak --> N

    M["📸 Baca & Hapus temp_plot.png\nimage_bytes = file content\nos.remove temp_plot.png"] --> N

    N["📦 Return Result\n{answer, image_bytes, messages}"] --> O

    O["🖥️ Streamlit Render\nTampilkan teks markdown\nTampilkan gambar jika ada"] --> END(["✅ Respons Ditampilkan"])

    style USER fill:#6366f1,color:#fff
    style C fill:#f59e0b,stroke:#d97706,color:#000
    style H fill:#f59e0b,stroke:#d97706,color:#000
    style F1 fill:#22c55e,color:#fff
    style F2 fill:#22c55e,color:#fff
    style F3 fill:#22c55e,color:#fff
    style END fill:#3b82f6,color:#fff
```

---

## 4. Flowchart Tool Routing & Tool Selection

> **Prompt ke AI:** "Buatkan flowchart decision tree untuk pemilihan tool berdasarkan intent pengguna dalam sistem AI agent"

```mermaid
flowchart TD
    Q(["❓ Pertanyaan Pengguna"]) --> A

    A["🧠 Gemini LLM\nAnalisis Intent Pertanyaan\nvia System Prompt + Docstring Tool"] --> B

    B{Kategori Intent?}

    B --> C1["🔵 HISTORIS / ANALITIK DATA"]
    B --> C2["🟡 PREDIKSI / FORECAST"]
    B --> C3["🟢 RISET PASAR / KOMPETITOR"]
    B --> C4["⚪ OUT-OF-SCOPE / UMUM"]

    C1 --> D1["Trigger Keywords:\nomzet, revenue, transaksi,\ntren, grafik, produk terlaris,\nAOV, pembayaran, pesanan,\nmetode bayar, kota, hari, ongkir"]
    C2 --> D2["Trigger Keywords:\nprediksi, peramalan, proyeksi,\nforecast, perkiraan,\nbulan depan, minggu depan"]
    C3 --> D3["Trigger Keywords:\nharga, pasaran, kompetitor,\npesaing, Shopee, Tokopedia,\nmarketplace, riset pasar"]
    C4 --> D4["Pertanyaan non-bisnis:\ncuaca, resep, transfer uang,\npertanyaan umum, basa-basi"]

    D1 --> E1["🔬 Tool: Python_REPL\nInput: kode Python Pandas\nData: user_data.csv"]
    D2 --> E2["📈 Tool: forecast_tool\nInput: n_weeks (1-12)\nData: best_model_weekly.pkl"]
    D3 --> E3["🌐 Tool: marketplace_scraper\nInput: query string\nSumber: Yahoo/DuckDuckGo"]
    D4 --> E4["💬 Jawab Langsung\nTanpa tool\nRespon sopan/informatif"]

    E1 --> F1["Output: Teks statistik\n+ temp_plot.png opsional"]
    E2 --> F2["Output: Prediksi per minggu\nTotal + rata-rata akumulasi"]
    E3 --> F3{Data ditemukan?}
    E4 --> F4["Jawaban langsung\ndalam bahasa bisnis"]

    F3 -- Ya --> F3A["Output: Daftar produk\n+ snippet harga\n+ URL sumber wajib"]
    F3 -- Tidak --> F3B["⚠️ FAIL-CLOSED\nTidak boleh mengarang harga!\nSarankan kata kunci lain"]

    style C1 fill:#dbeafe,stroke:#3b82f6
    style C2 fill:#fef9c3,stroke:#ca8a04
    style C3 fill:#dcfce7,stroke:#16a34a
    style C4 fill:#f1f5f9,stroke:#94a3b8
    style E1 fill:#3b82f6,color:#fff
    style E2 fill:#ca8a04,color:#fff
    style E3 fill:#16a34a,color:#fff
    style F3B fill:#ef4444,color:#fff
```

---

## 5. Flowchart Python REPL Sandbox & Keamanan

> **Prompt ke AI:** "Buatkan flowchart 3-layer security architecture untuk sandboxed Python code execution oleh AI agent"

```mermaid
flowchart TD
    IN(["📥 Kode Python dari LLM"]) --> L1A

    subgraph L1["🛡️ LAPISAN 1: AST Static Analysis"]
        L1A["ast.parse(code)\nParse ke Abstract Syntax Tree"] --> L1B{Syntax error?}
        L1B -- Ya --> ERR1["❌ TOLAK: SyntaxError"]
        L1B -- Tidak --> L1C
        L1C["Walk seluruh AST nodes"] --> L1D{Ada Import?}
        L1D -- Ya --> L1E{Modul dalam whitelist?}
        L1E -- Tidak --> ERR2["❌ TOLAK\nModul tidak diizinkan\nWhitelist: pandas, numpy,\nmatplotlib, math, statistics,\ndatetime, json, re, collections"]
        L1E -- Ya --> L1F
        L1D -- Tidak --> L1F
        L1F{Ada nama berbahaya?} -- Ya --> ERR3["❌ TOLAK\nBlacklist: eval, exec, open,\nos, sys, subprocess, socket,\nrequests, urllib, getattr..."]
        L1F -- Tidak --> L1G{Ada atribut dunder?}
        L1G -- Ya --> ERR4["❌ TOLAK\n__class__, __bases__\nSandbox-escape prevention"]
        L1G -- Tidak --> PASS1["✅ LOLOS Lapisan 1"]
    end

    PASS1 --> L2A

    subgraph L2["🏠 LAPISAN 2: Subprocess Isolation"]
        L2A["tempfile.mkdtemp()\nBuat direktori sandbox sementara"] --> L2B
        L2B["shutil.copy2(user_data.csv, sandbox)\nos.chmod(dst, 0o444) → read-only"] --> L2C
        L2C["Tulis script ke sandbox\n_sandbox_script.py"] --> L2D
        L2D["subprocess.run(\n  python -I script,\n  cwd=sandbox_dir,\n  timeout=10s\n)"] --> L2E{Timeout?}
        L2E -- Ya --> ERR5["❌ HENTIKAN PAKSA\nMelebihi 10 detik"]
        L2E -- Tidak --> L2F{returncode != 0?}
        L2F -- Ya --> ERR6["❌ Runtime Error\nKembalikan stderr ke LLM"]
        L2F -- Tidak --> PASS2["✅ LOLOS Lapisan 2"]
    end

    subgraph RL["📊 Resource Limits POSIX"]
        R1["RLIMIT_CPU: 10 detik CPU"]
        R2["RLIMIT_AS: 512 MB memori"]
        R3["RLIMIT_NPROC: 16 proses"]
        R4["RLIMIT_NOFILE: 64 file descriptor"]
    end

    L2D -.->|preexec_fn| RL

    PASS2 --> L3A

    subgraph L3["📤 LAPISAN 3: Output Processing"]
        L3A["Ambil stdout\nproc.stdout.strip()"] --> L3B{Ada temp_plot.png\ndi sandbox?}
        L3B -- Ya --> L3C["shutil.copy2(\n  sandbox/temp_plot.png,\n  root/temp_plot.png\n)"]
        L3B -- Tidak --> L3D
        L3C --> L3D["shutil.rmtree(sandbox)\nBersihkan direktori sementara"]
    end

    L3D --> OUT(["✅ Output: stdout text\n+ temp_plot.png jika ada"])

    style ERR1 fill:#ef4444,color:#fff
    style ERR2 fill:#ef4444,color:#fff
    style ERR3 fill:#ef4444,color:#fff
    style ERR4 fill:#ef4444,color:#fff
    style ERR5 fill:#ef4444,color:#fff
    style ERR6 fill:#f97316,color:#fff
    style PASS1 fill:#22c55e,color:#fff
    style PASS2 fill:#22c55e,color:#fff
    style OUT fill:#3b82f6,color:#fff
```

---

## 6. Flowchart Forecasting — Rolling Forecast ML

> **Prompt ke AI:** "Buatkan flowchart proses rolling forecast machine learning dengan feature engineering time series"

```mermaid
flowchart TD
    START(["📈 forecast_tool dipanggil\nn_weeks = 1 sampai 12"]) --> A

    A{File model tersedia?\nbest_model_weekly.pkl} -- Tidak --> FALLBACK["⚠️ Fallback Response\nEstimasi rata-rata Rp 15 jt/minggu"]
    A -- Ya --> B

    B["Load artifacts:\nbest_model_weekly.pkl → model\nscaler_weekly.pkl → StandardScaler\nfeature_columns_weekly.json → metadata"] --> C

    C["Load data historis\nuser_data.csv → groupby W → weekly_hist"] --> D

    D["last_dt = tanggal minggu terakhir\ndf_roll = salinan weekly_hist"] --> E

    E(["🔁 MULAI ROLLING LOOP\ni = 0 hingga n_weeks - 1"]) --> F

    F["future_dt = last_dt + (i+1) minggu\nTambah placeholder baris ke df_roll"] --> G

    subgraph FE["⚙️ FEATURE ENGINEERING per iterasi"]
        G["📅 Kalender: year, month,\nweek_of_year, quarter"] --> G1
        G1["〰️ Fourier: sin_month, cos_month,\nsin_year, cos_year"] --> G2
        G2["📉 Trend: week_index\n(hari dari 2023-12-31 / 7)"] --> G3
        G3["⏪ Lag Features:\nlag_1, lag_4, lag_12\n(dari history yang ada)"] --> G4
        G4["📊 Rolling Features:\nrolling_4w_mean, rolling_12w_mean"] --> G5
        G5["🌙 Kalender Islam:\nis_ramadan (hijri_converter)\nis_harbolnas (10-18 Desember)"] --> G6
        G6["🚩 is_holiday_gap = 0"]
    end

    G6 --> H["scaler.transform(X[scale_cols])\nNormalisasi fitur numerik"] --> J

    J["🔮 Prediksi:\ny_log = model.predict(X)\ny_idr = np.expm1(y_log)\n(inverse log1p transform)"] --> K

    K["Simpan:\norder_week = future_dt\npredicted_revenue = y_idr"] --> L

    L["Update df_roll:\nTambahkan baris prediksi\nagar iterasi berikut bisa pakai\nnilai ini sebagai lag"] --> M

    M{i < n_weeks - 1?} -- Ya, lanjut loop --> F
    M -- Selesai --> N

    N["Format output string:\nPrediksi per minggu\nTotal akumulasi\nRata-rata mingguan"] --> END(["✅ Return string prediksi ke LLM"])

    style START fill:#6366f1,color:#fff
    style FALLBACK fill:#f59e0b,color:#000
    style END fill:#22c55e,color:#fff
```

---

## 7. Flowchart Web Scraper Marketplace

> **Prompt ke AI:** "Buatkan flowchart web scraping dengan dual-path strategy dan fail-closed safety mechanism untuk keamanan data"

```mermaid
flowchart TD
    START(["🌐 marketplace_scraper dipanggil\nquery = string pencarian"]) --> A

    A["URL Encode query\nurllib.parse.quote(query)"] --> B

    subgraph YAHOO["🔵 JALUR UTAMA — Yahoo Search"]
        B["GET https://search.yahoo.com/search\n?p={encoded_query}\nheaders: User-Agent Indonesia"] --> B1{Status 200?}
        B1 -- Tidak --> CHECK
        B1 -- Ya --> B2["BeautifulSoup parse HTML\nCari div.algo max 5 hasil"]
        B2 --> B3["Ekstrak per hasil:\ntitle dari h3\nraw_url dari a href\nsnippet dari compText/p"]
        B3 --> B4["Bersihkan URL Yahoo redirect:\nRegex /RU=(.*?)/RK=\nurllib.parse.unquote"]
        B4 --> B5{Title & URL ada?}
        B5 -- Ya --> B6["Tambah ke extracted_data:\n- Title\n  Ringkasan: snippet\n  Tautan: clean_url"]
        B5 -- Tidak --> CHECK
        B6 --> CHECK
    end

    CHECK{extracted_data kosong?} -- Tidak kosong --> OUT
    CHECK -- Masih kosong --> C

    subgraph DDG["🟡 JALUR CADANGAN — DuckDuckGo HTML"]
        C["GET https://html.duckduckgo.com\n/html/?q={encoded_query}"] --> C1{Status 200?}
        C1 -- Tidak --> FAIL
        C1 -- Ya --> C2["BeautifulSoup parse HTML\nCari div.result__body max 4"]
        C2 --> C3["Ekstrak:\na.result__url\na.result__snippet"]
        C3 --> C4["Bersihkan DDG redirect\nparse param 'uddg='"]
        C4 --> C5{URL & snippet ada?}
        C5 -- Ya --> C6["Tambah ke extracted_data"]
        C5 -- Tidak --> FAIL
        C6 --> OUT
    end

    FAIL["extracted_data = kosong"] --> FC

    subgraph FAILCLOSED["🔴 FAIL-CLOSED MECHANISM"]
        FC["Return SUMBER_TIDAK_TERSEDIA:\nDILARANG berikan estimasi harga\nDILARANG mengarang angka\nSarankan kata kunci lebih spesifik"]
    end

    OUT(["✅ Return hasil riset pasar:\nDaftar produk + harga + URL sumber"])

    style START fill:#6366f1,color:#fff
    style FC fill:#ef4444,color:#fff
    style FAIL fill:#fca5a5,stroke:#ef4444
    style OUT fill:#22c55e,color:#fff
```

---

## 8. Use Case Diagram

> **Prompt ke AI:** "Buatkan use case diagram UML lengkap dengan semua aktor, use case, dan relasi include/extend untuk sistem AI Analyst E-commerce"

```mermaid
graph LR
    subgraph ACTORS["👥 AKTOR"]
        U["👤 Pemilik Toko\n(Primary User)"]
        DEV["👨‍💻 Developer\n(Admin)"]
        EXT1["🤖 Google Gemini\n(External AI)"]
        EXT2["🌐 Search Engine\n(Yahoo / DDG)"]
    end

    subgraph SYSTEM["📱 SISTEM: E-commerce AI Analyst"]
        subgraph UPLOAD["🔷 Upload & Data Management"]
            UC1["UC-01\nUpload File Laporan\n.xlsx / .csv"]
            UC2["UC-02\nLihat Ringkasan Data\nMetrik & Visualisasi"]
        end

        subgraph FORECAST["🔶 Forecasting"]
            UC3["UC-03\nLihat Prediksi Omzet\n1-12 Minggu ke Depan"]
            UC4["UC-04\nLihat Analyst Insights\nOtomatis (Tren, Peak, Restock)"]
        end

        subgraph CHATBOT["🟢 AI Chatbot Agent"]
            UC5["UC-05\nTanya Analisis Data\nHistoris Toko"]
            UC6["UC-06\nMinta Visualisasi\nGrafik Batang / Line"]
            UC7["UC-07\nMinta Prediksi\nvia Chat"]
            UC8["UC-08\nRiset Harga Kompetitor\nMarketplace"]
            UC9["UC-09\nLihat Riwayat Chat"]
            UC10["UC-10\nReset Percakapan"]
        end

        subgraph SETTINGS["⚙️ Pengaturan"]
            UC11["UC-11\nInput Google API Key"]
            UC12["UC-12\nPilih Model Gemini"]
        end

        subgraph EVAL["🛡️ Evaluasi (Developer)"]
            UC13["UC-13\nJalankan 28 Test Query\nHeadless Evaluation"]
            UC14["UC-14\nExport Hasil Evaluasi\nCSV Report"]
        end
    end

    U --> UC1 & UC2 & UC3 & UC4
    U --> UC5 & UC6 & UC7 & UC8 & UC9 & UC10
    U --> UC11 & UC12
    DEV --> UC13 & UC14

    UC5 -.->|uses| EXT1
    UC6 -.->|uses| EXT1
    UC7 -.->|uses| EXT1
    UC8 -.->|uses| EXT1
    UC8 -.->|uses| EXT2

    UC1 -.->|includes| UC2
    UC3 -.->|includes| UC4
    UC5 -.->|extends| UC6
    UC7 -.->|extends| UC3

    style U fill:#6366f1,color:#fff
    style DEV fill:#8b5cf6,color:#fff
    style EXT1 fill:#f59e0b,color:#000
    style EXT2 fill:#f59e0b,color:#000
```

---

## 9. Business Process Flow (Alur Proses Bisnis)

> **Prompt ke AI:** "Buatkan BPMN-style business process diagram alur lengkap penggunaan sistem dari perspektif pemilik toko UMKM"

```mermaid
flowchart TD
    subgraph OWNER["🏪 PEMILIK TOKO"]
        P_START(["Punya data laporan\nShopee / Tokopedia"]) --> P1
        P1["Buka Aplikasi\nE-commerce AI Analyst"] --> P2
        P2{Sudah punya\nAPI Key?} -- Tidak --> P3
        P2 -- Ya --> P5
        P3["Buka Google AI Studio\nDaftar / Login gratis"] --> P4
        P4["Copy API Key\nPaste ke menu Pengaturan"] --> P5
        P5["📁 Upload file laporan\n.xlsx atau .csv"] --> P6
        P6{Upload berhasil?} -- Tidak --> P7
        P6 -- Ya --> P8
        P7["Cek format file\nCoba file lain"] --> P5
        P8["Baca Ringkasan Otomatis:\nTotal transaksi, revenue\nProduk terlaris, tren"] --> P9
        P9{Ingin prediksi?} -- Ya --> P10
        P9 -- Ingin tanya AI --> P12
        P10["📊 Buka Dashboard Prediksi\nPilih jumlah minggu 1-12"] --> P11
        P11["Baca Analyst Insights:\nTren naik/turun/stabil\nVolatilitas, periode puncak\nRekomen restocking & marketing"] --> P12
        P12["💬 Buka Chatbot AI\nKetik pertanyaan bisnis"] --> AGENT
    end

    subgraph AGENT["🤖 AI AGENT (Background)"]
        A1["Analisis intent\nGemini LLM"] --> A2
        A2{Intent?}
        A2 -- Analisis Historis --> A3["Python_REPL\nAnalisis user_data.csv"]
        A2 -- Forecast --> A4["forecast_tool\nRolling ML Forecast"]
        A2 -- Riset Pasar --> A5["marketplace_scraper\nYahoo / DuckDuckGo"]
        A3 & A4 & A5 --> A6["Rangkum dalam\nbahasa bisnis"]
        A6 --> P14
    end

    subgraph RESULT["📊 HASIL & KEPUTUSAN BISNIS"]
        P14["Baca Jawaban AI\nteks + grafik jika ada"] --> P15
        P15{Jawaban memadai?} -- Tidak --> P16
        P15 -- Ya --> P17
        P16["Tanya ulang dengan\npertanyaan lebih spesifik"] --> P12
        P17["🎯 Keputusan Bisnis"] --> P18 & P19 & P20
        P18["🛒 Atur stok produk\nberdasarkan prediksi peak"]
        P19["💰 Tentukan harga jual\nberdasarkan riset kompetitor"]
        P20["📢 Jadwalkan promosi\nH-7 sebelum peak week"]
        P18 & P19 & P20 --> PEND(["✅ Keputusan Bisnis\nBerbasis Data"])
    end

    style P_START fill:#6366f1,color:#fff
    style PEND fill:#22c55e,color:#fff
    style A1 fill:#f59e0b,color:#000
    style A3 fill:#3b82f6,color:#fff
    style A4 fill:#3b82f6,color:#fff
    style A5 fill:#3b82f6,color:#fff
```

---

## 10. Entity Relationship & Data Flow Diagram

> **Prompt ke AI:** "Buatkan DFD (Data Flow Diagram) level 1 yang menampilkan seluruh aliran data, proses, dan penyimpanan dalam sistem"

```mermaid
flowchart LR
    subgraph INPUT["📥 INPUT"]
        I1[/"File Laporan\n.xlsx / .csv"/]
        I2[/"API Key\nGoogle Gemini"/]
        I3[/"Pertanyaan Pengguna\nNatural Language"/]
    end

    subgraph PROCESS["⚙️ PROSES UTAMA"]
        P1["🔄 Data Pipeline\nread → clean → aggregate"]
        P2["🤖 Agent Orchestrator\nLangGraph ReAct"]
        P3["📈 ML Forecasting\nRolling Forecast Engine"]
        P4["🔬 Code Execution\nSandboxed Python REPL"]
        P5["🌐 Web Scraping\nMarketplace Search"]
    end

    subgraph STORE["💾 DATA STORE"]
        DS1[("user_data.csv\nData Aktif Toko")]
        DS2[("ecommerce_cleaned.csv\nFallback Dataset")]
        DS3[("best_model_weekly.pkl\nML Model Terlatih")]
        DS4[("scaler_weekly.pkl\nFeature Scaler")]
        DS5[("feature_columns.json\nModel Metadata")]
        DS6[("Session State\nst.session_state")]
        DS7[("temp_plot.png\nGrafik Sementara")]
    end

    subgraph OUTPUT["📤 OUTPUT"]
        O1[/"Dashboard\nMetrik & Visualisasi"/]
        O2[/"Prediksi Omzet\n+ Analyst Insights"/]
        O3[/"Jawaban AI\nTeks + Grafik"/]
        O4[/"Hasil Evaluasi\nCSV Report"/]
    end

    I1 --> P1
    I2 & I3 --> P2
    P1 --> DS1 & DS6
    P1 --> O1
    DS1 --> P2
    DS2 --> P2
    P2 --> P3 & P4 & P5
    DS3 & DS4 & DS5 --> P3
    P3 --> O2
    DS1 --> P4
    P4 --> DS7
    DS7 --> O3
    P5 --> O3
    P3 --> O3
    P2 --> O4

    style I1 fill:#dbeafe,stroke:#3b82f6
    style I2 fill:#dbeafe,stroke:#3b82f6
    style I3 fill:#dbeafe,stroke:#3b82f6
    style O1 fill:#dcfce7,stroke:#22c55e
    style O2 fill:#dcfce7,stroke:#22c55e
    style O3 fill:#dcfce7,stroke:#22c55e
    style O4 fill:#dcfce7,stroke:#22c55e
    style DS1 fill:#fef3c7,stroke:#f59e0b
    style DS3 fill:#fef3c7,stroke:#f59e0b
```

---

## 11. Sequence Diagram — Skenario Lengkap

### 11A. Skenario: Tanya Analisis Data (Python REPL)

> **Prompt ke AI:** "Buatkan sequence diagram interaksi antara user, Streamlit, LangGraph, Gemini LLM, dan Python REPL sandbox"

```mermaid
sequenceDiagram
    actor User as 👤 Pemilik Toko
    participant ST as 🖥️ Streamlit UI
    participant AC as ⚙️ agent_core.py
    participant LG as 🔗 LangGraph
    participant GEM as 🤖 Gemini LLM
    participant REPL as 🔬 Python REPL
    participant FS as 💾 Filesystem

    User->>ST: Ketik: "Produk apa yang paling laris bulan ini?"
    ST->>ST: Simpan ke chat history
    ST->>AC: run_agent_query(agent_graph, query, "user_data.csv")
    AC->>LG: agent_graph.invoke(messages:[HumanMessage])
    LG->>GEM: Kirim messages + system prompt
    GEM->>GEM: Analisis intent → pilih Python_REPL
    GEM-->>LG: AIMessage tool_calls: Python_REPL(code=...)
    LG->>REPL: python_repl(code="import pandas as pd...")
    REPL->>REPL: _sandbox_validate_code → AST check
    REPL->>FS: tempfile.mkdtemp() → buat sandbox
    FS-->>REPL: /tmp/agent_sandbox_xxx/
    REPL->>FS: copy user_data.csv → sandbox + chmod 0o444
    REPL->>REPL: subprocess.run(python -I script, timeout=10s)
    REPL-->>REPL: stdout: "Top 5 Produk Terlaris: 1. Celengan..."
    REPL->>FS: shutil.rmtree(sandbox) → bersihkan
    REPL-->>LG: ToolMessage: "Top 5 Produk Terlaris..."
    LG->>GEM: Kirim ToolMessage sebagai observasi
    GEM->>GEM: Rangkum dalam bahasa bisnis
    GEM-->>LG: AIMessage: "Berdasarkan Data Penjualan Toko Anda..."
    LG-->>AC: messages: [..., AIMessage]
    AC-->>ST: answer, image_bytes=None, messages
    ST->>ST: Render markdown
    ST-->>User: Tampilkan jawaban teks
```

### 11B. Skenario: Minta Grafik + Prediksi (Multi-Tool)

```mermaid
sequenceDiagram
    actor User as 👤 Pemilik Toko
    participant ST as 🖥️ Streamlit UI
    participant LG as 🔗 LangGraph
    participant GEM as 🤖 Gemini LLM
    participant REPL as 🔬 Python REPL
    participant FC as 📈 forecast_tool
    participant FS as 💾 Filesystem

    User->>ST: "Grafik omzet mingguan dan prediksi 4 minggu depan"
    ST->>LG: invoke messages
    LG->>GEM: Analisis intent

    Note over GEM: Intent 1: Grafik historis → Python_REPL
    GEM-->>LG: tool_call: Python_REPL (buat grafik)
    LG->>REPL: python_repl(code=...plt.savefig('temp_plot.png')...)
    REPL->>FS: Eksekusi di sandbox → hasilkan temp_plot.png
    REPL->>FS: Copy temp_plot.png ke root proyek
    REPL-->>LG: ToolMessage: "Grafik berhasil dibuat"

    Note over GEM: Intent 2: Prediksi → forecast_tool
    GEM-->>LG: tool_call: forecast_tool(n_weeks=4)
    LG->>FC: forecast_tool(n_weeks=4)
    FC->>FS: Load best_model_weekly.pkl + scaler + metadata
    FC->>FC: Rolling Forecast × 4 iterasi dengan feature engineering
    FC-->>LG: ToolMessage: "Prediksi: Minggu 1: Rp X..."

    GEM->>GEM: Gabungkan semua observasi
    GEM-->>LG: AIMessage jawaban final
    LG-->>ST: answer, image_bytes(temp_plot.png), messages
    ST->>FS: Baca temp_plot.png → image_bytes
    ST->>FS: os.remove temp_plot.png
    ST->>ST: st.markdown(answer) + st.image(image_bytes)
    ST-->>User: Tampilkan teks + grafik omzet
```

---

## 12. State Machine Diagram — Session State

> **Prompt ke AI:** "Buatkan state machine diagram manajemen state aplikasi Streamlit multi-halaman dengan session state"

```mermaid
stateDiagram-v2
    [*] --> INIT : App dibuka

    state INIT {
        [*] --> CheckAPIKey
        CheckAPIKey --> NoAPIKey : Belum ada API Key
        CheckAPIKey --> HasAPIKey : API Key tersedia
    }

    INIT --> ONBOARDING : First visit / no data

    state ONBOARDING {
        [*] --> ShowWelcome
        ShowWelcome --> ShowFeatures
        ShowFeatures --> PromptUpload
    }

    ONBOARDING --> UPLOAD_PAGE : User klik mulai

    state UPLOAD_PAGE {
        [*] --> WaitingFile
        WaitingFile --> Processing : File dipilih
        Processing --> DataReady : Berhasil clean
        Processing --> ErrorState : Gagal parse
        ErrorState --> WaitingFile : User coba lagi
    }

    DataReady --> DATA_READY : Upload sukses

    state DATA_READY {
        note right of DATA_READY
            st.session_state:
            df_clean = DataFrame bersih
            weekly_hist = Time Series
            user_data.csv = ditulis ke disk
        end note
    }

    DATA_READY --> DASHBOARD_PAGE : Klik Dashboard
    DATA_READY --> CHATBOT_PAGE : Klik Chatbot
    DATA_READY --> UPLOAD_PAGE : Upload data baru

    state DASHBOARD_PAGE {
        [*] --> RunForecast
        RunForecast --> ForecastDone : Model tersedia
        RunForecast --> FallbackMode : Model tidak ada
        ForecastDone --> ShowInsights
        ShowInsights --> ShowCharts
    }

    state CHATBOT_PAGE {
        [*] --> WaitingQuery
        WaitingQuery --> Processing : User kirim pesan
        Processing --> ToolExec : LLM pilih tool
        ToolExec --> WaitingLLM : Tool selesai
        WaitingLLM --> ShowResponse : LLM jawaban final
        ShowResponse --> WaitingQuery : Siap query berikut
        ShowResponse --> Cleared : User klik reset
        Cleared --> WaitingQuery
    }

    DASHBOARD_PAGE --> DATA_READY : Kembali
    CHATBOT_PAGE --> DATA_READY : Kembali

    state SETTINGS_PAGE {
        [*] --> EnterAPIKey
        EnterAPIKey --> ValidateKey : Submit
        ValidateKey --> KeyValid : Valid
        ValidateKey --> KeyInvalid : Invalid
        KeyInvalid --> EnterAPIKey
    }

    DATA_READY --> SETTINGS_PAGE : Klik Pengaturan
    INIT --> SETTINGS_PAGE : Klik Pengaturan
    SETTINGS_PAGE --> INIT : Simpan
```

---

## 13. Component Diagram

> **Prompt ke AI:** "Buatkan component diagram UML yang menampilkan dependencies dan interface antar komponen sistem"

```mermaid
graph TB
    subgraph FRONTEND["📱 Frontend Components (streamlit_app.py)"]
        NAV["🧭 Navigation\nSidebar routing"]
        PAGE_UPLOAD["📁 Upload Page"]
        PAGE_DASH["📊 Dashboard Page"]
        PAGE_CHAT["💬 Chatbot Page"]
        PAGE_SET["⚙️ Settings Page"]
    end

    subgraph BACKEND["🔧 Backend Services"]
        SVC_DATA["📦 Data Service\nread_file()\nclean_raw()\nmake_weekly()"]
        SVC_FC["🔮 Forecast Service\nrun_forecast_weekly()\ngenerate_insights()"]
        SVC_AGENT["🤖 Agent Service\nbuild_agent()\nrun_agent_query()"]
        SVC_VIZ["📈 Viz Engine\nPlotly + Matplotlib"]
    end

    subgraph AGENT_CORE["🛠️ Agent Core (agent_core.py)"]
        TOOL_REPL["🔬 Python_REPL Tool\nrun_sandboxed_python()\n_sandbox_validate_code()"]
        TOOL_FC["📈 forecast_tool\nRolling Forecast\nFeature Engineering"]
        TOOL_SCR["🌐 marketplace_scraper\nYahoo + DDG\nFail-Closed"]
    end

    subgraph AI["🧠 AI Framework"]
        LG["LangGraph\ncreate_react_agent()"]
        LC["LangChain\nChatGoogleGenerativeAI"]
        GEMINI["Google Gemini API\n(External)"]
    end

    subgraph PERSIST["💾 Persistence"]
        CSV_USER["user_data.csv"]
        CSV_FALLBACK["ecommerce_cleaned.csv"]
        PKL_MODEL["best_model_weekly.pkl"]
        PKL_SCALER["scaler_weekly.pkl"]
        JSON_META["feature_columns_weekly.json"]
        SESSION["st.session_state"]
    end

    PAGE_UPLOAD --> SVC_DATA
    PAGE_DASH --> SVC_FC & SVC_VIZ
    PAGE_CHAT --> SVC_AGENT
    PAGE_SET --> SVC_AGENT
    NAV --> PAGE_UPLOAD & PAGE_DASH & PAGE_CHAT & PAGE_SET

    SVC_DATA --> CSV_USER & SESSION
    SVC_FC --> PKL_MODEL & PKL_SCALER & JSON_META
    SVC_AGENT --> LG

    LG --> LC & TOOL_REPL & TOOL_FC & TOOL_SCR
    LC --> GEMINI

    TOOL_REPL --> CSV_USER & CSV_FALLBACK
    TOOL_FC --> PKL_MODEL & PKL_SCALER
    TOOL_SCR -.->|HTTP| GEMINI

    style GEMINI fill:#f59e0b,color:#000
    style LG fill:#6366f1,color:#fff
    style TOOL_REPL fill:#22c55e,color:#fff
    style TOOL_FC fill:#22c55e,color:#fff
    style TOOL_SCR fill:#22c55e,color:#fff
```

---

## 14. Deployment Diagram

> **Prompt ke AI:** "Buatkan deployment diagram yang menampilkan deployment lokal dan cloud dari aplikasi Streamlit dengan semua dependencies eksternal"

```mermaid
graph TB
    subgraph USER_ENV["💻 USER ENVIRONMENT"]
        BROWSER["🌐 Web Browser\nChrome / Firefox\nlocalhost:8501 atau URL cloud"]
    end

    subgraph SERVER["🖥️ SERVER / LOCAL MACHINE"]
        subgraph STREAMLIT_PROC["Streamlit Process (Python 3.10+)"]
            APP["streamlit run streamlit_app.py\nPort: 8501"]
            CACHE["@st.cache_resource\nAgent Graph singleton\n@st.cache_data"]
        end

        subgraph FS["📁 File System"]
            F1["streamlit_app.py"]
            F2["agent_core.py"]
            F3["eval_agentic_ai.py"]
            F4["user_data.csv (runtime)"]
            F5["best_model_weekly.pkl 231KB"]
            F6["scaler_weekly.pkl"]
            F7["feature_columns_weekly.json"]
            F8["ecommerce_cleaned.csv 4.7MB"]
            F9["temp_plot.png (ephemeral)"]
        end

        subgraph SANDBOX["🏠 Sandbox Process (subprocess)"]
            SBX["Python -I _sandbox_script.py\nIsolated mode\ncwd: /tmp/agent_sandbox_xxx\nTimeout: 10 detik"]
        end
    end

    subgraph EXTERNAL["☁️ EXTERNAL SERVICES"]
        GAPI["Google AI Studio\ngenerativelanguage.googleapis.com\nGemini Models"]
        YAHOO["Yahoo Search\nsearch.yahoo.com"]
        DDG["DuckDuckGo HTML\nhtml.duckduckgo.com"]
    end

    subgraph CLOUD["🚀 CLOUD: Streamlit Community Cloud"]
        STCLOUD["streamlit.io/cloud\nAuto-deploy from GitHub"]
        SECRETS["st.secrets\nAPI Key storage aman"]
    end

    BROWSER <-->|HTTP/WebSocket| APP
    APP --> CACHE & F1 & F2 & F3
    APP -->|baca tulis| F4
    APP -->|baca| F5 & F6 & F7 & F8
    APP -->|generate hapus| F9
    APP -->|spawn| SBX
    SBX -->|baca read-only copy| F4
    APP <-->|HTTPS REST| GAPI
    APP <-->|HTTP| YAHOO & DDG
    STCLOUD -->|deploy| APP
    STCLOUD --> SECRETS

    style GAPI fill:#f59e0b,color:#000
    style YAHOO fill:#3b82f6,color:#fff
    style DDG fill:#e85d04,color:#fff
    style SBX fill:#dcfce7,stroke:#16a34a
    style STCLOUD fill:#6366f1,color:#fff
```

---

## 📋 RINGKASAN NARASI SISTEM

### Alur Utama Lengkap

```
[PEMILIK TOKO]
     │
     ▼ (1) Buka aplikasi → Input API Key → Upload file Shopee/Tokopedia
     │
[DATA PIPELINE]
     │ read_file() → clean_raw() → make_weekly() → user_data.csv
     │
     ├──► (2) DASHBOARD UI
     │         Tampilkan metrik, grafik tren, top produk terlaris
     │
     ├──► (3) FORECASTING ML
     │         Rolling Forecast → Analyst Insights:
     │         • Tren (naik/turun/stabil)
     │         • Deteksi periode puncak (Ramadan, Harbolnas)
     │         • Rekomendasi restocking & jadwal marketing
     │
     └──► (4) CHATBOT AI AGENT (LangGraph ReAct)
               │
               ├── Tanya analisis historis
               │     → Python_REPL (sandbox 3-layer) → Pandas → Grafik opsional
               │
               ├── Tanya prediksi
               │     → forecast_tool → Rolling ML Forecast (1-12 minggu)
               │
               └── Tanya harga kompetitor
                     → marketplace_scraper → Yahoo/DDG → URL wajib disertakan
                     → FAIL-CLOSED jika kosong (DILARANG mengarang harga!)
```

### Keamanan Python REPL — 3 Lapis Perlindungan

```
Kode Python dari LLM
     │
     ▼ [Lapisan 1] AST Static Analysis
     │  Whitelist modul: pandas, numpy, matplotlib, math, statistics, datetime, json, re
     │  Blacklist builtin: os, sys, subprocess, requests, eval, exec, open, __import__
     │  Blokir atribut dunder: sandbox-escape prevention
     │
     ▼ [Lapisan 2] Subprocess Isolation
     │  python -I (isolated mode, abaikan PYTHONPATH)
     │  cwd = direktori sandbox sementara di /tmp/
     │  Timeout 10 detik wall-clock
     │  Resource limits POSIX: CPU 10s, RAM 512MB, 16 proses, 64 file descriptor
     │
     ▼ [Lapisan 3] Read-Only Dataset
        user_data.csv disalin ke sandbox, chmod 0o444 (read-only)
        Data toko asli tidak dapat dimodifikasi oleh kode LLM
```

---

## 💡 PANDUAN PENGGUNAAN FILE INI

| Platform | Cara Pakai |
|----------|------------|
| **Mermaid.live** | Copy isi `````mermaid ... ````` → Paste ke [mermaid.live](https://mermaid.live) |
| **GitHub / GitLab** | Push file ini → Mermaid otomatis ter-render di README |
| **draw.io** | Import → From text → Pilih Mermaid |
| **VS Code** | Install ekstensi "Markdown Preview Mermaid Support" |
| **ChatGPT / Claude** | Copy section + prompt: *"Buatkan flowchart visual dari deskripsi ini"* |
| **Eraser.io** | Paste Mermaid → Auto render + bisa edit secara visual |
| **Lucidchart** | Import Mermaid diagram → Customize layout |
