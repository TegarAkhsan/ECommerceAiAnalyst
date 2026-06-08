# Dokumentasi Sistem: AI Agent Analisis E-commerce v3

Dokumen ini menjelaskan secara menyeluruh tentang apa yang telah Anda buat, bagaimana sistem berjalan, alur proses data, fungsi masing-masing menu, serta detail model *Machine Learning* dan AI Agent yang ada di dalam aplikasi `streamlit_app.py`.

---

## 1. Apa yang Telah Anda Buat?
Anda telah mengembangkan sebuah **Aplikasi Web Analitik E-commerce Berbasis AI**. Aplikasi ini bukan sekadar *dashboard* biasa, melainkan sebuah sistem cerdas terintegrasi yang menggabungkan:
1. **Data Pipeline Otomatis**: Kemampuan membersihkan dan memformat data mentah dari *marketplace* (khususnya Shopee) secara otomatis.
2. **Machine Learning Forecasting**: Kemampuan memprediksi (*forecasting*) tren omzet/pendapatan di masa depan menggunakan model XGBoost.
3. **AI Assistant (Chatbot Agent)**: Asisten virtual cerdas berarsitektur *ReAct (Reasoning + Acting)* yang bisa melakukan riset kompetitor di internet dan mengeksekusi kode Python untuk menganalisis data Anda secara instan.

Tampilan antarmuka (*UI*) dibangun menggunakan **Streamlit** dengan modifikasi CSS yang sangat modern (tema *Obsidian dark mode*, *glassmorphism*, dan animasi interaktif) sehingga terasa premium dan profesional.

---

## 2. Alur Proses Data (*Data Flow*)
Proses data berjalan dari hulu (upload) hingga hilir (prediksi & tanya jawab AI) dengan alur sebagai berikut:

1. **Input Data**: User mengunggah satu atau beberapa file laporan penjualan (`.csv` atau `.xlsx`).
2. **Pembacaan & Deteksi Encoding (`read_file`)**: Sistem secara cerdas mencoba membaca file dengan berbagai format encoding (utf-8, latin-1, cp1252) untuk menghindari error karakter khusus.
3. **Pembersihan Data (`clean_raw`)**:
   - Menyamakan format nama kolom (contoh: 'Waktu Pengiriman Diatur' menjadi 'Waktu Pesanan Dibuat').
   - Memperbaiki format nominal angka (menghilangkan titik ribuan dan mengubah koma menjadi titik desimal).
   - Memfilter hanya pesanan dengan **Status "Selesai"**.
   - Mengisi nilai kosong (*missing values*) pada kolom-kolom seperti diskon, alasan pembatalan, dan *returned quantity*.
   - Mengubah kolom tanggal menjadi format *datetime* standar.
4. **Agregasi Harian (`make_daily`)**: Data transaksi yang sudah bersih kemudian di-*grouping* berdasarkan hari (`order_date`) untuk mendapatkan total pendapatan (`total_revenue`) per hari. Hari dimana tidak ada penjualan diisi dengan angka 0.
5. **Penyimpanan State**: Data bersih disimpan sementara di *memory* (`st.session_state`) dan diekspor ke `user_data.csv` agar dapat diakses oleh AI Agent.

---

## 3. Fungsi Setiap Menu pada Aplikasi

### A. Upload Data
* **Fungsi**: Pintu masuk utama sistem. Tempat user mengunggah file riwayat penjualan.
* **Proses yang berjalan**: 
  - Melakukan *cleaning* dan agregasi harian seperti yang dijelaskan pada Alur Proses Data.
  - Menampilkan metrik ringkasan performa: Total Transaksi, Total Revenue, Jumlah Produk Unik, dan Rentang Tanggal.
  - Membuat visualisasi menggunakan library **Plotly**: Grafik tren *revenue* harian (Area Chart) dan Top 10 Produk Terlaris berdasarkan pendapatan (Bar Chart horizontal).

### B. Dashboard Forecasting
* **Fungsi**: Menyajikan prediksi (ramalan) pendapatan untuk 30, 60, atau 90 hari ke depan.
* **Proses yang berjalan**:
  - Sistem memuat model *Machine Learning* yang sudah di-*train* sebelumnya (`model_xgb_tuned.pkl`), beserta *scaler* (`scaler.pkl`) dan meta-fiturnya (`feature_columns.json`).
  - Fungsi `run_forecast` melakukan **Rolling Forecast**: Hari esok diprediksi menggunakan data hari ini. Prediksi lusa menggunakan prediksi hari esok sebagai lag (historis). 
  - *Imputasi Cerdas*: Jika ada hari libur (revenue 0), sistem tidak menggunakannya sebagai lag, melainkan di-*impute* dengan nilai median agar prediksi tidak jatuh ke 0 pada bulan berikutnya.
  - Menampilkan metrik hasil prediksi, performa error (RMSE, R², MAPE), serta grafik interaktif batas estimasi (menampilkan rentang ±42% batas atas dan bawah dari nilai rata-rata).

### C. Chatbot AI Agent
* **Fungsi**: Asisten cerdas (*copilot*) tempat user bisa bertanya apa pun tentang data mereka atau tentang tren pasar eksternal.
* **Proses yang berjalan**:
  - Menggunakan **LangChain** dengan arsitektur **ReAct (Reasoning and Acting)**.
  - **LLM Engine**: Memanfaatkan model dari **Google Gemini** (Gemini 1.5 Flash / 2.0 Flash dsb).
  - **Sistem Routing Pintar**: AI diberikan dua *tools*:
    1. **Python REPL Tool**: Jika user meminta grafik, rata-rata penjualan, atau analisis data CSV. AI akan otomatis menulis skrip Python dengan `pandas` dan `matplotlib`, mengeksekusinya di *background*, menyimpan grafik sebagai `temp_plot.png`, dan aplikasi Streamlit akan langsung menampilkan gambarnya.
    2. **Tavily Web Search Tool**: Jika user menanyakan tentang "strategi promosi", "harga kompetitor di Shopee", "tren 2025", AI akan mencari informasi aktual di internet.
  - Fitur anti-error (*auto-retry*): Jika API Gemini mengalami isu pemanggilan fungsi (*Malformed Function Call*), sistem memiliki mekanisme *fallback* untuk tetap merespons dengan mode terbatas, memberikan panduan kepada user.

---

## 4. Model Machine Learning dan Fiturnya

Sistem menggunakan model **XGBoost Regressor** yang telah di-*tuning* (*hyperparameter tuning*). Model ini dilatih untuk memprediksi `total_revenue` harian.

### Fitur-Fitur (Features) yang Dibangun (`build_features_daily`):
Sistem mengubah data deret waktu (*time-series*) sederhana menjadi format *supervised learning* dengan ekstraksi fitur (*feature engineering*) ekstensif:
1. **Time Features**: `year`, `month`, `day`, `dayofweek`, `quarter`, `week_of_year`, `day_index` (hari ke-N sejak data dimulai).
2. **Categorical / Event Features**: Flag biner (1/0) untuk mendeteksi `is_weekend` (akhir pekan), `is_month_start` (awal bulan / gajian), `is_month_end`, `is_ramadan` (efek musiman), `is_harbolnas` (tanggal kembar 10.10, 11.11, 12.12), dan `is_holiday_gap`.
3. **Cyclical Features (Trigonometri)**: `sin_week`, `cos_week`, `sin_month`, `cos_month`, `sin_year`, `cos_year`. Digunakan agar model memahami bahwa hari Minggu (6) dan Senin (0) itu berdekatan siklusnya, begitu juga bulan Desember (12) dan Januari (1).
4. **Lag & Rolling Features**: *Lag* (nilai hari sebelumnya) adalah fitur paling penting untuk *forecasting*.
   - `lag_1` (omzet H-1)
   - `lag_7` (omzet minggu lalu di hari yang sama)
   - `lag_30` (omzet bulan lalu di hari yang sama)
   - `rolling_7d_mean` & `rolling_30d_mean` (rata-rata pergerakan omzet).

### Karakteristik & Keterbatasan Model
- Model memprediksi dalam skala logaritmik, sehingga hasil prediksi dikembalikan menggunakan `np.expm1(predict)`.
- Karena ini model berbasis *Lag*, grafik prediksi masa depan cenderung membentuk pola garis yang lebih "stabil" menuju titik rata-rata (baseline), karena variasi fluktuatif ekstrem tidak bisa diprediksi secara pasti. Model sangat baik untuk mengestimasi rata-rata/baseline masa depan.

---

## Kesimpulan Arsitektur `streamlit_app.py`
File `streamlit_app.py` bertindak sebagai pusat kendali *(orchestrator)* seluruh aplikasi:
1. **Baris 1-450**: Import library dan setup UI/CSS kustom yang sangat detail.
2. **Baris 450-650**: Fungsi *backend* data science (`clean_raw`, `make_daily`, `build_features_daily`, `run_forecast`).
3. **Baris 660-820**: Setup LangChain ReAct Agent beserta *system prompt* untuk mendikte cara AI merespons.
4. **Baris 820-950**: Layar pembuka / Onboarding presentasi awal aplikasi.
5. **Baris 960+**: Logika *routing* halaman navigasi Streamlit (Sidebar) dan implementasi tampilan (UI/UX) untuk ketiga menu utama.

Proyek ini telah menerapkan *best practices* dalam mengintegrasikan *Data Engineering*, *Machine Learning*, *Generative AI*, dan *Frontend Engineering* dalam satu kesatuan sistem yang harmonis.
