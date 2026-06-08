# AI Agent E-commerce Analytics - Data Analyst & ML Documentation

## 1. Pendahuluan
Proyek ini adalah sebuah sistem analitik cerdas (*intelligent analytics system*) berbasis web (Streamlit) yang dirancang khusus untuk membedah data e-commerce (fokus pada platform seperti Shopee). Dari sudut pandang Data Analyst dan Data Scientist, proyek ini menggabungkan proses **ETL (Extract, Transform, Load)** otomatis, **Time-Series Forecasting** menggunakan Machine Learning, serta asisten **Generative AI** berbasis ReAct Agent untuk *Ad-hoc Analysis*.

Tujuan utama dari sistem ini adalah mengubah data mentah (*raw data*) ekspor penjualan menjadi *actionable insights* secara otonom tanpa perlu intervensi manual dari pengguna (penjual/UMKM).

---

## 2. Arsitektur Data & Pipeline (ETL)
Pipeline data dirancang untuk menerima data sekotor apapun dari hasil ekspor platform e-commerce dan memprosesnya hingga siap masuk ke model ML atau dianalisis oleh AI.

### A. Data Ingestion
Sistem mendukung *multi-file upload* dengan format `.csv` dan `.xlsx`.
- **Intelligent Encoding Reading**: File CSV dari Shopee seringkali mengalami isu *encoding*. Sistem mencoba `utf-8`, `latin-1`, hingga `cp1252` secara iteratif (pada fungsi `read_file()`) agar karakter unik atau format lama tetap terbaca tanpa `UnicodeDecodeError`.

### B. Data Cleaning (`clean_raw()`)
Pada fase ini, noise pada data dibersihkan agar metrik keuangan menjadi akurat:
1. **Normalisasi Kolom**: Penyelarasan skema data (misalnya, `Waktu Pengiriman Diatur` disamakan menjadi `Waktu Pesanan Dibuat`).
2. **Parsing Tipe Data Numerik**: 
   - Kolom finansial (`Total Pembayaran`) sering terbaca sebagai string ("35.663"). Sistem otomatis membuang titik separator ribuan dan mengganti koma menjadi titik desimal.
   - Kolom berat (`Total Berat`) yang berakhiran "gr" diekstrak murni nilai numeriknya menggunakan Regular Expression (RegEx).
3. **Filtering Kondisional**: Model analitik pendapatan HANYA memperhitungkan *Completed Revenue*. Oleh karena itu, sistem mem-filter berdasarkan `Status Pesanan` yang berawalan "Pesanan diterima" atau sudah diselesaikan ("Selesai").
4. **Imputasi Missing Values (NaN)**: Mengisi kolom non-kritis dengan *default value* (misal: Diskon = 0, Alasan Pembatalan = "Tidak Ada").
5. **Konversi Tipe Waktu**: Mengubah format tanggal mentah menjadi objek `datetime` di pandas dan mengekstrak tanggal dasarnya (*normalized order date*).

### C. Data Aggregation (`make_daily()`)
Data pesanan yang telah dibersihkan (tingkat *order/item*) dikelompokkan (*groupby*) menjadi deret waktu tingkat harian (`daily frequency`).
- **Resampling & Reindexing**: Jika dalam historis terdapat hari dimana toko tidak beroperasi (tidak ada data penjualan), kalender data direkonstruksi (`pd.date_range`) dan hari kosong tersebut diisi (*impute*) dengan *revenue* bernilai 0. Hal ini menjamin konsistensi jarak baris pada *time-series*.

---

## 3. Feature Engineering untuk Model Forecasting
Kunci akurasi dari model *Machine Learning* dalam proyek ini terletak pada fungsi `build_features_daily()`. Deret waktu univariat (hanya tanggal dan *revenue*) ditransformasi secara ekstensif menjadi data *tabular* multivariat (*supervised learning* format).

Sistem menghasilkan **4 kelompok fitur utama**:

1. **Temporal Features (Fitur Waktu Dasar)**
   - `year`, `month`, `day`, `dayofweek`, `quarter`, `week_of_year`, `day_index` (indeks hari sejak hari pertama *training*).
   
2. **Event & Categorical Flags (Biner 1/0)**
   - Mendeteksi perilaku diskrit: `is_weekend`, `is_month_start` (menangkap efek siklus gajian/payday), `is_month_end`.
   - **Fitur Spesifik Indonesia/E-commerce**: `is_ramadan` (tanggal-tanggal ramadan tahun 2024-2026 yang biasanya omzet naik tajam), dan `is_harbolnas` (promo *double date* 10.10, 11.11, 12.12).
   - `is_holiday_gap`: Penanda saat data asli bernilai 0 (toko tutup/tidak ada penjualan).

3. **Cyclical / Trigonometric Features**
   - Waktu bersifat siklik. Model *tree-based* seperti XGBoost kesulitan memahami bahwa bulan Desember (12) dan Januari (1) itu berdekatan jaraknya. 
   - *Solusi*: Fitur waktu dikodekan menjadi representasi Sinus dan Cosinus (`sin_week`, `cos_week`, `sin_month`, `cos_month`, `sin_year`, `cos_year`).

4. **Autoregressive (Lag) & Rolling Features**
   - **Lags**: `lag_1` (H-1), `lag_7` (Hari yang sama minggu lalu), `lag_30` (Hari yang sama bulan lalu).
   - **Rolling Means**: Rata-rata pergerakan `rolling_7d_mean` dan `rolling_30d_mean` untuk meredam *noise* harian dan menangkap tren fundamental (dihitung *shifted* 1 hari ke belakang agar tidak terjadi *data leakage*).
   - **Smart Imputation for Lags**: Jika hari-hari sebelumnya toko tutup (*revenue* 0), *lag* di-*impute* menggunakan nilai *median non-zero* dari keseluruhan historis. Jika dipertahankan 0, saat memprediksi bulan depan (*rolling forecast*), nilainya bisa kolaps ke 0 secara beruntun (*death spiral*).

---

## 4. Machine Learning Model (XGBoost)
Pusat dari fitur *Forecasting* adalah pemodelan prediktif berbasis *Gradient Boosting Tree*.

- **Algoritma**: `XGBoost Regressor` karena kemampuannya menangani hubungan non-linear pada data terstruktur dan ketahanannya terhadap *outliers*.
- **Target Variable**: Target penjualan (`total_revenue`) kemungkinan memiliki distribusi *skewed* (*Right-skewed* dengan beberapa *outlier* pendapatan masif). Prediksi dilakukan pada ranah *logarithmic* atau menggunakan metrik deviasi yang diratakan, di mana hasil prediksi dieksponensialkan kembali menggunakan `np.expm1()`.
- **Scaling**: Fitur numerik diskalakan menggunakan `StandardScaler` / `MinMaxScaler` yang disimpan pada `scaler.pkl` agar distribusi fitur input saat *inference* sama dengan saat *training*.
- **Rolling Inference (`run_forecast()`)**:
  Untuk memprediksi 30 hingga 90 hari ke depan, metode *recursive/rolling forecasting* digunakan. Model memprediksi Hari ke-1, nilai prediksinya dimasukkan kembali sebagai histori (menjadi *lag* 1) untuk memprediksi Hari ke-2, begitu seterusnya.

---

## 5. Integrasi LLM AI Agent (Langchain ReAct)
Bukan hanya model *predictive*, sistem dilengkapi *Agentic AI* untuk memfasilitasi analisis eksploratif spontan (Eksplorasi Data *Conversational*).

- **Architecture**: Menggunakan *framework* **LangGraph (ReAct - Reasoning and Acting)** yang dikendalikan oleh LLM Google Gemini (lewat `langchain-google-genai`). ReAct memungkinkan agen untuk *"Berpikir (Thought) -> Memilih Tool (Action) -> Mengamati Hasil (Observation) -> Menyimpulkan"*.
- **Tools Integrations**:
  1. **Python REPL Tool (`PythonREPLTool`)**: Agen dapat menulis dan mengeksekusi *script* Python (Pandas & Matplotlib) secara sekuensial *di belakang layar*. Jika pengguna meminta *"Buatkan grafik produk terlaris"*, AI akan membaca `user_data.csv`, memanipulasi kolom (Groupby Product), dan mem-*plot* grafik untuk dirender oleh Streamlit (`temp_plot.png`).
  2. **Tavily Web Search Tool**: Agen dapat melakukan pencarian *real-time* ke internet untuk membandingkan harga dengan kompetitor eksternal atau mencari tren e-commerce terkini.
- **Smart Routing**: Melalui *System Prompt* yang ketat, LLM diarahkan secara deterministik untuk mengetahui kapan harus menganalisis data CSV lokal (Python REPL) dan kapan harus meriset web (Tavily).

---

## 6. Persyaratan dan Instalasi untuk Environment Data Science

Untuk para *Data Analyst/Scientist* yang ingin mengeksplorasi lebih lanjut (misal, retrain model, mengubah metode lag, atau menambah fitur Regressor eksogen), instalasi *environment* sangat sederhana:

```bash
# 1. Klon repositori / masuk ke folder project
cd Proyekstupen

# 2. Buat Virtual Environment (opsional tapi direkomendasikan)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Instal semua dependencies khusus Data Science
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit lokal
streamlit run streamlit_app.py
```

### Panduan Retraining (Pengembangan Lebih Lanjut)
Jika model perlu dilatih ulang menggunakan data baru:
1. Pastikan Anda memiliki *pipeline* terpisah untuk `fit` `XGBoost`.
2. Simpan objek model ke `model_xgb_tuned.pkl`.
3. Simpan objek scaler ke `scaler.pkl`.
4. Perbarui list *feature columns* ke dalam array/JSON `feature_columns.json`.
5. Pastikan urutan fitur pada `build_features_daily()` sama persis dengan urutan array saat model di-fit untuk menghindari *feature names mismatch error*.
