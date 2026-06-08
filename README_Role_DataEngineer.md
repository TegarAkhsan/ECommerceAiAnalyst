# Role: Data Engineer (Data Pipeline Owner)

## 🎯 Peran Inti
Bertanggung jawab penuh dari hulu ke hilir terkait pemrosesan data. Mengubah data mentah (*raw data*) yang berantakan dari berbagai sumber (*multi-format export* e-commerce) menjadi data bersih, valid, berformat standar, dan siap konsumsi oleh Machine Learning Engineer dan Data Analyst.

---

## 📋 Tugas & Tanggung Jawab Detail

### 1. Data Ingestion & Loading
- Menerima dan memuat dataset riwayat pesanan berskala besar (20.000+ baris data transaksi e-commerce).
- Mengatasi isu *encoding* yang sering terjadi pada file ekspor lokal (menangani `utf-8`, `latin-1`, `cp1252`).

### 2. Data Cleaning & Transformation
Memastikan integritas data sebelum masuk ke tahap analitik:
- **Penanganan Missing Value**: Mengimputasi data kosong pada kolom kritis (misal: mengisi NaN dengan 0 untuk Diskon, atau "Tidak Ada" untuk alasan pembatalan).
- **Penanganan Outlier**: Mendeteksi anomali pada kolom transaksi (misal, pesanan fiktif dengan nominal tidak wajar).
- **Standardisasi Tipe Data**: Mem-parsing format teks nominal mata uang ("35.663") menjadi *float* murni, dan mengekstrak nominal berat ("600 gr") menjadi *numeric*.
- **Filtering**: Membuang transaksi yang gagal atau dibatalkan, hanya memfokuskan pada `Status Pesanan == "Selesai"`.

### 3. Feature Engineering & Preprocessing
Mentransformasi data mentah menjadi bentuk matematika yang siap dicerna algoritma:
- **Encoding**: Menerapkan *One-Hot Encoding* untuk fitur kategorikal (seperti status promo, metode pembayaran, atau *flag* hari libur).
- **Ekstraksi Fitur Waktu (*Time-Series*)**: Mengekstrak komponen tanggal menjadi fitur `hari`, `bulan`, `tahun`, `kuartal`, `siklus sinus/cosinus` (untuk menangkap pola perulangan).
- **Scaling**: Mengaplikasikan `StandardScaler` (atau *MinMaxScaler*) untuk menyelaraskan rentang variabel numerik agar model konvergen lebih cepat tanpa *bias* skala.

### 4. Data Splitting & Balancing
Menyiapkan kerangka validasi model:
- Memecah dataset bersih menjadi **70% Data Latih (Train)** dan **30% Data Uji (Test)** berbasis kronologi (*time-series split*) agar model tidak mengintip masa depan (*data leakage*).
- **Cek Distribusi Target**: Memastikan apakah distribusi kelas/target seimbang. Jika terdapat imbalansi ekstrem pada regresi/klasifikasi, mengaplikasikan metode sintesis seperti **SMOGN** (*Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise*) (Opsional).

---

## 📦 Output Final
1. **Dataset Bersih (Cleaned Dataset)**: File data yang siap dilahap oleh model tanpa ada *error* atau nilai nol.
2. **Artefak Preprocessing**: Objek *Scaler* (`scaler.pkl`) yang tersimpan rapi untuk digunakan kembali pada data baru saat fase *inference* di aplikasi.
3. **Pipeline Kode Otomatis**: *Script* Python (seperti fungsi `clean_raw()` dan `build_features()`) yang dapat dipanggil kapan saja saat user mengunggah file baru.
