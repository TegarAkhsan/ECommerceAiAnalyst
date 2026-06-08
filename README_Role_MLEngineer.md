# Role: Machine Learning Engineer (Modeling & Experiment Lead)

## 🎯 Peran Inti
Berperan sebagai "otak prediktif" dalam sistem. Bertanggung jawab mencari, melatih, mengevaluasi, dan memilih model algoritma terbaik yang mampu membaca pola data historis penjualan untuk memprediksi omzet (*forecasting*) di masa depan secara akurat.

---

## 📋 Tugas & Tanggung Jawab Detail

### 1. Eksperimen Algoritma (Modeling)
Mengimplementasikan berbagai variasi model pembelajaran mesin, mulai dari yang sederhana hingga *ensemble learning* yang kompleks:
- **Linear Regression**: Digunakan sebagai *baseline model* untuk mengukur seberapa baik hubungan linier standar antar variabel.
- **Random Forest Regressor**: Menggunakan pendekatan *bagging* untuk menangani relasi non-linier dan mengurangi *variance* dari fitur.
- **XGBoost Regressor**: Menggunakan pendekatan *gradient boosting* tingkat lanjut untuk mendominasi performa metrik, khusus untuk menangani relasi kompleks pada *time-series tabular*.

### 2. Skenario *Training* & *Hyperparameter Tuning*
Melakukan pengujian ketat untuk mencari performa optimal:
- **Skenario Default**: Melatih ketiga model di atas dengan parameter standar bawaan pustaka (*out-of-the-box*).
- **Skenario Grid Search CV**: Menerapkan validasi silang lima lipatan (5-Fold Cross Validation) untuk mencegah *overfitting*.
- **Tuning**: Melakukan iterasi *hyperparameter tuning* (seperti `learning_rate`, `max_depth`, `n_estimators` pada XGBoost) dengan merujuk pada *best-practices* literatur akademis/industri.

### 3. Evaluasi Metrik
Model tidak dipilih secara asal. Mengukur keakuratan model dengan standar industri secara statistik:
- **RMSE (Root Mean Square Error)**: Menjadi metrik **Utama**. Memberikan penalti besar untuk eror (prediksi meleset jauh).
- **R² (R-Squared)**: Mengukur seberapa baik fitur menjelaskan varians dalam target penjualan (makin mendekati 1 makin baik).
- **MAE (Mean Absolute Error)**: Rata-rata absolut selisih nilai prediksi dengan nilai aktual.
- **MAPE (Mean Absolute Percentage Error)**: Kesalahan persentase (cocok untuk komunikasi ke tim bisnis).

### 4. Seleksi & Ekspor Model
- Menentukan pemenang eksperimen (*Champion Model*) berdasarkan kombinasi **RMSE Terendah** dan **R² Tertinggi** pada himpunan data *test*.
- Menyimpan (*serialize*) model terbaik ke dalam format *pickle* (`.pkl`) agar ringan, portabel, dan siap dimuat ke aplikasi *production*.

---

## 📦 Output Final
1. **Model Prediktif Siap Deploy (`model_xgb_tuned.pkl`)**: Algoritma yang telah "belajar" pola penjualan masa lalu dan mampu meramal pendapatan bulan depan.
2. **Laporan Komparasi (*Benchmark*)**: Tabel perbandingan (matriks evaluasi) komprehensif antara Linear Regression, Random Forest, dan XGBoost sebelum dan sesudah *tuning*.
3. **Dokumentasi Eksperimen**: Catatan parameter mana yang menghasilkan performa maksimal untuk panduan perbaikan (re-training) bulan depan.
