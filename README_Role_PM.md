# Role: Project Manager (PM) + Product & Deployment Lead

## 🎯 Peran Inti
Mengubah semua komponen teknis yang dikembangkan oleh tim data dan AI menjadi satu produk akhir (*end-to-end product*) yang dapat digunakan (*usable*), stabil, dan memberikan nilai langsung kepada pengguna akhir (UMKM/penjual e-commerce).

---

## 📋 Tugas & Tanggung Jawab Detail

### 1. Menentukan Alur Utama Produk (*User Flow*)
Merancang pengalaman pengguna yang mulus dari awal hingga akhir:
- **Fase 1: Ingestion** -> Pengguna mengunggah dataset penjualan e-commerce (mendukung CSV/XLSX).
- **Fase 2: Preprocessing Transparan** -> Sistem melakukan pembersihan data di balik layar dan menampilkan ringkasan data.
- **Fase 3: Predictive Analytics** -> Pengguna melihat visualisasi hasil *forecasting* penjualan untuk 30-90 hari ke depan.
- **Fase 4: Conversational Insight** -> Pengguna berinteraksi dengan AI Chatbot untuk menggali *insight* lebih dalam atau membandingkan data dengan pasar eksternal.

### 2. Mendesain Struktur Aplikasi Streamlit
Bertanggung jawab atas tata letak antarmuka pengguna (UI/UX) pada *framework* Streamlit:
- **Halaman Upload Data (Landing Page)**: Area *dropzone* untuk file, dengan validasi format instan.
- **Dashboard Forecasting**: Panel metrik utama dan grafik *time-series* interaktif.
- **Halaman Chatbot AI**: Antarmuka percakapan (*chat interface*) yang responsif dengan riwayat pesan.

### 3. Mengatur Timeline & Manajemen Proyek
Memastikan tidak ada *bottleneck* antartim (DE -> DA -> MLE -> AIE -> PM):
- **Milestone 1**: Pipa data (*Data Pipeline*) siap dari Data Engineer.
- **Milestone 2**: Model Machine Learning disahkan oleh ML Engineer (target RMSE & R² tercapai).
- **Milestone 3**: Agent AI dari AI Engineer diintegrasikan dan stabil.
- **Milestone 4**: Finalisasi UI dan *User Acceptance Testing* (UAT).

### 4. Deployment & Infrastruktur
- Mengelola proses *deployment* aplikasi secara *live* ke **Streamlit Cloud**.
- Mengelola *environment variables* (API Keys) secara aman via `secrets.toml`.
- Memastikan dependensi di `requirements.txt` ter-kunci (pinned) untuk mencegah konflik versi di *production*.

### 5. Integrasi Sistem
Menggabungkan seluruh artefak teknis:
- Melakukan *load* model ML (`.pkl`) dan sistem *scaler* ke dalam memori aplikasi.
- Menyambungkan *logic* AI Agent ke *frontend* Streamlit menggunakan fungsi `st.chat_message`.

---

## 📦 Output Final
1. **Aplikasi Final**: Platform analitik *all-in-one* dengan UI/UX modern bertema gelap (*Obsidian mode*) yang terasa premium.
2. **Link Deployment Publik**: Tautan aplikasi yang bisa diakses oleh *stakeholders* atau publik secara langsung via internet.
3. **Dokumentasi Sistem**: Menjaga kejelasan repositori proyek untuk *handover* dan *scaling* di masa depan.
