# Role: AI Engineer (LangChain Agent Builder)

## 🎯 Peran Inti
Menciptakan "pembeda utama" (*unique selling proposition*) proyek ini dibandingkan *dashboard* biasa. AI Engineer bertugas merancang dan mengintegrasikan **AI Agent otonom** yang mampu memahami konteks bisnis, mengeksekusi analisis data mentah, serta mencari informasi eksternal layaknya asisten manusia profesional.

---

## 📋 Tugas & Tanggung Jawab Detail

### 1. Arsitektur Agen dengan Pola ReAct
Membangun Agen berarsitektur **ReAct (Reasoning and Acting)** menggunakan *framework* LangChain (atau LangGraph).
- **Reasoning**: Agen dilatih untuk memahami pertanyaan pengguna, memecahkannya menjadi tahapan logis.
- **Acting**: Agen menentukan secara sadar "alat" apa yang harus dipakai berdasarkan kesimpulan rasionalnya.

### 2. Integrasi LLM (Large Language Model)
- Menghubungkan sistem dengan otak AI mutakhir, menggunakan **LLM Gemini** (atau GPT/Llama).
- Menulis *System Prompt* kompleks untuk memberi persona kepada agen: "AI Data Analyst E-commerce Indonesia" dengan panduan aturan ketat (*guardrails*).

### 3. Setup Tools & Kapabilitas
Memberikan "Tangan dan Mata" pada AI melalui integrasi spesifik:
- **Python REPL Tool**: Memberikan agen kemampuan membuat, memvalidasi, dan menjalankan kode Python (`pandas`, `matplotlib`) secara dinamis terhadap file CSV. Ini memungkinkan AI menjawab pertanyaan ad-hoc dengan men-generate grafik/plot atau menghitung statistik otomatis.
- **Web Search Tool (Tavily/SerpAPI)**: Menyambungkan agen dengan dunia luar agar mampu mencari harga kompetitor secara *real-time*, tren pasar lokal, atau panduan pemasaran dari Google.

### 4. Merancang Logika *Routing* (Pendelegasian)
Mengembangkan logika *decision-making* yang presisi:
- Jika *user* bertanya: *"Tolong tampilkan grafik 5 produk terlaris dari data saya"*, Agen secara logis **memilih Python REPL** -> menulis kode -> membuat grafik plot.
- Jika *user* bertanya: *"Bagaimana strategi promosi celana panjang di Shopee bulan ini?"*, Agen **memilih Web Search** -> membaca web eksternal -> menyusun laporan.

### 5. Integrasi Frontend (Streamlit)
Menjembatani *engine* LangChain yang berjalan di *backend* dengan *chat interface* UI Streamlit agar interaksi berjalan *real-time*, *streaming*, responsif, dan mampu me-*render* grafik.

---

## 📦 Output Final
1. **Sistem Chatbot Interaktif**: Asisten virtual yang tertanam langsung di aplikasi pengguna.
2. **Automated Analytics Engine**: Kemampuan menganalisis tabel data hingga mencetak visualisasi tanpa pengguna perlu mengetahui pemrograman.
3. **External Market Researcher**: Kemampuan meriset kondisi persaingan e-commerce di luar platform klien secara aktual.
