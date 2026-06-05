# 🧠 Student Lifestyle Risk & Productivity Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-lifestyle-dashboard.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Dashboard interaktif untuk memprediksi tingkat stres mahasiswa berdasarkan gaya hidup, performa akademik, dan faktor psikologis.

---

## 📖 Deskripsi Singkat Proyek

**Student Lifestyle Risk & Productivity Monitoring System** adalah dashboard berbasis machine learning yang memprediksi tingkat stres mahasiswa (Low/Medium/High) berdasarkan faktor gaya hidup seperti kualitas tidur, beban belajar, tingkat kecemasan, dan depresi. Dengan akurasi **87.73%**, sistem ini membantu mahasiswa dan dosen melakukan deteksi dini risiko stres akademik serta memberikan rekomendasi penanganan yang tepat.

**Fungsi/Tujuan:**
- Mengidentifikasi faktor paling berpengaruh terhadap stres mahasiswa
- Memprediksi tingkat stres secara objektif berbasis data
- Menyediakan insight dan rekomendasi untuk pencegahan

**Kelebihan:**
- Gratis dan mudah diakses via web
- Visualisasi interaktif dengan Plotly
- Insight otomatis dari data yang difilter

**Inovasi:**
- Sistem terpadu (prediksi + segmentasi perilaku + visualisasi)
- Model ML dengan akurasi tinggi (87.73%)
- Dashboard yang dapat digunakan tanpa pelatihan teknis

---

## 👥 Tim Pengembang

| Nama | Peran |
|------|-------|
| Sri Hartati Setia Ningrum | Dashboard, Visualisasi, Insight |
| Rafli Maulana | Preprocessing, Modeling, Evaluasi |
| Iman Rasyid Sayuti | Dokumentasi, Laporan, Presentasi |

**Tim Capstone:** CC26-PRU475 | **Tema:** Healthy Lives & Well-being

---

## 🛠 Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| Bahasa Pemrograman | Python 3.12 |
| Framework Dashboard | Streamlit |
| Machine Learning | Scikit-learn (HistGradientBoostingClassifier) |
| Visualisasi | Plotly, Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---


---

## 🔧 Petunjuk Penggunaan Aplikasi

### Prasyarat
- Python 3.12 atau lebih baru
- Git (opsional)
- Minimal RAM 4GB

### Langkah-langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/username/student-lifestyle-dashboard.git
cd student-lifestyle-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pastikan Anda berada di direktori root proyek
cd student-lifestyle-dashboard

# 4. Jalankan dashboard Streamlit
streamlit run dashboard.py

---
