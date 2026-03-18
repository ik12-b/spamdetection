# Spam Detection AI with FastAPI & XGBoost

Proyek ini adalah sistem deteksi pesan spam otomatis menggunakan model **XGBClassifier** dan engine **FastAPI** untuk antarmuka web yang modern dan cepat.

---

## 🛠️ Cara Kerja (How It Works)

Aplikasi ini bekerja melalui beberapa tahap penting yang didefinisikan dalam notebook eksperimen:

1.  **Preprocessing Teks**: 
    - Dataset asli (`spam.csv`) diproses untuk membersihkan kata-kata yang tidak perlu (*stopwords*) menggunakan library NLTK.
    - Teks dikonversi menjadi format string yang konsisten sebelum masuk ke tahap vektorisasi.
2.  **Vektorisasi (TF-IDF)**:
    - Pesan teks diubah menjadi representasi matematika menggunakan `TfidfVectorizer`. Ini membantu model memahami bobot signifikansi setiap kata dalam konteks spam vs ham.
3.  **Klasifikasi Model (XGBoost)**:
    - Model menggunakan algoritma **XGBClassifier** (Extreme Gradient Boosting).
    - Model telah dilatih untuk membedakan pola bahasa dalam pesan spam (seperti penawaran hadiah, klik tautan, dll.) dibandingkan pesan asli (*ham*).
    - **Catatan Versi**: Model ini dilatih pada XGBoost versi terbaru (3.x). Jika Anda mengalami kesalahan saat memuat model di Google Colab, pastikan untuk mengupdate library dengan `pip install -g xgboost`.

---

## 🚀 Cara Penggunaan (How to Use)

Aplikasi web ini sangat mudah dijalankan di Windows:

### 1. Persiapan
Pastikan Anda memiliki Python terinstal di sistem Anda.

### 2. Instalasi Dependensi
Buka terminal/Command Prompt di folder `web-app` dan jalankan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Cukup klik dua kali pada file **`run.bat`** yang ada di folder `web-app`. Skrip ini akan otomatis memulai server FastAPI.

### 4. Mengakses Web
Buka browser favorit Anda (Chrome, Edge, dll.) dan ketik alamat berikut:
```
http://localhost:8000
```

---

## 🖥️ Antarmuka (Interface)

Aplikasi web menggunakan desain **Premium Glassmorphism**:
- **Textarea**: Masukkan atau tempel pesan yang ingin Anda periksa.
- **Tombol "Check for Spam"**: Klik untuk menganalisis pesan secara real-time.
- **Hasil Visual**: Jika pesan terdeteksi sebagai **SPAM**, box akan berwarna merah. Jika **ASLI (HAM)**, box akan berwarna hijau.
- **Riwayat Analisis**: Tabel riwayat di bawah tombol utama menyimpan catatan 5 analisis terakhir Anda.

---

## 📁 Struktur Folder
```text
/
├── SpamPrediction.ipynb (Notebook Pelatihan)
├── SpamInference.ipynb (Notebook Pengujian)
├── spam.csv (Dataset)
└── web-app/ (Aplikasi Web FastAPI)
    ├── main.py (Kode Server & Frontend)
    ├── requirements.txt (Daftar Library)
    ├── run.bat (Skrip Start Windows)
    ├── spamprediction.pkl (Model Terlatih)
    └── vectorizer.pkl (Vectorizer TF-IDF)
```

---
*Dibuat oleh [User Name] sebagai proyek latihan AI kedua.*
