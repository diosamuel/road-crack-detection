# 🛣️ Road Crack Detection System

Sistem deteksi keretakan jalan otomatis menggunakan GLCM Feature Extraction dan Deep Learning.

## 🌟 Fitur Utama

- **Upload & Analisis**: Upload gambar jalan dan dapatkan hasil analisis instan
- **GLCM Features**: Ekstraksi 5 fitur tekstur (Contrast, Dissimilarity, Homogeneity, Energy, Correlation)
- **Deep Learning**: Klasifikasi menggunakan Neural Network yang telah dioptimasi
- **Visualisasi**: Grafik dan tabel interaktif untuk memahami hasil
- **UI Menarik**: Interface yang user-friendly dan responsif

## 📋 Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)

## 🚀 Cara Instalasi

1. **Clone atau download repository ini**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Pastikan file model ada**

- `best_glcm_model.h5` - Model neural network yang sudah di-training
- `scaler_glcm.pkl` - Scaler untuk normalisasi fitur

## 💻 Cara Menjalankan

1. **Jalankan aplikasi Streamlit**

```bash
streamlit run app.py
```

2. **Buka browser**

- Aplikasi akan otomatis terbuka di browser
- Atau buka manual di: `http://localhost:8501`

3. **Gunakan aplikasi**

- Upload gambar jalan
- Klik "Analisis Gambar"
- Lihat hasil deteksi dan fitur yang diekstrak

## 📁 Struktur Folder

```
mayan/
├── app.py                      # Aplikasi Streamlit
├── GLCM_FEATURE_EXTRACTION.ipynb  # Notebook training
├── main.py                     # File testing
├── best_glcm_model.h5         # Model yang sudah di-training
├── scaler_glcm.pkl            # Scaler untuk normalisasi
├── requirements.txt           # Dependencies
└── README.md                  # Dokumentasi ini
```

## 🔬 Cara Kerja

1. **Input**: User upload gambar jalan
2. **Preprocessing**: Resize ke 128x128 dan konversi ke grayscale
3. **Feature Extraction**: Ekstraksi 5 fitur GLCM
4. **Normalization**: Normalisasi fitur menggunakan StandardScaler
5. **Classification**: Prediksi menggunakan Neural Network
6. **Output**: Hasil (Retak/Tidak Retak) + Confidence score + Visualisasi

## 📊 Fitur GLCM

- **Contrast**: Variasi intensitas lokal
- **Dissimilarity**: Perbedaan rata-rata intensitas
- **Homogeneity**: Keseragaman tekstur
- **Energy**: Keteraturan gambar
- **Correlation**: Ketergantungan linear pixel

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning
- **Scikit-image**: Image processing & GLCM
- **OpenCV**: Computer vision
- **Pandas**: Data manipulation
- **Matplotlib**: Visualisasi

## 📝 Tips Penggunaan

✅ **Gunakan gambar dengan:**

- Pencahayaan yang baik
- Fokus yang tajam
- Resolusi minimal 256x256 pixel
- Format JPG, JPEG, atau PNG

❌ **Hindari gambar:**

- Terlalu gelap atau blur
- Dengan banyak bayangan
- Resolusi sangat rendah
- Format file tidak didukung

## 🎯 Hasil Prediksi

- **Confidence ≥ 50%**: Retak terdeteksi ⚠️
- **Confidence < 50%**: Tidak ada retak ✅

## 🔧 Troubleshooting

**Error: Model not found**

- Pastikan file `best_glcm_model.h5` dan `scaler_glcm.pkl` ada di folder yang sama dengan `app.py`

**Error: Package not found**

- Jalankan: `pip install -r requirements.txt`

**Aplikasi tidak bisa diakses**

- Cek apakah port 8501 sudah digunakan
- Coba jalankan dengan port berbeda: `streamlit run app.py --server.port 8502`

## 📞 Dukungan

Jika ada pertanyaan atau masalah:

- Buat issue di repository
- Email: [your-email@example.com]

## 📄 Lisensi

[Tentukan lisensi yang sesuai]

## 👥 Kontributor

- [Nama Anda]

---

Dibuat dengan ❤️ menggunakan Streamlit dan TensorFlow
