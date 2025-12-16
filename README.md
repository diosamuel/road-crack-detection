# 🛣️ Road Crack Detection System

<div align="center">
  <p><strong>Sistem deteksi keretakan jalan otomatis menggunakan GLCM Feature Extraction dan Machine Learning</strong></p>
  <p>Aplikasi web berbasis Streamlit untuk mendeteksi keretakan pada permukaan jalan melalui analisis tekstur gambar</p>
</div>

---

## 📖 Tentang Project

**Road Crack Detection System** adalah aplikasi berbasis web yang dirancang untuk membantu mengidentifikasi keretakan pada permukaan jalan secara otomatis. Sistem ini menggunakan pendekatan Computer Vision dengan **GLCM (Gray-Level Co-occurrence Matrix)** untuk ekstraksi fitur tekstur, dikombinasikan dengan **Neural Network** untuk klasifikasi.

### 🎯 Tujuan Project

- Memudahkan deteksi dini kerusakan jalan dengan teknologi
- Mengurangi biaya inspeksi manual yang tinggi
- Memberikan analisis kuantitatif kondisi jalan
- Membantu perencanaan pemeliharaan infrastruktur jalan

### 💡 Keunggulan

- ✅ **User-friendly**: Interface yang mudah digunakan
- ✅ **Real-time**: Hasil analisis instan
- ✅ **Akurat**: Menggunakan 5 fitur GLCM untuk analisis mendalam
- ✅ **Visualisasi**: Grafik dan tabel yang informatif
- ✅ **Open Source**: Kode terbuka untuk pembelajaran dan pengembangan

## 🔗 Credits

Project ini dikembangkan berdasarkan repository:

- **Original Repository**: [diosamuel/road-crack-detection](https://github.com/diosamuel/road-crack-detection)
- **Developer**: [diosamuel](https://github.com/diosamuel)
- **Reference Paper**: Menggunakan metodologi GLCM untuk analisis tekstur gambar

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
tubes_dl/
├── app.py                      # Aplikasi Streamlit utama
├── requirements.txt           # Dependencies Python
├── README.md                  # Dokumentasi ini
├── .streamlit/                # Konfigurasi Streamlit
│   └── config.toml
├── models/                    # Model terlatih
│   ├── best_glcm_model.h5    # Model Neural Network
│   └── scaler_glcm.pkl       # StandardScaler
├── scripts/                   # Utility scripts
│   └── train_model.py        # Script training model
└── dataset/                   # Dataset (opsional)
    ├── retak/                # Gambar jalan retak
    └── tidak-retak/          # Gambar jalan normal
```

## 🔬 Cara Kerja Sistem

### Pipeline Deteksi

```
📸 Upload Gambar → 🔄 Preprocessing → 🧮 GLCM Extraction → 📊 Normalization → 🤖 ML Prediction → ✅ Hasil
```

### Proses Detail

1. **Input Image**

   - User mengupload gambar jalan (JPG/PNG/JPEG)
   - Format gambar divalidasi

2. **Preprocessing**

   - Resize gambar ke ukuran 128x128 pixel
   - Konversi dari RGB ke grayscale
   - Normalisasi intensitas pixel (0-255)

3. **GLCM Feature Extraction**

   - Hitung Gray-Level Co-occurrence Matrix
   - Ekstraksi 5 fitur statistik tekstur:
     - Contrast: Mengukur variasi intensitas lokal
     - Dissimilarity: Perbedaan rata-rata intensitas piksel
     - Homogeneity: Keseragaman tekstur (nilai tinggi = uniform)
     - Energy: Keteraturan gambar (nilai tinggi = homogen)
     - Correlation: Ketergantungan linear antar piksel

4. **Normalization**

   - Scaling fitur menggunakan StandardScaler
   - Normalisasi untuk stabilitas model

5. **Classification**

   - Input ke Neural Network (MLP Classifier)
   - Arsitektur: 5 → 256 → 512 → 1024 → 512 → 256 → 1 neuron
   - Aktivasi: ReLU untuk hidden layers, Sigmoid untuk output
   - Output: Probabilitas keretakan (0-1)

6. **Output & Visualization**
   - Hasil prediksi: Retak/Tidak Retak
   - Confidence score dalam persentase
   - Visualisasi fitur GLCM dalam grafik
   - Tabel detail nilai fitur

## 📊 Penjelasan Fitur GLCM

GLCM (Gray-Level Co-occurrence Matrix) adalah metode untuk menganalisis tekstur gambar dengan menghitung frekuensi kemunculan pasangan piksel dengan intensitas tertentu.

| Fitur             | Deskripsi                         | Interpretasi                                   |
| ----------------- | --------------------------------- | ---------------------------------------------- |
| **Contrast**      | Mengukur variasi intensitas lokal | Nilai tinggi = tekstur kasar (banyak retak)    |
| **Dissimilarity** | Perbedaan rata-rata intensitas    | Nilai tinggi = heterogenitas tinggi            |
| **Homogeneity**   | Keseragaman tekstur               | Nilai tinggi = permukaan uniform (tidak retak) |
| **Energy**        | Keteraturan/kehalusan gambar      | Nilai tinggi = tekstur homogen                 |
| **Correlation**   | Ketergantungan linear piksel      | Menunjukkan pola tekstur yang predictable      |

### Kenapa GLCM Efektif untuk Deteksi Retak?

- ✅ Retak menghasilkan pola tekstur yang **tidak teratur** (contrast tinggi, homogeneity rendah)
- ✅ Jalan normal memiliki tekstur yang **uniform** (homogeneity tinggi, contrast rendah)
- ✅ GLCM menangkap pola spasial yang tidak terlihat oleh mata telanjang
- ✅ Metode yang terbukti efektif dalam penelitian Computer Vision

## 🛠️ Teknologi yang Digunakan

### Frontend & UI

- **Streamlit** `1.28.0+` - Framework web application interaktif
- **Matplotlib** `3.7.0+` - Library visualisasi data dan grafik

### Machine Learning & AI

- **Scikit-learn** `1.3.0+` - MLPClassifier untuk Neural Network
- **StandardScaler** - Normalisasi fitur
- **TensorFlow** `2.13.0+` - (Opsional) Support untuk model Keras

### Image Processing

- **Scikit-image** `0.21.0+` - GLCM feature extraction
- **OpenCV (cv2)** `4.8.0+` - Computer vision operations
- **Pillow (PIL)** `10.0.0+` - Image manipulation

### Data Processing

- **NumPy** `1.24.0+` - Operasi array dan numerical computing
- **Pandas** `2.0.0+` - Data manipulation dan analysis

### Utilities

- **Joblib** `1.3.0+` - Model serialization (save/load model)

## 📝 Tips Penggunaan

✅ **Gunakan gambar dengan:**

- Pencahayaan yang baik
- Fokus yang tajam
- Resolusi minimal 256x256 pixel
- Format JPG, JPEG, a & FAQ

### ❓ Masalah Umum

**Q: Model files tidak ditemukan / Mode DEMO aktif**

```bash
# Jalankan training untuk generate model
python scripts/train_model.py
```

- Model akan disimpan di folder `models/`
- Menggunakan dummy data jika dataset tidak tersedia
- Untuk production, siapkan dataset real di `dataset/retak/` dan `dataset/tidak-retak/`

**Q: Error saat install dependencies**

```bash
# Windows
pip install -r requirements.txt

# Mac/Linux
pip3 install -r requirements.txt

# Jika ada konflik, gunakan virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

**Q: TensorFlow error (DLL initialization failed)**

- Sistem akan otomatis fallback ke sklearn
- T🎓 Referensi & Learning Resources

### Paper & Artikel

- GLCM Feature Extraction for Texture Analysis
- Deep Learning for Road Damage Detection
- Computer Vision in Infrastructure Maintenance

### Tutorial

- [Scikit-image GLCM Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Kontribusi

Contributions are welcome! Jika Anda ingin berkontribusi:

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add some amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

### Areas untuk Kontribusi

- 🐛 Bug fixes
- ✨ New features (multi-class classification, severity level)
- 📝 Documentation improvements
- 🧪 Testing dan validation
- 🎨 UI/UX improvements

## 📞 Dukungan & Komunitas

Jika ada pertanyaan atau masalah:

- 🔗 **GitHub Repository**: [https://github.com/diosamuel/road-crack-detection](https://github.com/diosamuel/road-crack-detection)
- 🐛 **Report Bug**: [Issues](https://github.com/diosamuel/road-crack-detection/issues)
- 💡 **Request Feature**: [New Issue](https://github.com/diosamuel/road-crack-detection/issues/new)
- 📧 **Contact**: Via GitHub Issues atau Pull Request

## 📄 Lisensi

Project ini dilisensikan di bawah **MIT License** - lihat file LICENSE untuk detail lengkap.

Anda bebas untuk:

- ✅ Menggunakan untuk keperluan komersial
- ✅ Memodifikasi source code
- ✅ Mendistribusikan ulang
- ✅ Menggunakan secara pribadi

Dengan syarat:

- 📝 Mencantumkan copyright notice
- 📝 Menyertakan lisensi dalam distribusi

## 👥 Developer & Credits

- **Original Developer**: [diosamuel](https://github.com/diosamuel)
- **Repository**: [road-crack-detection](https://github.com/diosamuel/road-crack-detection)
- **Inspiration**: Research papers on texture analysis dan computer vision

## 🌟 Acknowledgments

Terima kasih kepada:

- Open source community untuk tools dan libraries
- Contributors yang telah membantu pengembangan
- Researchers dalam bidang Computer Vision
- Streamlit team untuk framework yang amazing

---

<div align="center">
  <p>Dibuat dengan ❤️ menggunakan Streamlit, scikit-learn, dan scikit-image</p>
  <p><strong>Road Crack Detection System</strong> - Membantu menjaga infrastruktur jalan lebih baik</p>
  
  ⭐ **Star project ini jika bermanfaat!** ⭐
</div>
1. Buat folder struktur:
   ```
   dataset/
   ├── retak/           # Gambar jalan yang retak
   └── tidak-retak/     # Gambar jalan normal
   ```
2. Masukkan gambar ke folder masing-masing
3. Jalankan: `python scripts/train_model.py`
4. Model baru akan di-generate di folder `models/`

### 💡 Tips Optimasi

- Gunakan gambar dengan format JPG untuk ukuran file lebih kecil
- Crop gambar ke area yang ingin dianalisis saja
- Hindari gambar yang terlalu besar (>5MB) untuk performa lebih baik
- Untuk dataset training, gunakan minimal 100 gambar per kelas

**Error: Model not found**

- Pastikan file `best_glcm_model.h5` dan `scaler_glcm.pkl` ada di folder `models/`
- Atau jalankan training: `python scripts/train_model.py`

**Error: Package not found**

- Jalankan: `pip install -r requirements.txt`

**Aplikasi tidak bisa diakses**

- Cek apakah port 8501 sudah digunakan
- Coba jalankan dengan port berbeda: `streamlit run app.py --server.port 8502`

## 📞 Dukungan

Jika ada pertanyaan atau masalah:

- GitHub Repository: [https://github.com/diosamuel/road-crack-detection](https://github.com/diosamuel/road-crack-detection)
- Issues: [Report Bug/Request Feature](https://github.com/diosamuel/road-crack-detection/issues)

## 📄 Lisensi

MIT License

## 👥 Developer

- **Original Developer**: [diosamuel](https://github.com/diosamuel)
- **Repository**: [road-crack-detection](https://github.com/diosamuel/road-crack-detection)

---

Dibuat dengan ❤️ menggunakan Streamlit, scikit-learn, dan scikit-image
