import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import util
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Road Crack Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .crack-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .no-crack {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Try loading TensorFlow model first
        try:
            model = tf.keras.models.load_model('models/best_glcm_model.h5')
        except:
            # If TensorFlow fails, load sklearn model with pickle
            import pickle
            with open('models/best_glcm_model.h5', 'rb') as f:
                model = pickle.load(f)
        
        scaler = joblib.load('models/scaler_glcm.pkl')
        return model, scaler, True
    except Exception as e:
        st.warning(f"⚠️ Model files tidak ditemukan. Menjalankan dalam mode DEMO.")
        st.info("💡 Untuk mode penuh, jalankan notebook training terlebih dahulu untuk menghasilkan file model.")
        return None, None, False

# GLCM Feature Extraction Function
def glcm_process(img_array):
    """Extract GLCM features from image array"""
    # Convert to PIL Image if needed
    if isinstance(img_array, np.ndarray):
        im_frame = Image.fromarray(img_array)
    else:
        im_frame = img_array
    
    # Convert RGBA to RGB if needed
    if im_frame.mode == "RGBA":
        im_frame = im_frame.convert("RGB")
    
    # Resize image
    im_frame = im_frame.resize((128, 128))
    
    # Convert to grayscale
    image = (256 * rgb2gray(np.array(im_frame))).astype(np.uint8)
    image = util.img_as_ubyte(image)
    
    # Calculate GLCM
    distances = [50]
    angles = [np.pi/2]
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    
    # Extract features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return contrast, dissimilarity, homogeneity, energy, correlation, image

# Main App
def main():
    # Header
    st.markdown("<h1 class='stTitle'>🛣️ Road Crack Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Deteksi Keretakan Jalan menggunakan GLCM Feature Extraction & Deep Learning</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/road.png", width=100)
        st.title("📋 Menu")
        page = st.radio("Pilih Halaman:", ["🏠 Deteksi Retak", "📊 Tentang Sistem", "📖 Panduan"])
        
        st.markdown("---")
        st.markdown("### ℹ️ Informasi")
        st.info("Sistem ini menggunakan:\n- GLCM Feature Extraction\n- Deep Learning (Neural Network)\n- 5 Fitur Tekstur")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("Created by [diosamuel](https://github.com/diosamuel)")
        st.markdown("[🔗 GitHub Repository](https://github.com/diosamuel/road-crack-detection)")
    
    # Load model
    model, scaler, model_loaded = load_model_and_scaler()
    
    # Page routing
    if page == "🏠 Deteksi Retak":
        detection_page(model, scaler, model_loaded)
    elif page == "📊 Tentang Sistem":
        about_page()
    else:
        guide_page()

def detection_page(model, scaler, model_loaded):
    """Main detection page"""
    st.markdown("### 📸 Upload Gambar Jalan")
    
    if not model_loaded:
        st.warning("🔧 Mode DEMO - Fitur prediksi tidak tersedia. Hanya menampilkan ekstraksi fitur GLCM.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Pilih gambar jalan yang ingin dianalisis",
            type=['jpg', 'jpeg', 'png'],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Read and display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Gambar Original", use_container_width=True)
            
            # Process button
            if st.button("🔍 Analisis Gambar", type="primary", use_container_width=True):
                with st.spinner("Sedang menganalisis gambar..."):
                    process_image(image, model, scaler, col2, model_loaded)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
            <h3>💡 Tips:</h3>
            <ul>
                <li>Gunakan gambar jalan yang jelas</li>
                <li>Pastikan area yang ingin dideteksi terlihat</li>
                <li>Resolusi minimal 256x256 pixel</li>
                <li>Format gambar: JPG, JPEG, atau PNG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def process_image(image, model, scaler, display_col, model_loaded):
    """Process uploaded image and display results"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Extract GLCM features
        contrast, dissimilarity, homogeneity, energy, correlation, processed_img = glcm_process(img_array)
        
        # Display processed image
        with display_col:
            st.image(processed_img, caption="🔬 Gambar Grayscale (Processed)", use_container_width=True, clamp=True)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis")
        
        # Prediction result - only if model is loaded
        if model_loaded and model is not None and scaler is not None:
            # Prepare features for prediction
            features = np.array([[contrast, dissimilarity, homogeneity, energy, correlation]])
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            
            # Handle both sklearn and Keras model output formats
            if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                # Keras format: 2D array [[prob]]
                confidence = prediction[0][0]
            else:
                # sklearn format: 1D array [class] - need to get probability
                confidence = model.predict_proba(features_scaled)[0][1]
            
            # Prediction result
            if confidence >= 0.5:
                st.markdown(f"""
                    <div class='result-box crack-detected'>
                        ⚠️ RETAK TERDETEKSI<br>
                        <span style='font-size: 1.2rem;'>Confidence: {confidence*100:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.error("Jalan memerlukan perbaikan segera!")
            else:
                st.markdown(f"""
                    <div class='result-box no-crack'>
                        ✅ TIDAK ADA RETAK<br>
                        <span style='font-size: 1.2rem;'>Confidence: {(1-confidence)*100:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.success("Kondisi jalan dalam keadaan baik!")
        else:
            st.info("🔧 Mode DEMO - Prediksi tidak tersedia. Menampilkan fitur GLCM saja.")
        
        # Display features
        st.markdown("### 🔢 Fitur GLCM yang Diekstrak")
        
        # Create columns for metrics
        metric_cols = st.columns(5)
        
        features_data = [
            ("Contrast", contrast, "Variasi intensitas lokal"),
            ("Dissimilarity", dissimilarity, "Perbedaan intensitas"),
            ("Homogeneity", homogeneity, "Keseragaman tekstur"),
            ("Energy", energy, "Keteraturan gambar"),
            ("Correlation", correlation, "Ketergantungan pixel")
        ]
        
        for i, (name, value, desc) in enumerate(features_data):
            with metric_cols[i]:
                st.metric(label=name, value=f"{value:.3f}")
                st.caption(desc)
        
        # Feature visualization
        st.markdown("### 📈 Visualisasi Fitur")
        fig, ax = plt.subplots(figsize=(10, 4))
        features_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
        features_values = [contrast, dissimilarity, homogeneity, energy, correlation]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax.bar(features_names, features_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Nilai', fontsize=12, fontweight='bold')
        ax.set_title('Nilai Fitur GLCM', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature table
        st.markdown("### 📋 Detail Fitur")
        df_features = pd.DataFrame({
            'Fitur': features_names,
            'Nilai': features_values,
            'Deskripsi': [desc for _, _, desc in features_data]
        })
        st.dataframe(df_features, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")

def about_page():
    """About system page"""
    st.markdown("## 📊 Tentang Sistem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Tujuan Sistem
        Sistem ini dirancang untuk mendeteksi keretakan pada permukaan jalan secara otomatis menggunakan 
        teknik Computer Vision dan Deep Learning.
        
        ### 🔬 Teknologi yang Digunakan
        - **GLCM (Gray-Level Co-occurrence Matrix)**: Untuk ekstraksi fitur tekstur
        - **Deep Learning**: Neural Network untuk klasifikasi
        - **TensorFlow/Keras**: Framework deep learning
        - **Scikit-image**: Untuk image processing
        - **Streamlit**: Framework untuk web application
        
        ### 📐 Arsitektur Sistem
        1. **Input**: Gambar jalan (JPG/PNG)
        2. **Preprocessing**: Resize dan konversi ke grayscale
        3. **Feature Extraction**: GLCM features (5 fitur)
        4. **Normalization**: StandardScaler
        5. **Classification**: Neural Network
        6. **Output**: Retak/Tidak Retak + Confidence
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Fitur GLCM yang Diekstrak
        
        **1. Contrast**
        - Mengukur variasi intensitas lokal
        - Nilai tinggi = perbedaan besar antar pixel tetangga
        
        **2. Dissimilarity**
        - Mengukur rata-rata perbedaan intensitas
        - Nilai tinggi = tekstur lebih heterogen
        
        **3. Homogeneity**
        - Mengukur keseragaman tekstur
        - Nilai tinggi = tekstur lebih uniform
        
        **4. Energy**
        - Mengukur keteraturan gambar
        - Nilai tinggi = tekstur lebih homogen
        
        **5. Correlation**
        - Mengukur ketergantungan linear antar pixel
        - Nilai tinggi = tekstur lebih predictable
        
        ### 🎯 Akurasi Model
        Model ini dilatih menggunakan:
        - Random Search hyperparameter optimization
        - Early stopping untuk mencegah overfitting
        - Stratified train-test split (80:20)
        """)
    
    workflow_cols = st.columns(5)
    
    workflow_steps = [
        ("1️⃣", "Upload", "Gambar"),
        ("2️⃣", "Preprocess", "Resize & Grayscale"),
        ("3️⃣", "Extract", "GLCM Features"),
        ("4️⃣", "Predict", "Neural Network"),
        ("5️⃣", "Result", "Crack/No Crack")
    ]
    
    for i, (icon, title, desc) in enumerate(workflow_steps):
        with workflow_cols[i]:
            st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
                    <div style='font-size: 2rem;'>{icon}</div>
                    <div style='font-weight: bold; margin: 0.5rem 0;'>{title}</div>
                    <div style='font-size: 0.8rem; color: #666;'>{desc}</div>
                </div>
            """, unsafe_allow_html=True)

def guide_page():
    """User guide page"""
    st.markdown("## 📖 Panduan Penggunaan")
    
    st.markdown("""
    ### 🚀 Cara Menggunakan Sistem
    
    #### 1️⃣ Persiapan
    - Pastikan Anda memiliki gambar jalan yang ingin dianalisis
    - Format gambar yang didukung: JPG, JPEG, PNG
    - Resolusi minimal 256x256 pixel untuk hasil terbaik
    
    #### 2️⃣ Upload Gambar
    - Klik pada halaman "🏠 Deteksi Retak" di sidebar
    - Klik tombol "Browse files" untuk memilih gambar
    - Atau drag & drop gambar ke area upload
    
    #### 3️⃣ Analisis
    - Setelah gambar ter-upload, klik tombol "🔍 Analisis Gambar"
    - Tunggu beberapa detik hingga proses selesai
    - Sistem akan menampilkan hasil analisis
    
    #### 4️⃣ Membaca Hasil
    - **Hasil Utama**: Menunjukkan apakah jalan retak atau tidak
    - **Confidence Score**: Tingkat kepercayaan prediksi (0-100%)
    - **Fitur GLCM**: Nilai-nilai fitur yang diekstrak dari gambar
    - **Visualisasi**: Grafik dan tabel untuk memahami hasil lebih detail
    
    ### 💡 Tips untuk Hasil Terbaik
    
    ✅ **DO:**
    - Gunakan gambar dengan pencahayaan yang baik
    - Pastikan area jalan terlihat jelas
    - Gunakan gambar close-up untuk keretakan kecil
    - Ambil gambar tegak lurus terhadap permukaan jalan
    
    ❌ **DON'T:**
    - Jangan gunakan gambar yang terlalu gelap atau blur
    - Hindari gambar dengan banyak bayangan
    - Jangan gunakan gambar yang terlalu jauh dari objek
    - Hindari gambar dengan resolusi sangat rendah
    
    ### 🔧 Troubleshooting
    
    **Q: Hasil prediksi tidak akurat?**
    - Pastikan gambar berkualitas baik dan pencahayaan cukup
    - Coba ambil gambar dari jarak yang lebih dekat
    - Pastikan area retak terlihat jelas dalam gambar
    
    **Q: Error saat upload gambar?**
    - Cek format file (harus JPG, JPEG, atau PNG)
    - Pastikan ukuran file tidak terlalu besar (< 10MB)
    - Coba refresh halaman dan upload ulang
    
    **Q: Proses analisis lama?**
    - Ini normal untuk gambar berukuran besar
    - Tunggu hingga proses selesai (biasanya < 10 detik)
    - Jangan refresh halaman saat proses berjalan
    
    ### 📞 Bantuan Lebih Lanjut
    
    Jika Anda mengalami masalah atau memiliki pertanyaan:
    - GitHub: [https://github.com/diosamuel/road-crack-detection](https://github.com/diosamuel/road-crack-detection)
    - Issues: [Report Bug/Request Feature](https://github.com/diosamuel/road-crack-detection/issues)
    """)
    
    # Example images section
    st.markdown("---")
    st.markdown("### 📷 Contoh Gambar yang Baik")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        st.markdown("""
        <div style='padding: 1rem; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
            <div style='font-size: 3rem;'>✅</div>
            <div style='font-weight: bold; margin: 0.5rem 0;'>Pencahayaan Baik</div>
            <div style='font-size: 0.9rem;'>Gambar jelas dengan cahaya cukup</div>
        </div>
        """, unsafe_allow_html=True)
    
    with example_cols[1]:
        st.markdown("""
        <div style='padding: 1rem; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
            <div style='font-size: 3rem;'>✅</div>
            <div style='font-weight: bold; margin: 0.5rem 0;'>Jarak Tepat</div>
            <div style='font-size: 0.9rem;'>Close-up dengan detail jelas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with example_cols[2]:
        st.markdown("""
        <div style='padding: 1rem; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
            <div style='font-size: 3rem;'>✅</div>
            <div style='font-weight: bold; margin: 0.5rem 0;'>Fokus Tajam</div>
            <div style='font-size: 0.9rem;'>Tidak blur atau buram</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
