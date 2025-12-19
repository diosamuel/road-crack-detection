import streamlit as st
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import joblib

from glcm import glcm_process

# Page config
st.set_page_config(
    page_title="Road Crack Detection",
    page_icon="🛣️",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .crack-detected {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .no-crack {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🛣️ Road Crack Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image to detect road cracks using GLCM feature extraction</p>', unsafe_allow_html=True)

# Load model and scaler (cached for performance)
@st.cache_resource
def load_model():
    model = joblib.load("./model/glcm_model_v2.pkl")
    scaler = joblib.load("./model/glcm_model_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a road surface image to analyze"
)

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Grayscale Preview")
        # Convert to grayscale for preview
        img_resized = image.convert("RGB").resize((256, 256))
        gray_array = (255 * rgb2gray(np.array(img_resized))).astype(np.uint8)
        st.image(gray_array, use_container_width=True)
    
    # Process button
    if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Extracting GLCM features..."):
            # Extract GLCM features
            features_df = glcm_process(image)
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Predict
            prediction = model.predict(features_scaled)
        
        st.divider()
        
        # Handle different prediction output formats
        if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
            pred_value = prediction[0][0]
        else:
            pred_value = prediction[0]
        
        # Display result
        st.subheader("📊 Analysis Result")
        
        if pred_value < 0.5:
            st.markdown(
                '<div class="result-box no-crack">✅ No Crack Detected</div>',
                unsafe_allow_html=True
            )
            st.success("The road surface appears to be in good condition.")
        else:
            st.markdown(
                '<div class="result-box crack-detected">⚠️ Crack Detected!</div>',
                unsafe_allow_html=True
            )
            st.warning("The road surface shows signs of cracking and may need maintenance.")
        
        # Show confidence score
        confidence = abs(pred_value - 0.5) * 2 * 100  # Convert to percentage
        st.metric("Prediction Score", f"{pred_value:.4f}")
        
        # Show extracted features (expandable)
        with st.expander("📋 View Extracted GLCM Features"):
            st.dataframe(features_df.T.rename(columns={0: "Value"}), use_container_width=True)

else:
    # Placeholder when no image is uploaded
    st.info("👆 Upload an image to get started")
    
    # Instructions
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Upload** a road surface image (JPG, PNG, or BMP)
        2. **Click** the "Analyze Image" button
        3. **View** the crack detection result
        
        The system uses **GLCM (Gray Level Co-occurrence Matrix)** features to analyze 
        texture patterns in the road surface and detect potential cracks.
        """)
