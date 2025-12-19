import streamlit as st
st.write("Hai")
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import util
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# # Load model and scaler
model = joblib.load("./model/glcm_model_v2.pkl")
scaler = joblib.load("./model/glcm_model_scaler.pkl")
st.write("Model loaded")
st.write(model)
# # Load test data for evaluation
# try:
#     X_test = joblib.load("X_test.pkl")
#     y_test = joblib.load("y_test.pkl")
#     y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
# except FileNotFoundError:
#     accuracy = precision = recall = f1 = None

# def glcm_process(img):
#     if img.mode == "RGBA":
#         img = img.convert("RGB")
#     img_resized = img.resize((128, 128))
#     image = (256 * rgb2gray(np.array(img_resized))).astype(np.uint8)
#     image = util.img_as_ubyte(image)
#     distances = [50]
#     angles = [0, 45, 90, 135, 180]
#     glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
#     contrast = graycoprops(glcm, 'contrast')
#     dissimilarity = graycoprops(glcm, 'dissimilarity')
#     homogeneity = graycoprops(glcm, 'homogeneity')
#     energy = graycoprops(glcm, 'energy')
#     correlation = graycoprops(glcm, 'correlation')
#     brightness = np.mean(image)
#     img_array = np.array(img_resized)
#     image_contrast = img_array.std()
#     return contrast, dissimilarity, homogeneity, energy, correlation, brightness, image_contrast

# st.markdown("""
# <style>
#     .title {
#         text-align: center;
#         color: #2E86AB;
#         font-size: 2em;
#         margin-bottom: 5px;
#         font-weight: bold;
#     }
#     .subtitle {
#         text-align: center;
#         color: #666;
#         font-size: 1em;
#         margin-bottom: 10px;
#     }
#     .success-msg {
#         background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
#         color: #155724;
#         padding: 8px;
#         border-radius: 8px;
#         text-align: center;
#         margin: 5px 0;
#         border: 1px solid #28A745;
#         font-size: 0.9em;
#     }
#     .compact-metric {
#         background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
#         padding: 8px;
#         border-radius: 8px;
#         text-align: center;
#         margin: 2px;
#         border: 1px solid #2196F3;
#     }
#     .metric-value {
#         font-size: 1.2em;
#         font-weight: bold;
#         color: #0D47A1;
#     }
#     .metric-label {
#         font-size: 0.8em;
#         color: #1565C0;
#         margin-top: 2px;
#     }
#     .stButton button {
#         height: 40px;
#         font-size: 1em;
#     }
# </style>
# """, unsafe_allow_html=True)

# st.markdown('<h1 class="title">🛣️ Road Crack Detection System</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Sistem Deteksi Retak Jalan Menggunakan GLCM & Neural Network</p>', unsafe_allow_html=True)

# # Upload gambar
# uploaded_file = st.file_uploader("Pilih gambar jalan", type=["jpg", "jpeg", "png"], help="Upload gambar jalan untuk analisis retak")

# if uploaded_file is not None:
#     # Layout gambar dan tombol
#     col_img, col_space, col_btn = st.columns([1.2, 0.1, 1.5])

#     with col_img:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="🖼️ Gambar Jalan yang Diupload", width=280)

#     with col_btn:
#         st.markdown("### Analisis Gambar")
#         analyze_button = st.button("🔍 Mulai Analisis", type="primary", use_container_width=True)

#     if analyze_button:
#         # Analisis gambar
#         with st.spinner("🔍 Menganalisis..."):
#             contrast, dissimilarity, homogeneity, energy, correlation, brightness, image_contrast = glcm_process(image)
#             features = np.concatenate([contrast.flatten(), dissimilarity.flatten(), homogeneity.flatten(), energy.flatten(), correlation.flatten()]).reshape(1, -1)
#             features_scaled = scaler.transform(features)
#             prediction = model.predict(features_scaled).item()
#             prob_crack = prediction
#             prob_no_crack = 1 - prediction

#         st.markdown('<div class="success-msg">✅ Analisis selesai!</div>', unsafe_allow_html=True)

#         # Hasil Prediksi dalam satu baris
#         col_pred1, col_pred2, col_conf = st.columns([1, 1, 2])
#         with col_pred1:
#             st.metric("Retak", f"{prob_crack * 100:.1f}%")
#         with col_pred2:
#             st.metric("Tidak Retak", f"{prob_no_crack * 100:.1f}%")
#         with col_conf:
#             confidence = max(prob_crack, prob_no_crack)
#             if confidence > 0.8:
#                 st.success(f"🎯 Tingkat Kepercayaan Tinggi: {confidence:.1f}")
#             elif confidence > 0.6:
#                 st.warning(f"⚠️ Tingkat Kepercayaan Sedang: {confidence:.1f}")
#             else:
#                 st.error(f"❌ Tingkat Kepercayaan Rendah: {confidence:.1f}")

#         # Confusion Matrix Heatmap
#         st.subheader("📊 Matriks Evaluasi Model")
#         if accuracy is not None:
#             from sklearn.metrics import confusion_matrix
#             cm = confusion_matrix(y_test, y_pred)
#             tn, fp, fn, tp = cm.ravel()

#             # Create confusion matrix heatmap
#             fig, ax = plt.subplots(figsize=(4, 3))
#             cm_data = [[tn, fp], [fn, tp]]
#             labels = [['TN', 'FP'], ['FN', 'TP']]

#             # Create heatmap
#             im = ax.imshow(cm_data, cmap='Blues', aspect='equal')

#             # Add text annotations
#             for i in range(2):
#                 for j in range(2):
#                     text = ax.text(j, i, f'{labels[i][j]}\n{cm_data[i][j]}',
#                                  ha="center", va="center", color="white", fontsize=12, fontweight='bold')

#             ax.set_xticks([0, 1])
#             ax.set_yticks([0, 1])
#             ax.set_xticklabels(['Predicted\nNo Crack', 'Predicted\nCrack'])
#             ax.set_yticklabels(['Actual\nNo Crack', 'Actual\nCrack'])
#             ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

#             # Add colorbar
#             cbar = plt.colorbar(im, ax=ax, shrink=0.8)
#             cbar.set_label('Count', rotation=270, labelpad=15)

#             plt.tight_layout()
#             st.pyplot(fig)

#             # Confusion Matrix Values in Boxes
#             st.markdown("### Nilai Matriks Evaluasi")
#             col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
#             with col_cm1:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{tn}</div>
#                     <div class="metric-label">True Negative (TN)</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_cm2:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{fp}</div>
#                     <div class="metric-label">False Positive (FP)</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_cm3:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{fn}</div>
#                     <div class="metric-label">False Negative (FN)</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_cm4:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{tp}</div>
#                     <div class="metric-label">True Positive (TP)</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.info("Confusion matrix belum tersedia")

#         # Charts dalam satu baris
#         col_chart1, col_chart2 = st.columns(2)

#         with col_chart1:
#             st.subheader("📈 Fitur GLCM")
#             features_dict = {
#                 "Contrast": contrast[0][0],
#                 "Dissimilarity": dissimilarity[0][0],
#                 "Homogeneity": homogeneity[0][0],
#                 "Energy": energy[0][0],
#                 "Correlation": correlation[0][0],
#                 "Brightness": brightness,
#                 "Image Contrast": image_contrast
#             }
#             fig, ax = plt.subplots(figsize=(4, 2.5))
#             bars = ax.bar(range(len(features_dict)), features_dict.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
#             ax.set_xticks(range(len(features_dict)))
#             ax.set_xticklabels([k[:4] + '...' if len(k) > 4 else k for k in features_dict.keys()], fontsize=6, rotation=45)
#             ax.set_ylabel("Nilai", fontsize=7)
#             ax.set_title("Fitur GLCM Gambar", fontsize=9, fontweight='bold')
#             ax.tick_params(axis='y', labelsize=6)
#             max_val = max(features_dict.values())
#             if max_val > 0:
#                 ax.set_ylim(0, max_val * 1.1)
#             plt.tight_layout()
#             st.pyplot(fig)

#         with col_chart2:
#             st.subheader("📈 Training History")
#             if history is not None:
#                 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))

#                 # Plot accuracy
#                 ax1.plot(history['accuracy'], 'b-', label='Training', linewidth=1.5)
#                 ax1.plot(history['val_accuracy'], 'r--', label='Validation', linewidth=1.5)
#                 ax1.set_title('Model Accuracy', fontsize=9, fontweight='bold')
#                 ax1.set_ylabel('Accuracy', fontsize=7)
#                 ax1.legend(loc='lower right', fontsize=6)
#                 ax1.grid(True, alpha=0.3)
#                 ax1.tick_params(labelsize=6)

#                 # Plot loss
#                 ax2.plot(history['loss'], 'b-', label='Training', linewidth=1.5)
#                 ax2.plot(history['val_loss'], 'r--', label='Validation', linewidth=1.5)
#                 ax2.set_title('Model Loss', fontsize=9, fontweight='bold')
#                 ax2.set_ylabel('Loss', fontsize=7)
#                 ax2.set_xlabel('Epoch', fontsize=7)
#                 ax2.legend(loc='upper right', fontsize=6)
#                 ax2.grid(True, alpha=0.3)
#                 ax2.tick_params(labelsize=6)

#                 plt.tight_layout()
#                 st.pyplot(fig)
#             else:
#                 st.info("Training history tidak tersedia")

#         # Metrik Evaluasi Model dalam kotak (paling bawah)
#         st.subheader("📊 Metrik Evaluasi Model")
#         if accuracy is not None:
#             col_met1, col_met2, col_met3, col_met4 = st.columns(4)
#             with col_met1:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{accuracy:.3f}</div>
#                     <div class="metric-label">Accuracy</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_met2:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{precision:.3f}</div>
#                     <div class="metric-label">Precision</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_met3:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{recall:.3f}</div>
#                     <div class="metric-label">Recall</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             with col_met4:
#                 st.markdown(f"""
#                 <div class="compact-metric">
#                     <div class="metric-value">{f1:.3f}</div>
#                     <div class="metric-label">F1-Score</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.info("Metrik evaluasi belum tersedia")
