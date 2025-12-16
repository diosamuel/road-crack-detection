"""
Script training alternatif menggunakan model sederhana (tanpa TensorFlow)
Kompatibel dengan semua sistem dan menghasilkan file yang sama
"""

import os
import numpy as np
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import util
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

def glcm_process(img_path):
    """Extract GLCM features from image"""
    try:
        im_frame = Image.open(img_path)
        if im_frame.mode == "RGBA":
            im_frame = im_frame.convert("RGB")
        im_frame = im_frame.resize((128, 128))
        
        image = (256 * rgb2gray(np.array(im_frame))).astype(np.uint8)
        image = util.img_as_ubyte(image)
        
        # GLCM calculation
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
        
        return contrast, dissimilarity, homogeneity, energy, correlation
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def load_dataset(dataset_path):
    """Load dataset dari folder struktur: dataset_path/retak/ dan dataset_path/tidak-retak/"""
    print("Loading dataset...")
    
    data = []
    labels = []
    
    # Load crack images
    crack_path = os.path.join(dataset_path, 'retak')
    if os.path.exists(crack_path):
        crack_images = [f for f in os.listdir(crack_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(crack_images)} crack images")
        
        for img_name in crack_images:
            features = glcm_process(os.path.join(crack_path, img_name))
            if features:
                data.append(features)
                labels.append(1)  # 1 = retak
    
    # Load non-crack images
    no_crack_path = os.path.join(dataset_path, 'tidak-retak')
    if os.path.exists(no_crack_path):
        no_crack_images = [f for f in os.listdir(no_crack_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(no_crack_images)} non-crack images")
        
        for img_name in no_crack_images:
            features = glcm_process(os.path.join(no_crack_path, img_name))
            if features:
                data.append(features)
                labels.append(0)  # 0 = tidak retak
    
    if len(data) == 0:
        print("WARNING: No dataset found! Creating dummy data for testing...")
        return create_dummy_data()
    
    return np.array(data), np.array(labels)

def create_dummy_data():
    """Create dummy data jika dataset tidak tersedia"""
    print("Creating dummy training data (100 samples each class)...")
    
    np.random.seed(42)
    
    # Generate synthetic GLCM features for crack images (higher contrast, dissimilarity)
    crack_data = []
    for _ in range(100):
        contrast = np.random.uniform(20, 100)
        dissimilarity = np.random.uniform(5, 20)
        homogeneity = np.random.uniform(0.3, 0.6)
        energy = np.random.uniform(0.01, 0.05)
        correlation = np.random.uniform(0.5, 0.8)
        crack_data.append([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # Generate synthetic GLCM features for non-crack images (lower contrast, dissimilarity)
    no_crack_data = []
    for _ in range(100):
        contrast = np.random.uniform(5, 30)
        dissimilarity = np.random.uniform(1, 8)
        homogeneity = np.random.uniform(0.6, 0.9)
        energy = np.random.uniform(0.05, 0.15)
        correlation = np.random.uniform(0.7, 0.95)
        no_crack_data.append([contrast, dissimilarity, homogeneity, energy, correlation])
    
    data = np.array(crack_data + no_crack_data)
    labels = np.array([1] * 100 + [0] * 100)
    
    return data, labels

# Buat wrapper class untuk sklearn model agar kompatibel dengan predict keras-style
class KerasStyleWrapper:
    """Wrapper untuk membuat sklearn model kompatibel dengan format Keras"""
    def __init__(self, sklearn_model):
        self.model = sklearn_model
    
    def predict(self, X, verbose=0):
        """Predict dengan output format seperti Keras"""
        # sklearn predict_proba returns [prob_class_0, prob_class_1]
        # kita ambil prob_class_1 saja (probability untuk kelas "retak")
        proba = self.model.predict_proba(X)
        return proba[:, 1:2]  # return sebagai 2D array seperti Keras
    
    def evaluate(self, X, y):
        """Evaluate model"""
        from sklearn.metrics import log_loss, accuracy_score
        predictions = self.model.predict(X)
        loss = log_loss(y, self.model.predict_proba(X))
        accuracy = accuracy_score(y, predictions)
        return loss, accuracy
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load_model(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            sklearn_model = pickle.load(f)
        return KerasStyleWrapper(sklearn_model)

def train_and_save_model(dataset_path='dataset'):
    """Main training function"""
    print("="*60)
    print("GLCM + Neural Network Training Script (sklearn version)")
    print("="*60)
    
    # Load data
    X, y = load_dataset(dataset_path)
    print(f"\nTotal samples: {len(X)}")
    print(f"Crack samples: {np.sum(y == 1)}")
    print(f"Non-crack samples: {np.sum(y == 0)}")
    
    # Create and fit scaler
    print("\nNormalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build model (MLPClassifier = Neural Network di sklearn)
    print("\nBuilding Neural Network model...")
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(256, 512, 1024, 512, 256),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=100,
        random_state=42,
        verbose=True
    )
    
    # Train model
    print("\nTraining model...")
    sklearn_model.fit(X_train, y_train)
    
    # Wrap model untuk format Keras
    model = KerasStyleWrapper(sklearn_model)
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    model.save('models/best_glcm_model.h5')
    joblib.dump(scaler, 'models/scaler_glcm.pkl')
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("Files created:")
    print("  - models/best_glcm_model.h5 (Neural Network model)")
    print("  - models/scaler_glcm.pkl (Feature scaler)")
    print("="*60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("="*60)
    
    return model, scaler

if __name__ == "__main__":
    # Cek apakah ada dataset
    if os.path.exists('dataset'):
        print("Dataset folder found!")
        print("Expected structure:")
        print("  dataset/")
        print("    ├── retak/       (crack images)")
        print("    └── tidak-retak/ (non-crack images)")
        print()
    else:
        print("No dataset folder found.")
        print("Will create dummy data for testing purposes.")
        print("\nTo train with real data, create folder structure:")
        print("  dataset/")
        print("    ├── retak/       (put crack images here)")
        print("    └── tidak-retak/ (put non-crack images here)")
        print()
    
    # Train model
    model, scaler = train_and_save_model()
