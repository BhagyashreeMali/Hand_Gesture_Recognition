"""
Streamlit Web Dashboard for Hand Gesture Recognition
Features: Upload & Predict, Real-time Webcam, Model Performance Visualization
"""

import os
import sys
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 64
MODEL_DIR = "saved_models"
RESULTS_DIR = "results"

# ============================================================
# Helper Functions
# ============================================================
@st.cache_resource
def load_gesture_model():
    """Load the trained gesture model."""
    model_path = os.path.join(MODEL_DIR, "best_advanced_cnn.h5")
    if not os.path.exists(model_path):
        model_path = "gesture_model.h5"
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

@st.cache_data
def load_label_map():
    """Load the gesture label map."""
    label_path = os.path.join(MODEL_DIR, "label_map.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            return json.load(f)
    return None

def preprocess_image(image, img_size=IMG_SIZE):
    """Preprocess image for prediction."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_size, img_size))
    image = image.reshape(1, img_size, img_size, 1) / 255.0
    return image

def predict_gesture(model, image, label_map):
    """Predict gesture from preprocessed image."""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    # Reverse the label map
    idx_to_label = {v: k for k, v in label_map.items()}
    gesture_name = idx_to_label.get(class_idx, f"Class {class_idx}")
    
    return gesture_name, confidence, prediction[0]

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="🖐️ Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high { color: #00c853; font-weight: bold; }
    .confidence-medium { color: #ffab00; font-weight: bold; }
    .confidence-low { color: #ff1744; font-weight: bold; }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🖐️ Hand Gesture Recognition</h1>
    <p>Powered by Deep Learning • TensorFlow • Mediapipe • OpenCV</p>
</div>
""", unsafe_allow_html=True)

# Load model and labels
model = load_gesture_model()
label_map = load_label_map()

if model is None:
    st.error("⚠️ No trained model found! Run `python train_advanced.py` first.")
    st.stop()

if label_map is None:
    st.warning("⚠️ Label map not found. Predictions will show class indices only.")
    label_map = {f"Class_{i}": i for i in range(10)}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow", width=200)
    st.markdown("---")
    st.header("🎛️ Settings")
    
    mode = st.radio("Select Mode", ["📤 Upload Image", "📊 Model Performance"], index=0)
    
    st.markdown("---")
    st.header("📋 Supported Gestures")
    gesture_emojis = {
        "01_palm": "🖐️ Palm", "02_l": "🤟 L-Sign", "03_fist": "✊ Fist",
        "04_fist_moved": "👊 Fist Moved", "05_thumb": "👍 Thumb",
        "06_index": "☝️ Index", "07_ok": "👌 OK",
        "08_palm_moved": "🖐️ Palm Moved", "09_c": "🤏 C-Sign",
        "10_down": "👇 Down"
    }
    for gesture, emoji in gesture_emojis.items():
        st.write(f"  {emoji}")
    
    st.markdown("---")
    st.caption("Built with ❤️ by Bhagyashree Mali")

# ============================================================
# Main Content
# ============================================================
if mode == "📤 Upload Image":
    st.header("📤 Upload & Predict")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a hand gesture image", type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert to numpy array for prediction
            img_array = np.array(image)
            gesture_name, confidence, all_probs = predict_gesture(model, img_array, label_map)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🎯 Prediction Result")
            
            # Confidence color
            if confidence > 0.9:
                conf_class = "confidence-high"
            elif confidence > 0.7:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>{gesture_name}</h2>
                <p class="{conf_class}">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all class probabilities
            st.markdown("### 📊 Class Probabilities")
            idx_to_label = {v: k for k, v in label_map.items()}
            prob_data = {idx_to_label.get(i, f"Class {i}"): float(all_probs[i]) 
                        for i in range(len(all_probs))}
            st.bar_chart(prob_data)

elif mode == "📊 Model Performance":
    st.header("📊 Model Performance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training curves
        curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training & Validation Curves")
        else:
            st.info("Training curves not yet generated. Run `python train_advanced.py` first.")
    
    with col2:
        # Confusion matrix
        cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.info("Confusion matrix not yet generated. Run `python train_advanced.py` first.")
    
    # Classification report
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        st.markdown("### 📋 Classification Report")
        with open(report_path) as f:
            st.code(f.read(), language="text")
