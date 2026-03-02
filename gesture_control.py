"""
Real-Time Gesture Control Demo
Uses OpenCV webcam + trained model to recognize gestures in real-time.
Demonstrates gesture-to-action mapping (volume control simulation, keyboard shortcuts).
"""

import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to path 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.hand_tracking import HandTracker

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 64
MODEL_DIR = "saved_models"

# Gesture-to-Action mapping
GESTURE_ACTIONS = {
    "01_palm": "⏸️  PAUSE",
    "02_l":    "⏭️  NEXT",
    "03_fist": "🔇 MUTE",
    "04_fist_moved": "🔊 VOLUME UP",
    "05_thumb": "👍 LIKE",
    "06_index": "☝️  SELECT",
    "07_ok":   "✅ CONFIRM",
    "08_palm_moved": "⏮️  PREVIOUS",
    "09_c":    "📷 CAPTURE",
    "10_down": "⏬ VOLUME DOWN"
}

def load_resources():
    """Load the model and label map."""
    model_path = os.path.join(MODEL_DIR, "best_advanced_cnn.h5")
    if not os.path.exists(model_path):
        model_path = "gesture_model.h5"
    if not os.path.exists(model_path):
        print("Error: No model found! Run train_advanced.py first.")
        return None, None
    
    model = load_model(model_path)
    
    label_path = os.path.join(MODEL_DIR, "label_map.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_map = json.load(f)
    else:
        label_map = {f"Class_{i}": i for i in range(10)}
    
    return model, label_map

def draw_info_panel(frame, gesture_name, confidence, action):
    """Draw a styled info panel on the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 140), (w - 10, h - 10), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture_name}", (20, h - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Confidence bar
    bar_width = int((w - 60) * confidence)
    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (20, h - 80), (20 + bar_width, h - 65), color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%", (w - 80, h - 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Action
    cv2.putText(frame, f"Action: {action}", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    
    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 40), (100, 50, 200), -1)
    cv2.putText(frame, "Hand Gesture Recognition - Real-Time Demo", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    model, label_map = load_resources()
    if model is None:
        return
    
    idx_to_label = {v: k for k, v in label_map.items()}
    tracker = HandTracker(detection_con=0.7, track_con=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        print("If running remotely, use 'streamlit run app.py' for the web dashboard instead.")
        return
    
    print("=" * 50)
    print("REAL-TIME GESTURE CONTROL DEMO")
    print("Press 'q' to quit | Press 's' to screenshot")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)
        
        # Extract hand region and predict
        h, w = frame.shape[:2]
        positions = tracker.get_positions(frame, draw=False)
        
        gesture_name = "No Hand Detected"
        confidence = 0.0
        action = "---"
        
        if positions:
            # Get bounding box of hand
            x_coords = [p[1] for p in positions]
            y_coords = [p[2] for p in positions]
            x_min, x_max = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
            y_min, y_max = max(0, min(y_coords) - 30), min(h, max(y_coords) + 30)
            
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            if hand_region.size > 0:
                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                processed = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                
                prediction = model.predict(processed, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = float(np.max(prediction))
                gesture_name = idx_to_label.get(class_idx, f"Class {class_idx}")
                action = GESTURE_ACTIONS.get(gesture_name, "Unknown")
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        draw_info_panel(frame, gesture_name, confidence, action)
        
        cv2.imshow("Hand Gesture Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
