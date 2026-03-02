import os
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

IMG_SIZE = 64
MODEL_PATH = "gesture_model.h5"
DATA_PATH = "Data Set /archive/leapGestRecog"

def get_sample_image():
    # Pick a random subject and gesture
    subjects = [s for s in os.listdir(DATA_PATH) if s.isdigit()]
    subject = random.choice(subjects)
    subject_path = os.path.join(DATA_PATH, subject)
    
    gestures = [g for g in os.listdir(subject_path) if not g.startswith('.')]
    gesture = random.choice(gestures)
    gesture_path = os.path.join(subject_path, gesture)
    
    images = [i for i in os.listdir(gesture_path) if i.endswith(('.jpg', '.png', '.jpeg'))]
    image_name = random.choice(images)
    return os.path.join(gesture_path, image_name), gesture

def predict_gesture(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found. Run train.py first.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    model = load_model(MODEL_PATH)
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(image)
    gesture_class = np.argmax(prediction)
    
    return gesture_class

if __name__ == "__main__":
    test_img = "test_image.jpg"
    if not os.path.exists(test_img):
        print("test_image.jpg not found, picking a sample from the dataset...")
        test_img, true_gesture = get_sample_image()
        print(f"Selected sample: {test_img} (True Gesture: {true_gesture})")
    
    gesture_class = predict_gesture(test_img)
    if gesture_class is not None:
        print(f"Predicted Gesture Class: {gesture_class}")
