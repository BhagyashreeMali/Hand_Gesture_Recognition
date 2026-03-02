import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path, img_size=64):
    """
    Loads and preprocesses the Leap GestRecog dataset.
    """
    X = []
    y = []
    label_map = {}
    label_id = 0

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} not found.")

    # Iterate through subject folders (00, 01, ...)
    for subject in sorted(os.listdir(data_path)):
        if not subject.isdigit(): continue
        
        subject_path = os.path.join(data_path, subject)
        if not os.path.isdir(subject_path): continue

        # Iterate through gesture folders (01_palm, 02_l, ...)
        for gesture in sorted(os.listdir(subject_path)):
            if gesture.startswith('.'): continue
            
            if gesture not in label_map:
                label_map[gesture] = label_id
                label_id += 1

            gesture_path = os.path.join(subject_path, gesture)
            if not os.path.isdir(gesture_path): continue

            for img in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None: continue
                image = cv2.resize(image, (img_size, img_size))
                X.append(image)
                y.append(label_map[gesture])

    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    return X, np.array(y), label_map

def get_augmentor():
    """
    Returns an ImageDataGenerator for data augmentation.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
