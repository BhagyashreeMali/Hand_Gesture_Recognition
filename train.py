import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import create_model

IMG_SIZE = 64
DATA_PATH = "Data Set /archive/leapGestRecog"

X = []
y = []

label_map = {}
label_id = 0

# Check if DATA_PATH exists
if not os.path.exists(DATA_PATH):
    print(f"Error: dataset path {DATA_PATH} not found.")
    exit(1)

print("Loading data...")
# Iterate through subject folders (00, 01, ...)
for subject in sorted(os.listdir(DATA_PATH)):
    # Skip non-digit directories (like 'leapGestRecog' or other metadata)
    if not subject.isdigit(): continue
    
    subject_path = os.path.join(DATA_PATH, subject)
    if not os.path.isdir(subject_path): continue

    # Iterate through gesture folders (01_palm, 02_l, ...)
    for gesture in sorted(os.listdir(subject_path)):
        # Skip hidden files
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
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image)
            y.append(label_map[gesture])

num_classes = len(label_map)
print(f"Detected {num_classes} classes: {list(label_map.keys())}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

num_classes = len(label_map)
model = create_model((IMG_SIZE, IMG_SIZE, 1), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
# Reduced epochs for verification
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")