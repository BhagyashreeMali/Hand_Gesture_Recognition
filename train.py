import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64
DATA_PATH = "dataset/leapGestRecog"

X = []
y = []

label_map = {}
label_id = 0

for subject in os.listdir(DATA_PATH):
    subject_path = os.path.join(DATA_PATH, subject)

    for gesture in os.listdir(subject_path):
        if gesture not in label_map:
            label_map[gesture] = label_id
            label_id += 1

        gesture_path = os.path.join(subject_path, gesture)

        for img in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image)
            y.append(label_map[gesture])

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("gesture_model.h5")

image = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

prediction = model.predict(image)
gesture_class = np.argmax(prediction)

print("Predicted Gesture Class:", gesture_class)