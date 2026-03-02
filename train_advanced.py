"""
Advanced Training Pipeline for Hand Gesture Recognition
Includes: Data Augmentation, BatchNormalization CNN, EarlyStopping,
           ModelCheckpoint, TensorBoard, and Learning Rate Scheduling.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_utils import load_data, get_augmentor
from src.models.advanced_model import create_advanced_cnn, create_transfer_model

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 64
DATA_PATH = "Data Set /archive/leapGestRecog"
MODEL_DIR = "saved_models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_type", type=str, default="advanced_cnn",
                        choices=["advanced_cnn", "transfer_learning"],
                        help="Model architecture to use")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    return parser.parse_args()

def plot_training_history(history, save_path):
    """Plots and saves training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ============================================================
    # 1. Load Data
    # ============================================================
    print("=" * 60)
    print("HAND GESTURE RECOGNITION - ADVANCED TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nModel Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Augmentation: {args.augment}")
    print(f"Learning Rate: {args.lr}")
    print()
    
    print("Loading dataset...")
    X, y, label_map = load_data(DATA_PATH, IMG_SIZE)
    num_classes = len(label_map)
    
    # Save label map for inference
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Detected {num_classes} classes: {list(label_map.keys())}")
    print(f"Total samples: {len(X)}")
    
    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # ============================================================
    # 2. Build Model
    # ============================================================
    print("\nBuilding model...")
    if args.model_type == "advanced_cnn":
        model = create_advanced_cnn((IMG_SIZE, IMG_SIZE, 1), num_classes)
    else:
        # For transfer learning, convert grayscale to 3-channel
        X_train = np.repeat(X_train, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)
        model = create_transfer_model((IMG_SIZE, IMG_SIZE, 3), num_classes)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # ============================================================
    # 3. Callbacks
    # ============================================================
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f"best_{args.model_type}.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]
    
    # ============================================================
    # 4. Train
    # ============================================================
    print("\nStarting training...")
    if args.augment:
        augmentor = get_augmentor()
        history = model.fit(
            augmentor.flow(X_train, y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // args.batch_size
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
    
    # ============================================================
    # 5. Evaluate and Save Results
    # ============================================================
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, f"{args.model_type}_final.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Also save as gesture_model.h5 for backward compatibility
    model.save("gesture_model.h5")
    
    # Generate plots
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    
    # Classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = sorted(label_map, key=label_map.get)
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names,
                          os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Models saved to {MODEL_DIR}/")
    print(f"TensorBoard logs saved to {LOG_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
