# Hand Gesture Recognition

This project implements a Deep Learning model to recognize hand gestures using a Convolutional Neural Network (CNN).

## Project Overview

The project uses a CNN-based approach to classify hand gestures from the Leap GestRecog dataset. It includes scripts for model definition, training, and testing.

## Files Description

- `model.py`: Defines the CNN architecture using Keras.
- `train.py`: Script to load data, preprocess images, and train the model.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: System and large data files excluded from the repository.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/BhagyashreeMali/Hand_Gesture_Recognition.git
   cd Hand_Gesture_Recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

> [!IMPORTANT]
> The dataset is not included in this repository due to its large size (2.2GB).
> You can download the dataset from [Kaggle: Leap GestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) and place it in a folder named `Data Set /` in the project root.

## Usage

### Training

To train the model, run:

```bash
python train.py
```

### Testing

To test the model on an image:

```bash
python train.py # Currently contains testing logic as well
```

## Future Scope

- Real-time gesture recognition using webcam.
- Improving accuracy with deeper architectures.
- Adding more gesture classes.
