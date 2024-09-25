
# Gesture Classification Using Keras
## Project Overview
This project focuses on classifying hand gestures using a Convolutional Neural Network (CNN) built with Keras. The objective is to correctly identify various hand gestures from a dataset of images. This type of classification can be useful in applications such as gesture-based control systems, sign language recognition, and human-computer interaction.
### Dataset
The dataset consists of images representing different hand gestures. These images are categorized into distinct gesture classes such as:
Fist
Open Hand
Thumbs Up
Thumbs Down
Victory (Peace Sign)
Each image is preprocessed and resized before being fed into the neural network.
# Project Structure
├── dataset/
│   ├── train/
│   │   ├── fist/
│   │   ├── open_hand/
│   │   ├── thumbs_up/
│   │   ├── thumbs_down/
│   │   └── victory/
│   ├── test/
├── models/
│   ├── gesture_cnn_model.h5
├── notebooks/
│   └── gesture_classification.ipynb
├── README.md
└── requirements.txt

dataset/: Contains the hand gesture images, split into training and testing sets.
models/: Stores the trained CNN model.
notebooks/: Jupyter notebook containing the Keras model implementation.
requirements.txt: Lists the dependencies required to run the project.
## Preprocessing
The following preprocessing steps are performed on the images:

Resizing: All images are resized to a fixed shape of (64, 64, 3).
Normalization: Pixel values are normalized to a range of [0, 1].
One-Hot Encoding: Labels are one-hot encoded for multi-class classification.
## Model Architecture
The CNN model is built using Keras and consists of the following layers:

Convolutional Layers: Extract features from the images using filters.
Max Pooling Layers: Reduce the spatial dimensions to lower the computational complexity.
Flattening Layer: Convert the 2D feature maps into a 1D vector.
Fully Connected Layers (Dense): Perform the classification task.
Dropout Layer: Prevent overfitting by randomly setting a fraction of input units to 0 during training.
## Model Summary:
Input: (64, 64, 3)
Conv2D + MaxPooling: Multiple convolutional layers followed by max pooling.
Flatten: Converts 2D feature maps to 1D.
Dense + Dropout: Fully connected layers with dropout.
Output Layer: Softmax activation for multi-class classification.
## Model Training
The model is compiled using:

Loss Function: categorical_crossentropy for multi-class classification.
Optimizer: adam optimizer for efficient gradient descent.
Metrics: Accuracy to monitor the model’s performance.
The model is trained on the training dataset with validation using the test dataset.

## Evaluation
The model is evaluated using:

Accuracy: The percentage of correctly classified gestures.
Confusion Matrix: Visual representation of model predictions vs actual labels.
Loss and Accuracy Curves: Plots showing the training and validation accuracy over time.
Results
After training, the model achieves an accuracy of 99% on the test dataset. The confusion matrix shows the classification performance across each gesture class.
