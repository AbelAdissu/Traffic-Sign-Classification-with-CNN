# Import necessary libraries and modules
# opendatasets is used to download datasets from Kaggle
# Pandas for data manipulation, NumPy for numerical operations
# TensorFlow and Keras for deep learning
# PIL for image processing, os for file operations
# train_test_split for data splitting, and matplotlib for plotting
!pip install opendatasets
import opendatasets as od
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define the dataset URL
dataset_url = "https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification"

# Download the dataset using opendatasets
od.download(dataset_url, '.')

# Define the base path to your dataset
inputBasePath = './traffic-sign-dataset-classification'
trainingFolder = 'traffic_Data/DATA'
testingFolder = 'traffic_Data/TEST'

# Load the class labels from a CSV file
classes = pd.read_csv(os.path.join(inputBasePath, 'labels.csv'))

# Function to fetch and preprocess image data
def fetch_images(inputBasePath, trainingFolder):
    traffic_data = []  # Initialize an empty list to store image data and labels
    labels = []  # Initialize an empty list to store class labels

    # Loop through each class folder
    for classValue in os.listdir(os.path.join(inputBasePath, trainingFolder)):
        classPath = os.path.join(inputBasePath, trainingFolder, classValue)
        labels.append(classValue)  # Add the class label to the list
        # Loop through each image in the class folder
        for trafficSignal in os.listdir(classPath):
            imgTrafficSignal = Image.open(os.path.join(classPath, trafficSignal))  # Open the image
            imgTrafficSignal = imgTrafficSignal.convert("L")  # Convert to grayscale
            imgTrafficSignal = imgTrafficSignal.resize((90, 90))  # Resize to a fixed size
            imgTrafficSignal = np.array(imgTrafficSignal)  # Convert to NumPy array
            traffic_data.append((imgTrafficSignal, [int(classValue)]))  # Append image and label to the list

    labels = np.array(labels)  # Convert labels to a NumPy array
    return traffic_data, labels

# Fetch image data and labels
training_data, training_labels = fetch_images(inputBasePath, trainingFolder)

# Convert the data to NumPy arrays
training_data_features = np.array([item[0] for item in training_data])
training_data_labels = np.array([item[1] for item in training_data])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    training_data_features, training_data_labels, test_size=0.2, random_state=42
)

# Define the neural network model architecture
model = Sequential()  # Initialize a sequential model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(90, 90, 1)))  # Add a convolutional layer
model.add(MaxPool2D((2, 2)))  # Add max pooling
model.add(Conv2D(64, (3, 3), activation='relu'))  # Add another convolutional layer
model.add(MaxPool2D((2, 2)))  # Add max pooling
model.add(Flatten())  # Flatten the output
model.add(Dense(128, activation='relu'))  # Add a fully connected layer
model.add(Dropout(0.5))  # Add dropout for regularization
model.add(Dense(len(classes), activation='softmax'))  # Output layer with softmax activation

# Compile the model with an optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Plot training history to visualize model performance during training
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()

# Additional evaluation and plotting
# Predict on the validation set
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Generate a classification report
class_report = classification_report(y_val, y_pred, target_names=classes['Name'], output_dict=True)

# Convert the classification report to a DataFrame for better visualization
class_report_df = pd.DataFrame(class_report).transpose()

# Plot the classification report
plt.figure(figsize=(10, 8))
sns.heatmap(class_report_df.iloc[:-1, :3], annot=True, cmap='Blues')
plt.title('Classification Report (Precision, Recall, F1-Score)')

# Display all the plots
plt.show()
