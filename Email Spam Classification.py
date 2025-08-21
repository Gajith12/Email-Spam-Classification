# -*- coding: utf-8 -*-

# Import Necessary Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Initial Setup ---

# Mount Google Drive to access the dataset

drive.mount('/content/drive')

# Download NLTK stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Load and Preprocess the Dataset ---

file_path = '/content/drive/MyDrive/Project-2-v2.0/emails.csv'

# Load the dataset using pandas
print("--- Loading and Inspecting Data ---")
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please make sure the file path is correct and the file is uploaded to your Google Drive.")
    # Exit if the file can't be loaded
    exit()

# Drop the 'Unnamed: 0' column if it exists, as it's just an index.
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Use the existing 'spam' column as the numerical label
df = df.rename(columns={'spam': 'label_binary'})

# Define a function for text preprocessing
def preprocess_text(text):
    """
    Cleans the input text by:
    1. Removing non-alphabetic characters.
    2. Converting text to lowercase.
    3. Removing English stopwords.
    """
    # Remove all non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

print("\n--- Preprocessing Text Data ---")
# Apply the preprocessing function to the 'text' column
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Text data has been cleaned.")
print("Example of original vs. cleaned text:")
print("Original:", df['text'][0])
print("Cleaned:", df['cleaned_text'][0])

# --- Feature Extraction (TF-IDF Vectorization) ---

print("\n--- Performing TF-IDF Vectorization ---")

tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 features
X = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label_binary'].values

# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split complete. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# --- Utility Function for Evaluation ---
def evaluate_model(y_true, y_pred, model_name):
    """
    Calculates and prints evaluation metrics and plots a confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham (0)', 'Spam (1)'],
                yticklabels=['Ham (0)', 'Spam (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# --- Multinomial Naive Bayes (A Classic Choice for Text) ---

print("\n--- Training Model 1: Multinomial Naive Bayes ---")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test)

# Evaluate the model
evaluate_model(y_test, y_pred_nb, "Multinomial Naive Bayes")


# --- Simple Neural Network (Keras/TensorFlow) ---
# This model aligns with the .h5 deliverable.

print("\n--- Training Model 2: Simple Neural Network ---")

# Define the model architecture
nn_model = Sequential([
    # Input layer with dropout for regularization
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    # Hidden layer
    Dense(64, activation='relu'),
    Dropout(0.5),
    # Output layer: uses 'sigmoid' for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
nn_model.summary()

# Train the model
# Using a validation split to monitor performance on unseen data during training
history = nn_model.fit(X_train, y_train,
                       epochs=10,
                       batch_size=32,
                       validation_split=0.1,
                       verbose=1)

# Make predictions
# The model outputs probabilities, so we round them to get 0 or 1
y_prob_nn = nn_model.predict(X_test)
y_pred_nn = (y_prob_nn > 0.5).astype(int)

# Evaluate the neural network model
evaluate_model(y_test, y_pred_nn, "Simple Neural Network")


# --- Visualize Training History & Save Model ---

print("\n--- Visualizing NN Training History and Saving Model ---")
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Save the trained neural network model file
model_save_path = '/content/drive/MyDrive/Project-2-v2.0/Colab_Projects/spam_classifier_model.h5'
nn_model.save(model_save_path)
print(f"\n✅ Neural Network model saved successfully to: {model_save_path}")
