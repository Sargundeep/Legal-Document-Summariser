import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


# Load your dataset (make sure you have the pandas and keras libraries installed)
df = pd.read_csv('output.csv')

# Preprocessing
# Replace 'text_column' with the column name containing the text data.
X = df['Entire_Content'].values
y = df['old_class'].values

# Load your data and preprocess it
# For this example, let's assume you have 'X' as your text data and 'y' as labels.

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize your text data
max_words = 10000  # Define the maximum number of words to be used in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to have a consistent length
max_sequence_length = 100  # Define the maximum sequence length
X_train_pad = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
np.save('label_encoder_classes.npy', label_encoder.classes_)
# Define your deep learning model
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(np.max(y_train_encoded) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train_encoded, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_encoded)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

model.save("text_model")  # Save your model to a directory

model = keras.models.load_model("text_model")

# Assuming you have some new text data for prediction, you can preprocess it and then use the model for predictions.

# Preprocess new text data (replace 'new_text_data' with your actual data)
new_text_data = ["This is a sample text for prediction.", "Another text to predict."]

# Tokenize and pad the new text data
new_text_sequences = tokenizer.texts_to_sequences(new_text_data)
new_text_pad = pad_sequences(new_text_sequences, maxlen=max_sequence_length)
# Make predictions
predictions = model.predict(new_text_pad)
print(predictions)
# If you want to get the class labels based on the predictions, you can use the label encoder
predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))
print(predicted_labels)

# Print the predictions and predicted labels
for i in range(len(new_text_data)):
    print(f"Text: {new_text_data[i]}")
    print(f"Predicted Class: {predicted_labels[i]}")
    print(f"Class Probabilities: {predictions[i]}")


