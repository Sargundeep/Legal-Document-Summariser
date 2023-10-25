import streamlit as st
import re
from transformers import pipeline
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your TensorFlow SavedModel model
model = load_model("text_model")  # Import the model from models.py
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Define available models
available_models = {
    "bert-base-uncased": "BERT-based Summarization"
}

# Streamlit UI components
st.title("Legal Document Processing")
st.sidebar.header("Settings")

# Add a file uploader to allow users to upload a legal text document
file = st.sidebar.file_uploader("Upload a legal text document", type=["txt", "pdf"])

# Input field for specifying the word limit
word_limit = st.sidebar.number_input("Word Limit for Summarization", min_value=100, step=10)

# Selector for choosing the summarization model
selected_model = "BERT-based Summarization"

# Function to preprocess and summarize the legal document based on word limit and selected model
def summarize_and_classify_legal_document(document, word_limit, selected_model):
    # Create a summarization pipeline with the selected model
    summarizer = pipeline("summarization", model=list(available_models.keys())[list(available_models.values()).index(selected_model)])
    
    # Tokenize the document into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', document)

    summarized_text = ""
    current_word_count = 0

    for sentence in sentences:
        # Check if adding the sentence exceeds the word limit
        if current_word_count + len(sentence.split()) <= word_limit:
            summarized_text += sentence + " "
            current_word_count += len(sentence.split())
        else:
            break  # Stop if the word limit is reached

    # Display the summarized document
    st.subheader("Summarized Document")
    st.write(summarized_text)

    # Predict and display the category
    predicted_category = predict_category(document)
    st.subheader("Predicted Category")
    st.write(predicted_category)

    # Add the document to the output.csv file
    # add_document_to_output(document, predicted_category)

# Function to add the document to the output.csv file
def add_document_to_output(document, predicted_category):
    # Create a DataFrame with the document and predicted category
    df = pd.read_csv('output.csv')
    last_value = df['file_number'].iloc[-1]
    print(last_value)
    document = document.replace("\n","")
    new_data = pd.DataFrame({'file_number':[last_value+1],'old_class': [predicted_category], 'Entire_Content': [document]})

    # # Append the new data to the output.csv file
    output_df = pd.read_csv('output.csv')
    output_df = pd.concat([output_df, new_data], ignore_index=True)

    # # Save the updated DataFrame back to the output.csv file
    output_df.to_csv('output.csv', index=False)

# Function to predict the category of the input text document
def predict_category(text_document):
    # Tokenize and preprocess the text document
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([text_document])

    text_sequences = tokenizer.texts_to_sequences([text_document])
    text_sequences_pad = pad_sequences(text_sequences, maxlen=100)

    # Use your deep learning model to predict the category
    predictions = model.predict(text_sequences_pad)
    predicted_category = label_encoder.inverse_transform(predictions.argmax(axis=1))[0]
    
    return predicted_category

if file:
    document = file.read()
    # Ensure the document is a string
    if isinstance(document, bytes):
        document = document.decode('utf-8')

    if st.sidebar.button("Process Document"):
        summarize_and_classify_legal_document(document, word_limit, selected_model)
