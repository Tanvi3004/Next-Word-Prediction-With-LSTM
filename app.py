import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Streamlit app setup
st.title("Next Word Prediction With LSTM And Early Stopping")

# Initialize variables
model = None
tokenizer = None
max_sequence_len = None

# Load the model and handle errors
try:
    model = load_model('next_word_lstm.h5')
    max_sequence_len = model.input_shape[1] + 1
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the tokenizer and handle errors
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading the tokenizer: {e}")

# Check if all components are loaded
if not model:
    st.error("Failed to load the model.")
if not tokenizer:
    st.error("Failed to load the tokenizer.")
if max_sequence_len is None:
    st.error("Failed to determine max sequence length from the model.")

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# User interface for input and prediction
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    if model and tokenizer and max_sequence_len is not None:
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word: {next_word}')
    else:
        st.error("Model, tokenizer, or max sequence length not loaded; cannot predict the next word.")
