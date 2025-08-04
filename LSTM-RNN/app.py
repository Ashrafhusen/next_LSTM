import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("D:/next_LSTM/LSTM-RNN/next_word_lstm.h5")

with open("D:/next_LSTM/LSTM-RNN/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Could not predict"

# ----------------- Streamlit UI -----------------

st.set_page_config(
    page_title="Next Word Predictor",
    layout="centered",
    page_icon=None
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f4f6f8;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Next Word Prediction")
st.markdown("This app predicts the next word based on your input sequence using an LSTM model trained with Early Stopping.")

st.markdown("---")

with st.container():
    st.subheader("Enter your sentence:")
    input_text = st.text_input("", "To be or not to")

    if st.button("Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1
        with st.spinner("Processing..."):
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success("Prediction complete.")
        st.markdown(f"""
        <div style="text-align:center; padding: 20px; background-color: #e0f7fa; border-radius: 10px;">
            <h3 style="color: #00796b;">Next Word:</h3>
            <h1 style="color: #00796b;">{next_word}</h1>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built using TensorFlow and Streamlit")
