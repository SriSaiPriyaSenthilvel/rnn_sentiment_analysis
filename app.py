import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Sidebar with logo and navigation
with st.sidebar:
    st.image("logos.png", width=800)  # Make sure 'logo.png' is in your working directory
    st.title("Navigation")
    st.markdown("Use this app to analyze sentiment in movie reviews!")

# Load model and tokenizer
model = tf.keras.models.load_model('model.h5', compile=False)

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_LEN = 200

# App title and subheader
st.markdown('<h1 style="font-size: 36px; font-weight: bold;">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.subheader("Enter a movie review to predict its sentiment:")

# Text input
user_input = st.text_area("Your Review:")

# Sentiment prediction
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
        prediction = model.predict(padded)[0][0]
        sentiment = "🌟 Positive 😊" if prediction >= 0.5 else "💔 Negative 😞"

        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.info(f"🧠 Model Confidence: `{prediction:.2f}`")

# Footer
st.markdown("---")
st.markdown("© 2025 Movie Sentiment AI | Built with ❤️ using Streamlit", unsafe_allow_html=True)
