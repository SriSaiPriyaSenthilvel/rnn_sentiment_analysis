import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Sidebar with logo and navigation
with st.sidebar:
    st.image("logos.png", width=800)
    st.title("Navigation")
    st.markdown("Use this app to analyze sentiment in movie reviews!")

# Constants
VOCAB_SIZE = 10000   # use the same as during training
EMBEDDING_DIM = 128
MAX_LEN = 200

# ✅ Rebuild the model architecture (same as training)
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# ✅ Load weights
try:
    model.load_weights('model (1).h5')
except Exception as e:
    st.error(f"❌ Could not load model weights: {e}")
    st.stop()

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
