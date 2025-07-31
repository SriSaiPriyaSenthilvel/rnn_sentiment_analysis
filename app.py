import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Constants
VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
MAX_LEN = 200

# Sidebar
with st.sidebar:
    st.image("logos.png", width=800)
    st.title("Navigation")
    st.markdown("Use this app to analyze sentiment in movie reviews!")

# ✅ Recreate architecture (must match training)
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# ✅ Load weights only (you saved it this way)
try:
    model.load_weights("model (1).h5")
except Exception as e:
    st.error(f"❌ Could not load model weights: {e}")
    st.stop()

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# App UI
st.markdown('<h1 style="font-size: 36px; font-weight: bold;">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.subheader("Enter a movie review to predict its sentiment:")

user_input = st.text_area("Your Review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
        prediction = model.predict(padded)[0][0]
        sentiment = "🌟 Positive 😊" if prediction >= 0.5 else "💔 Negative 😞"
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.info(f"🧠 Model Confidence: `{prediction:.2f}`")

st.markdown("---")
st.markdown("© 2025 Movie Sentiment AI | Built with ❤️ using Streamlit", unsafe_allow_html=True)
