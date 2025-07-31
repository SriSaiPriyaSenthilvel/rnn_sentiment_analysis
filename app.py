import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Custom style with navbar logo
def set_style():
    st.markdown("""
        <style>
        /* Simulated Navbar with logo */
        .navbar-logo {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 9999;
        }

        .stApp {
            padding-top: 60px;
            background-color: #1e1e1e;
            color: white !important;
            font-weight: bold !important;
        }

        .title-text {
            color: white !important;
            font-size: 36px;
            font-weight: bold;
            text-shadow: 2px 2px 5px black;
        }

        .stTextArea textarea {
            color: white !important;
            font-weight: bold !important;
            background-color: rgba(0,0,0,0.6);
        }

        .stButton>button {
            font-weight: bold !important;
        }

        .stSuccess, .stInfo {
            color: white !important;
            font-weight: bold !important;
        }
        </style>

        <!-- Navbar Logo -->
        <div class="navbar-logo">
            <img src="logos.png" width="50">
        </div>
        """, unsafe_allow_html=True)

# Load tokenizer and model
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_LEN = 200

# Set style with navbar
set_style()

# Title
st.markdown('<h1 class="title-text">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Input box
st.markdown('<label style="color:white; font-weight:bold; font-size:18px;">📝 Enter a movie review:</label>', unsafe_allow_html=True)
user_input = st.text_area(label="", height=150)

# Prediction
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        prediction = model.predict(padded)[0][0]

        sentiment = "🌟 Positive 😊" if prediction >= 0.5 else "💔 Negative 😞"
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.info(f"🧠 Model Confidence: `{prediction:.2f}`")

# Footer
st.markdown("---")
st.markdown("© 2025 Movie Sentiment AI | Built with ❤️ using Streamlit", unsafe_allow_html=True)
