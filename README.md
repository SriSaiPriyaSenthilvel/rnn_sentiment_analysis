# Objective
The objective of this project is to develop a web-based application that analyzes the sentiment of movie reviews using deep learning techniques. The model is trained to classify text inputs into positive or negative sentiments, offering a user-friendly interface for real-time prediction.

## Model Overview
This project uses a Recurrent Neural Network (RNN) with LSTM (Long Short-Term Memory) layers, which are particularly effective for sequence-based data like natural language.

## Model Architecture:

Embedding Layer: Converts input words to dense vectors (size: 128)

LSTM Layer: Captures sequential dependencies with 128 units

Dense Layer: 128 units with ReLU activation

Output Layer: 1 unit with sigmoid activation (for binary classification)

## Training Specifications:

Vocabulary Size: 10,000 most frequent words

Input Length: 200 tokens

Loss Function: Binary Crossentropy

Optimizer: Adam

## Application Interface
The web app is built using Streamlit, providing a minimal yet interactive interface for sentiment prediction.

## Key Features:
Input any movie review in a text box

Real-time sentiment prediction (Positive or Negative)

Displays model confidence score

Sidebar with logo and navigation

## Screenshot

<img width="1852" height="906" alt="final_sentimentanalysis1" src="https://github.com/user-attachments/assets/30199592-cbac-44ad-ab4b-18070dae7380" />
<img width="1851" height="902" alt="final_sentimentanalysis2" src="https://github.com/user-attachments/assets/3932c74b-cda5-4119-bbf3-ea4dad2bc76e" />
