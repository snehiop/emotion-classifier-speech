import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from keras_self_attention import SeqSelfAttention

# Load the trained model
model = load_model("final_emotion_model.h5", custom_objects={'SeqSelfAttention': SeqSelfAttention})

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emoji_map = {
    'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}

# Extract log-mel spectrogram
def extract_features(file):
    y, sr = librosa.load(file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    if log_mel.shape[1] < 130:
        pad_width = 130 - log_mel.shape[1]
        log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :130]

    return log_mel.T  # (128, 130)

# --- STYLING ---
st.set_page_config(page_title="Emotion Classifier", page_icon="🎧", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 40px;
        text-align: center;
        margin-top: -20px;
        color: #F9A825;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #B0BEC5;
        margin-top: -10px;
        font-size: 16px;
    }
    .result-box {
        background-color: #1A237E;
        padding: 20px;
        border-radius: 12px;
        margin-top: 25px;
        text-align: center;
        font-size: 22px;
        color: #ffffff;
        font-weight: 500;
    }
    .confidence-box {
        background-color: #283593;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 18px;
        color: #ffffff;
        text-align: center;
        font-weight: 400;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.markdown("<div class='title'>🎧 Emotion Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a .wav file to detect the emotion from voice 🎤</div>", unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("", type=["wav"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing..."):
        features = extract_features(uploaded_file)
        features = features.reshape(1, 128, 130)
        prediction = model.predict(features)

        predicted_index = int(np.argmax(prediction))
        predicted_emotion = labels[predicted_index]
        emoji = emoji_map[predicted_emotion]
        confidence = float(np.max(prediction)) * 100

        # --- DISPLAY RESULT ---
        st.markdown(f"<div class='result-box'>🧠 Predicted Emotion: <strong>{emoji} {predicted_emotion.upper()}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-box'>🎯 Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

        # --- PROBABILITY CHART ---
        st.markdown("### 📊 Class Probabilities")
        st.bar_chart({labels[i]: prediction[0][i] for i in range(len(labels))})