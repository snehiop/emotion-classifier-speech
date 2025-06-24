import streamlit as st
import librosa
import numpy as np
import logging
from tensorflow.keras.models import load_model
from keras_self_attention import SeqSelfAttention
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Cache the model loading to improve performance
@st.cache_resource
def load_emotion_model():
    """Load the trained emotion classification model with caching."""
    try:
        model = load_model("final_emotion_model.h5", 
                          custom_objects={'SeqSelfAttention': SeqSelfAttention})
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None

# Initialize model and labels
model = load_emotion_model()
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emoji_map = {
    'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}

def extract_features(file):
    """Extract log-mel spectrogram features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(file, sr=22050, duration=30)  # Limit to 30 seconds
        
        # Check if audio is too short
        if len(y) < 1024:
            raise ValueError("Audio file is too short for processing")
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, 
                                           hop_length=512, n_fft=2048)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure consistent dimensions
        target_frames = 130
        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), 
                           mode='constant', constant_values=np.min(log_mel))
        else:
            log_mel = log_mel[:, :target_frames]
        
        # Normalize features
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        return log_mel.T  # Shape: (130, 128)
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise e

def predict_emotion(features):
    """Predict emotion from extracted features."""
    try:
        if model is None:
            raise ValueError("Model not loaded properly")
        
        # Reshape for model input: (batch_size, time_steps, features)
        features = features.reshape(1, 130, 128)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_index = int(np.argmax(prediction))
        predicted_emotion = labels[predicted_index]
        confidence = float(np.max(prediction)) * 100
        
        return predicted_emotion, confidence, prediction[0]
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise e

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
    .error-box {
        background-color: #d32f2f;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 16px;
        color: #ffffff;
        text-align: center;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #F9A825;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<div class='title'>🎧 Emotion Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a .wav file to detect the emotion from voice 🎤</div>", unsafe_allow_html=True)

# Model status check
if model is None:
    st.error("❌ Model failed to load. Please check if 'final_emotion_model.h5' exists in the project directory.")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"], 
                                label_visibility="collapsed")

if uploaded_file is not None:
    try:
        # Display audio player
        st.audio(uploaded_file, format="audio/wav")
        
        # Get file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        with st.expander("📁 File Information"):
            for key, value in file_details.items():
                st.write(f"{key}:** {value}")
        
        # Process the audio file
        with st.spinner("🔄 Analyzing audio... This may take a moment."):
            # Extract features
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            
            features = extract_features(uploaded_file)
            progress_bar.progress(50)
            
            # Make prediction
            predicted_emotion, confidence, probabilities = predict_emotion(features)
            progress_bar.progress(100)
            
            # Clear progress bar
            progress_bar.empty()
        
        # Get emoji for the predicted emotion
        emoji = emoji_map.get(predicted_emotion, '🤔')
        
        # --- DISPLAY RESULTS ---
        st.markdown(f"""<div class='result-box'>
            🧠 Predicted Emotion: <strong>{emoji} {predicted_emotion.upper()}</strong>
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div class='confidence-box'>
            🎯 Confidence: {confidence:.2f}%
        </div>""", unsafe_allow_html=True)
        
        # --- PROBABILITY CHART ---
        st.markdown("### 📊 Emotion Probabilities")
        
        # Create probability data for chart
        prob_data = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
        
        # Display bar chart
        st.bar_chart(prob_data)
        
        # Show detailed probabilities
        with st.expander("🔍 Detailed Probabilities"):
            for i, (label, prob) in enumerate(prob_data.items()):
                percentage = prob * 100
                st.write(f"{emoji_map[label]} {label.capitalize()}:** {percentage:.2f}%")
        
        # Interpretation
        with st.expander("💡 Interpretation"):
            if confidence > 80:
                st.success("High confidence prediction - the model is quite certain about this emotion.")
            elif confidence > 60:
                st.info("Moderate confidence - the prediction is likely correct but consider the context.")
            else:
                st.warning("Low confidence - the emotion might be ambiguous or the audio quality may affect accuracy.")
    
    except Exception as e:
        error_msg = str(e)
        st.markdown(f"""<div class='error-box'>
            ❌ Error processing audio: {error_msg}
        </div>""", unsafe_allow_html=True)
        
        with st.expander("🔧 Troubleshooting Tips"):
            st.write("""
            *Common issues and solutions:*
            - Ensure the audio file is in WAV format
            - Check that the file is not corrupted
            - Audio should be at least 1 second long
            - Try converting the audio to 22kHz sample rate
            - Make sure the file size is reasonable (< 10MB)
            """)
        
        # Log the full error for debugging
        logger.error(f"Full error traceback: {traceback.format_exc()}")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.markdown("### ℹ About")
    st.write("""
    This app uses a deep learning model to classify emotions from audio recordings.
    
    *Supported emotions:*
    - 😐 Neutral
    - 😌 Calm  
    - 😄 Happy
    - 😢 Sad
    - 😠 Angry
    - 😨 Fearful
    - 🤢 Disgust
    - 😲 Surprised
    """)
    
    st.markdown("### 🎤 Recording Tips")
    st.write("""
    - Use clear, high-quality audio
    - Minimize background noise
    - Record for 2-10 seconds
    - Speak naturally and expressively
    """)
    
    st.markdown("### 🔧 Technical Details")
    st.write(f"""
    - Model: Loaded {'✅' if model else '❌'}
    - Feature extraction: Log-mel spectrogram
    - Input shape: (130, 128)
    - Classes: {len(labels)}
    """)