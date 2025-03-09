import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import time
from skimage.transform import resize 
import soundfile as sf
import io

st.set_page_config(page_title="Melody Mosaic", page_icon="🎶", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;  
        }
        .title-header {
            color: #cce6ff;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin: 20px 0;
        }
    </style>
    <h1 class='title-header'>🎶 Melody Mosaic 🎶</h1>
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>🎶 Unlock the genre of your favourite tracks with AI-powered analysis! 🎶</h3>", unsafe_allow_html=True)
st.write("---")

# Load model
@st.cache_resource
def load_model():
    st.info("🔄 Loading model...")
    return tf.keras.models.load_model("Trained_model (2).keras")

model = load_model()

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Preprocessing function
def load_and_preprocess_file(file_data, sample_rate, target_shape=(140, 140)):
    data = []
    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(file_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = min(start + chunk_samples, len(file_data))
        chunk = file_data[start:end]

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        data.append(mel_spectrogram)

    return np.array(data)

# Prediction function
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)

    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]

    return max_elements[0]

# File uploader
uploaded_file = st.file_uploader("Upload a music file (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    # Load file correctly for librosa
    if uploaded_file.name.endswith('.mp3'):
        # Convert MP3 to WAV using librosa
        audio_data, sample_rate = librosa.load(io.BytesIO(uploaded_file.getvalue()), sr=None)
        temp_wav_path = "temp_audio.wav"
        sf.write(temp_wav_path, audio_data, sample_rate)
    else:
        audio_data, sample_rate = librosa.load(io.BytesIO(uploaded_file.getvalue()), sr=None)

    with st.spinner('🔍 Analyzing your track...'):
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent + 1)

        time.sleep(1)

    # Preprocess the uploaded file
    X_Test = load_and_preprocess_file(audio_data, sample_rate)

    # Predict genre
    c_index = model_prediction(X_Test)
    predicted_genre = classes[c_index]
    st.success(f"🎶 **Predicted Genre:** {predicted_genre}")
    st.write("---")

