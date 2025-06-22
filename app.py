import streamlit as st
import numpy as np
import librosa
import joblib

# Load model and scaler
model = joblib.load('knn_accent_model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)

        # --- MFCCs ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)

        # --- Pitch (mean only for simplicity in app) ---
        f0 = librosa.yin(y, fmin=50, fmax=500)
        pitch_mean = np.mean(f0)
        pitch_std = np.std(f0)

        # --- ZCR ---
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # --- Spectral Contrast ---
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = contrast.mean(axis=1)

        # Combine features
        features = np.hstack([
            mfccs_mean,
            pitch_mean,
            pitch_std,
            zcr_mean,
            contrast_mean
        ])
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# --- Streamlit UI ---

# --- Set Page Config ---
st.set_page_config(page_title="Accent Classifier 🎙️", page_icon="🧠", layout="centered")

# --- Custom Header ---
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>🎧 Native Language Accent Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>by <b>Mr. Android</b></h4>", unsafe_allow_html=True)

# --- Aim ---
st.markdown("### 🎯 What’s this all about?")
st.markdown("This mini-project uses machine learning (KNN algorithm) to detect your **Nigerian accent** from a short voice sample. Fun, right? 🇳🇬🔍")

# --- Instructions ---
st.markdown("### 📢 How to use this?")
st.markdown("""
1. Hit that sweet **Browse Files** button below 🎵  
2. Upload a `.wav` or `.mp3` audio of **you speaking** (about15 seconds long)  
3. Sit back while my smart model guesses your **native language**  
""")

# --- Upload Section ---
st.markdown("### ⬆️ Upload Your Audio File")
uploaded_file = st.file_uploader("Let’s hear that voice of yours 🎤", type=["wav", "mp3"])


# st.title("Native Language Accent Classifier")

# --- Prediction Logic ---
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("⏳ Analyzing your accent... Please wait.")

    try:
        features = extract_features(uploaded_file)
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]

        # Output section
        st.markdown("### 🧠 The Model Thinks Your Native Language is...")
        st.success(f"🔥 **{prediction.upper()}** 🔥")
        st.balloons()

    except Exception as e:
        st.error("⚠️ Something went wrong while processing the audio. Try another one.")
        st.exception(e)
        

st.markdown("---")
st.markdown("<small style='text-align: center; display: block;'>Made with ❤️ by Mr. Android</small>", unsafe_allow_html=True)
