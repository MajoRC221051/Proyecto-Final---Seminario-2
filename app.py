import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import yt_dlp
import joblib

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# =====================================================
# LOAD MODELS
# =====================================================

lr_model = joblib.load("lr_model_custom.pkl")
rf_model = joblib.load("rf_model_custom.pkl")
xgb_model = joblib.load("xgb_model_custom.pkl")

scaler = joblib.load("scaler_custom.pkl")
encoder = joblib.load("encoder_custom.pkl")

# accuracies obtenidas en tu notebook

lr_acc = 0.715
rf_acc = 0.730
xgb_acc = 0.735

# =====================================================
# AUDIO FEATURE EXTRACTION
# =====================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        duration=30
    )

    features = {}

    chroma_stft = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)

    rms = librosa.feature.rms(y=y)

    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)

    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    harmony = librosa.effects.harmonic(y)

    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)

    perceptr = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    features['tempo'] = float(
        np.asarray(tempo).item()
    )

    for i in range(20):

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )[i]

        features[f'mfcc{i+1}_mean'] = np.mean(mfcc)
        features[f'mfcc{i+1}_var'] = np.var(mfcc)

    return features

# =====================================================
# YOUTUBE DOWNLOAD
# =====================================================

def download_youtube_audio(url):

    ydl_opts = {

        "format":
            "bestaudio/best",

        "outtmpl":
            "youtube_song",

        "postprocessors": [

            {
                "key":
                    "FFmpegExtractAudio",

                "preferredcodec":
                    "wav"
            }
        ]
    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        ydl.download([url])

    return "youtube_song.wav"

# =====================================================
# PREDICTION
# =====================================================

def predict_song_genre(file_path):

    features = extract_features(
        file_path
    )

    features_df = pd.DataFrame(
        [features]
    )

    rf_features = features_df.reindex(
        columns=rf_model.feature_names_in_,
        fill_value=0
    )

    lr_features = features_df.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    rf_features = rf_features.astype(float)
    lr_features = lr_features.astype(float)

    scaled_features = scaler.transform(
        lr_features
    )

    predictions = {

        "Logistic Regression":
        encoder.inverse_transform(
            lr_model.predict(
                scaled_features
            )
        )[0],

        "Random Forest":
        encoder.inverse_transform(
            rf_model.predict(
                rf_features
            )
        )[0],

        "XGBoost":
        encoder.inverse_transform(
            xgb_model.predict(
                rf_features
            )
        )[0]
    }

    weights = {

        "Logistic Regression": lr_acc,
        "Random Forest": rf_acc,
        "XGBoost": xgb_acc
    }

    genre_scores = {}

    for model, genre in predictions.items():

        genre_scores[genre] = (

            genre_scores.get(
                genre,
                0
            )

            +

            weights[model]
        )

    final_prediction = max(
        genre_scores,
        key=genre_scores.get
    )

    confidence = (

        genre_scores[
            final_prediction
        ]

        /

        sum(
            genre_scores.values()
        )
    )

    return (
        predictions,
        final_prediction,
        confidence
    )

# =====================================================
# UI
# =====================================================

st.title(
    "🎵 Music Genre Classification"
)

tab1, tab2 = st.tabs(
    [
        "Upload Audio",
        "YouTube Link"
    ]
)

song_path = None

# =====================================================
# AUDIO UPLOAD
# =====================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload WAV file",
        type=["wav"]
    )

    if uploaded_file:

        with open(
            "temp.wav",
            "wb"
        ) as f:

            f.write(
                uploaded_file.read()
            )

        song_path = "temp.wav"

# =====================================================
# YOUTUBE
# =====================================================

with tab2:

    youtube_url = st.text_input(
        "Paste YouTube URL"
    )

    if st.button(
        "Analyze Song"
    ):

        with st.spinner(
            "Downloading audio..."
        ):

            song_path = download_youtube_audio(
                youtube_url
            )

# =====================================================
# RUN PREDICTION
# =====================================================

if song_path:

    st.audio(song_path)

    y, sr = librosa.load(
        song_path,
        duration=30
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8,3)
        )

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        ax.set_title(
            "Waveform"
        )

        st.pyplot(fig)

    with col2:

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(y)
            ),
            ref=np.max
        )

        fig, ax = plt.subplots(
            figsize=(8,3)
        )

        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='log',
            cmap='magma',
            ax=ax
        )

        ax.set_title(
            "Spectrogram"
        )

        st.pyplot(fig)

    predictions, final_prediction, confidence = \
        predict_song_genre(
            song_path
        )

    results_df = pd.DataFrame({

        "Model": [

            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Prediction": [

            predictions[
                "Logistic Regression"
            ],

            predictions[
                "Random Forest"
            ],

            predictions[
                "XGBoost"
            ]
        ],

        "Accuracy": [

            lr_acc,
            rf_acc,
            xgb_acc
        ]
    })

    st.subheader(
        "Model Predictions"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.success(
        f"Final Prediction: {final_prediction.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )
