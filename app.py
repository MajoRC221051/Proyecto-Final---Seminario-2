import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tempfile
import yt_dlp
from collections import Counter

# ==================================================

# PAGE CONFIG

# ==================================================

st.set_page_config(
page_title="Music Genre Classifier",
page_icon="🎵",
layout="wide"
)

# ==================================================

# LOAD MODELS

# ==================================================

@st.cache_resource
def load_models():

```
lr_model = joblib.load("lr_model_custom.pkl")
rf_model = joblib.load("rf_model_custom.pkl")
xgb_model = joblib.load("xgb_model_custom.pkl")

scaler = joblib.load("scaler_custom.pkl")
encoder = joblib.load("encoder_custom.pkl")

return (
    lr_model,
    rf_model,
    xgb_model,
    scaler,
    encoder
)
```

(
lr_model,
rf_model,
xgb_model,
scaler,
encoder
) = load_models()

# ==================================================

# FEATURE EXTRACTION

# ==================================================

def extract_features(file_path):

```
y, sr = librosa.load(
    file_path,
    duration=30
)

features = {}

chroma_stft = librosa.feature.chroma_stft(
    y=y,
    sr=sr
)

features["chroma_stft_mean"] = np.mean(chroma_stft)
features["chroma_stft_var"] = np.var(chroma_stft)

rms = librosa.feature.rms(y=y)

features["rms_mean"] = np.mean(rms)
features["rms_var"] = np.var(rms)

spectral_centroid = librosa.feature.spectral_centroid(
    y=y,
    sr=sr
)

features["spectral_centroid_mean"] = np.mean(spectral_centroid)
features["spectral_centroid_var"] = np.var(spectral_centroid)

spectral_bandwidth = librosa.feature.spectral_bandwidth(
    y=y,
    sr=sr
)

features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
features["spectral_bandwidth_var"] = np.var(spectral_bandwidth)

rolloff = librosa.feature.spectral_rolloff(
    y=y,
    sr=sr
)

features["rolloff_mean"] = np.mean(rolloff)
features["rolloff_var"] = np.var(rolloff)

zcr = librosa.feature.zero_crossing_rate(y)

features["zero_crossing_rate_mean"] = np.mean(zcr)
features["zero_crossing_rate_var"] = np.var(zcr)

harmony = librosa.effects.harmonic(y)

features["harmony_mean"] = np.mean(harmony)
features["harmony_var"] = np.var(harmony)

perceptr = librosa.feature.spectral_contrast(
    y=y,
    sr=sr
)

features["perceptr_mean"] = np.mean(perceptr)
features["perceptr_var"] = np.var(perceptr)

tempo, _ = librosa.beat.beat_track(
    y=y,
    sr=sr
)

features["tempo"] = float(
    np.asarray(tempo).item()
)

mfccs = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=20
)

for i in range(20):

    features[f"mfcc{i+1}_mean"] = np.mean(
        mfccs[i]
    )

    features[f"mfcc{i+1}_var"] = np.var(
        mfccs[i]
    )

return features
```

# ==================================================

# PREDICTION

# ==================================================

def predict_song_genre(file_path):

```
features = extract_features(
    file_path
)

features_df = pd.DataFrame(
    [features]
)

features_df = features_df.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)

features_df = features_df.astype(float)

scaled_features = scaler.transform(
    features_df
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
            features_df
        )
    )[0],

    "XGBoost":
    encoder.inverse_transform(
        xgb_model.predict(
            features_df
        )
    )[0]
}

return predictions
```

# ==================================================

# INTERFACE

# ==================================================

st.title("🎵 Music Genre Classification")

st.markdown("""
Upload a WAV file or paste a YouTube URL to predict its genre using:

* Logistic Regression
* Random Forest
* XGBoost
  """)

input_method = st.radio(
"Choose input method",
[
"Upload Audio",
"YouTube URL"
]
)

audio_path = None

# ==================================================

# UPLOAD AUDIO

# ==================================================

if input_method == "Upload Audio":

```
uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        audio_path = tmp_file.name
```

# ==================================================

# YOUTUBE URL

# ==================================================

if input_method == "YouTube URL":

```
youtube_url = st.text_input(
    "Paste YouTube URL"
)

if youtube_url:

    with st.spinner(
        "Downloading audio..."
    ):

        output_file = "youtube_audio.wav"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "youtube_audio",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }],
            "quiet": True
        }

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            ydl.download(
                [youtube_url]
            )

        audio_path = output_file
```

# ==================================================

# RESULTS

# ==================================================

if audio_path is not None:

```
predictions = predict_song_genre(
    audio_path
)

final_prediction = Counter(
    predictions.values()
).most_common(1)[0][0]

st.success(
    f"Final Genre Prediction: {final_prediction.upper()}"
)

results_df = pd.DataFrame({
    "Model": predictions.keys(),
    "Prediction": predictions.values()
})

st.subheader(
    "Model Predictions"
)

st.dataframe(
    results_df,
    use_container_width=True
)

y, sr = librosa.load(
    audio_path,
    duration=30
)

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Waveform"
    )

    fig, ax = plt.subplots(
        figsize=(8,3)
    )

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax
    )

    st.pyplot(fig)

with col2:

    st.subheader(
        "Spectrogram"
    )

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(8,3)
    )

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log",
        ax=ax
    )

    fig.colorbar(
        img,
        ax=ax
    )

    st.pyplot(fig)

st.subheader(
    "Training Accuracy"
)

performance_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        0.715,
        0.730,
        0.735
    ]
})

st.dataframe(
    performance_df,
    use_container_width=True
)
```
