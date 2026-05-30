import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
import requests
from collections import Counter
from io import BytesIO


st.set_page_config(
    page_title="🎵 Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Style
st.markdown("""
<style>
    .pred-box {
        background: #f0fdf4;
        border: 2px solid #1DB954;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        color: #166534;
        margin-bottom: 1rem;
    }
    .model-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .model-card .genre { font-size: 1.1rem; font-weight: 600; }
    .model-card .name  { font-size: 0.75rem; color: #6c757d; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS





@st.cache_resource(show_spinner="Cargando modelos...")
def load_models():
    files = {
        "lr":      "lr_model_custom.pkl",
        "rf":      "rf_model_custom.pkl",
        "xgb":     "xgb_model_custom.pkl",
        "scaler":  "scaler_custom.pkl",
        "encoder": "encoder_custom.pkl",
    }
    loaded = {}
    for key, fname in files.items():
        if os.path.exists(fname):
            loaded[key] = joblib.load(fname)
        else:
            try:
                r = requests.get(GITHUB_BASE + fname, timeout=30)
                r.raise_for_status()
                loaded[key] = joblib.load(BytesIO(r.content))
            except Exception as e:
                st.error(f"No se pudo cargar {fname}: {e}")
                st.stop()
    return loaded["lr"], loaded["rf"], loaded["xgb"], loaded["scaler"], loaded["encoder"]

# Feature Extraction

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    features = {}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_stft_mean"] = np.mean(chroma)
    features["chroma_stft_var"]  = np.var(chroma)

    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = np.mean(rms)
    features["rms_var"]  = np.var(rms)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = np.mean(sc)
    features["spectral_centroid_var"]  = np.var(sc)

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(sb)
    features["spectral_bandwidth_var"]  = np.var(sb)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_var"]  = np.var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    features["zero_crossing_rate_var"]  = np.var(zcr)

    harmony = librosa.effects.harmonic(y)
    features["harmony_mean"] = np.mean(harmony)
    features["harmony_var"]  = np.var(harmony)

    perceptr = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["perceptr_mean"] = np.mean(perceptr)
    features["perceptr_var"]  = np.var(perceptr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).item())

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_var"]  = np.var(mfccs[i])

    return features, y, sr

# Prediction

def predict_genre(file_path, lr_model, rf_model, xgb_model, scaler, encoder):
    features, y, sr = extract_features(file_path)

    feat_df = pd.DataFrame([features])
    feat_df = feat_df.reindex(columns=scaler.feature_names_in_, fill_value=0).astype(float)
    scaled  = scaler.transform(feat_df)

    classes = encoder.classes_
    results, probas = {}, {}

    results["Logistic Regression"] = encoder.inverse_transform(lr_model.predict(scaled))[0]
    probas["Logistic Regression"]  = dict(zip(classes, lr_model.predict_proba(scaled)[0]))

    results["Random Forest"] = encoder.inverse_transform(rf_model.predict(feat_df))[0]
    probas["Random Forest"]  = dict(zip(classes, rf_model.predict_proba(feat_df)[0]))

    results["XGBoost"] = encoder.inverse_transform(xgb_model.predict(feat_df))[0]
    probas["XGBoost"]  = dict(zip(classes, xgb_model.predict_proba(feat_df)[0]))

    return results, probas, y, sr

# Header

st.title(" ⋆˚࿔ Music Genre Classifier")
st.write("Sube un audio y los modelos predicen el género musical.")
st.divider()

# Upload audio

uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is None:
    st.info("⬆ Sube un audio para empezar")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded_file.read())
    audio_path = tmp.name

st.audio(audio_path)
st.divider()



lr_model, rf_model, xgb_model, scaler, encoder = load_models()

with st.spinner("Analizando audio..."):
    predictions, probas, y, sr = predict_genre(
        audio_path, lr_model, rf_model, xgb_model, scaler, encoder
    )

final_pred = Counter(predictions.values()).most_common(1)[0][0]

# Final results

st.subheader("‧₊ ♪˚⊹ Resultado final")
st.markdown(
    f'<div class="pred-box">🎶 {final_pred.upper()}</div>',
    unsafe_allow_html=True
)
st.caption("Votación mayoritaria entre los 3 modelos")

st.divider()

# Pred models
st.subheader("⋆✴︎˚｡⋆ Predicción por modelo")

col1, col2, col3 = st.columns(3)
for col, (model_name, pred) in zip([col1, col2, col3], predictions.items()):
    with col:
        st.markdown(f"""
        <div class="model-card">
            <div class="name">{model_name}</div>
            <div class="genre">🎵 {pred.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
# Prob graph

st.subheader(".☘︎ ݁˖ Confianza por género")

tab1, tab2, tab3 = st.tabs(list(predictions.keys()))

for tab, model_name in zip([tab1, tab2, tab3], predictions.keys()):
    with tab:
        genre_proba = probas[model_name]
        sorted_items = sorted(genre_proba.items(), key=lambda x: x[1], reverse=True)
        genres = [g.capitalize() for g, _ in sorted_items]
        values = [v * 100 for _, v in sorted_items]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        bar_colors = ["#1DB954" if i == 0 else "#d0d0d0" for i in range(len(genres))]
        ax.bar(genres, values, color=bar_colors, width=0.6)

        for i, (g, v) in enumerate(zip(genres, values)):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylabel("Confianza (%)")
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(f"{model_name}", fontsize=10, pad=8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.divider()

# WaveForm

st.subheader(".ೀ Visualización del audio")

col_w, col_s = st.columns(2)

with col_w:
    st.markdown("**Waveform**")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1DB954", alpha=0.8)
    ax.set_xlabel("Tiempo (s)", fontsize=8)
    ax.set_ylabel("Amplitud", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_s:
    st.markdown("**Espectrograma**")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_xlabel("Tiempo (s)", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

# Metrics/Model
st.subheader("📈 Métricas de los modelos .ೀ")

metrics_df = pd.DataFrame({
    "Modelo":    ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy":  [0.715, 0.730, 0.735],
    "F1 Score":  [0.708, 0.724, 0.730],
    "Precision": [0.720, 0.735, 0.738],
    "Recall":    [0.715, 0.730, 0.735],
})

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# cleanup
os.unlink(audio_path)
