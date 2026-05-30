import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import tempfile
import os
import requests
from collections import Counter
from io import BytesIO

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# CUSTOM CSS — Dark Studio Theme
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f;
    color: #e8e4d9;
}

.stApp {
    background: #0a0a0f;
}

/* ── Header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    background: linear-gradient(135deg, #f5c542 0%, #ff6b35 50%, #e8e4d9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    color: #6b6659;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Cards ── */
.card {
    background: #13131a;
    border: 1px solid #1f1f2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

.card-accent {
    border-left: 3px solid #f5c542;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #f5c542;
    margin-bottom: 0.6rem;
}

/* ── Prediction badge ── */
.prediction-badge {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f5c542;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.prediction-sub {
    font-size: 0.72rem;
    color: #6b6659;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Model pill ── */
.model-pill {
    display: inline-block;
    background: #1f1f2e;
    border: 1px solid #2a2a3e;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #a09c91;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── Progress bars (metrics) ── */
.metric-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}

.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: #a09c91;
    width: 160px;
    flex-shrink: 0;
}

.metric-bar-bg {
    flex: 1;
    height: 6px;
    background: #1f1f2e;
    border-radius: 3px;
    overflow: hidden;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #f5c542, #ff6b35);
}

.metric-value {
    font-size: 0.72rem;
    color: #f5c542;
    width: 45px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #13131a !important;
    border: 1.5px dashed #2a2a3e !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #f5c542 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6659 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f5c542 !important;
    border-bottom: 2px solid #f5c542 !important;
}

/* ── Plots background ── */
.stPlot > div {
    background: transparent !important;
}

/* ── Divider ── */
hr {
    border-color: #1f1f2e;
    margin: 1.5rem 0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #f5c542 !important;
}

/* ── Alerts ── */
.stAlert {
    background: #13131a !important;
    border: 1px solid #f5c542 !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# MATPLOTLIB STYLE — dark
# ==================================================

plt.style.use("dark_background")
PLOT_BG    = "#13131a"
PLOT_TEXT  = "#e8e4d9"
PLOT_GOLD  = "#f5c542"
PLOT_ORANGE= "#ff6b35"
PLOT_GRID  = "#1f1f2e"

def style_ax(ax, title=""):
    ax.set_facecolor(PLOT_BG)
    ax.figure.patch.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_TEXT, labelsize=8)
    ax.xaxis.label.set_color(PLOT_TEXT)
    ax.yaxis.label.set_color(PLOT_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_GRID)
    if title:
        ax.set_title(title, color=PLOT_TEXT, fontsize=9,
                     fontweight="bold", pad=8, loc="left")

# ==================================================
# LOAD MODELS  (ajusta las rutas a tu repo/local)
# ==================================================

GITHUB_BASE = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/"

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Carga modelos desde disco (si existen) o desde GitHub.
    Ajusta MODEL_FILES con tu URL base si los tienes en GitHub.
    """
    model_files = {
        "lr_model":  "lr_model_custom.pkl",
        "rf_model":  "rf_model_custom.pkl",
        "xgb_model": "xgb_model_custom.pkl",
        "scaler":    "scaler_custom.pkl",
        "encoder":   "encoder_custom.pkl",
    }

    loaded = {}
    for key, filename in model_files.items():
        if os.path.exists(filename):
            loaded[key] = joblib.load(filename)
        else:
            # Intenta desde GitHub si no está local
            url = GITHUB_BASE + filename
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                loaded[key] = joblib.load(BytesIO(r.content))
            except Exception as e:
                st.error(f"No se pudo cargar **{filename}**: {e}")
                st.stop()

    return (
        loaded["lr_model"],
        loaded["rf_model"],
        loaded["xgb_model"],
        loaded["scaler"],
        loaded["encoder"],
    )

# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    features = {}

    chroma       = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_stft_mean"] = np.mean(chroma)
    features["chroma_stft_var"]  = np.var(chroma)

    rms          = librosa.feature.rms(y=y)
    features["rms_mean"] = np.mean(rms)
    features["rms_var"]  = np.var(rms)

    sc           = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = np.mean(sc)
    features["spectral_centroid_var"]  = np.var(sc)

    sb           = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(sb)
    features["spectral_bandwidth_var"]  = np.var(sb)

    rolloff      = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_var"]  = np.var(rolloff)

    zcr          = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    features["zero_crossing_rate_var"]  = np.var(zcr)

    harmony      = librosa.effects.harmonic(y)
    features["harmony_mean"] = np.mean(harmony)
    features["harmony_var"]  = np.var(harmony)

    perceptr     = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["perceptr_mean"] = np.mean(perceptr)
    features["perceptr_var"]  = np.var(perceptr)

    tempo, _     = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).item())

    mfccs        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_var"]  = np.var(mfccs[i])

    return features, y, sr

# ==================================================
# PREDICTION (con probabilidades)
# ==================================================

def predict_genre(file_path, lr_model, rf_model, xgb_model, scaler, encoder):
    features, y, sr = extract_features(file_path)

    feat_df = pd.DataFrame([features])
    feat_df = feat_df.reindex(columns=scaler.feature_names_in_, fill_value=0).astype(float)
    scaled  = scaler.transform(feat_df)

    results = {}
    probas  = {}
    classes = encoder.classes_

    # Logistic Regression
    lr_pred     = encoder.inverse_transform(lr_model.predict(scaled))[0]
    lr_proba    = lr_model.predict_proba(scaled)[0]
    results["Logistic Regression"] = lr_pred
    probas["Logistic Regression"]  = dict(zip(classes, lr_proba))

    # Random Forest
    rf_pred     = encoder.inverse_transform(rf_model.predict(feat_df))[0]
    rf_proba    = rf_model.predict_proba(feat_df)[0]
    results["Random Forest"] = rf_pred
    probas["Random Forest"]  = dict(zip(classes, rf_proba))

    # XGBoost
    xgb_pred    = encoder.inverse_transform(xgb_model.predict(feat_df))[0]
    xgb_proba   = xgb_model.predict_proba(feat_df)[0]
    results["XGBoost"] = xgb_pred
    probas["XGBoost"]  = dict(zip(classes, xgb_proba))

    return results, probas, y, sr

# ==================================================
# PLOTS
# ==================================================

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    times = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(times, y, alpha=0.7, color=PLOT_GOLD, linewidth=0)
    ax.fill_between(times, y, alpha=0.3, color=PLOT_ORANGE, linewidth=0)
    ax.set_xlabel("Time (s)", color=PLOT_TEXT, fontsize=8)
    ax.set_ylabel("Amplitude", color=PLOT_TEXT, fontsize=8)
    style_ax(ax, "Waveform")
    fig.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    D   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log",
                                   ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB",
                 label="dB", location="right")
    ax.set_xlabel("Time (s)", color=PLOT_TEXT, fontsize=8)
    ax.set_ylabel("Frequency (Hz)", color=PLOT_TEXT, fontsize=8)
    style_ax(ax, "Mel Spectrogram")
    fig.tight_layout()
    return fig

def plot_model_probas(probas, model_name):
    """Gráfica horizontal de barras para las probabilidades de un modelo."""
    genre_proba = probas[model_name]
    sorted_items = sorted(genre_proba.items(), key=lambda x: x[1], reverse=False)
    genres = [g.capitalize() for g, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]

    colors = []
    for v in values:
        if v == max(values):
            colors.append(PLOT_GOLD)
        elif v >= sorted(values)[-2]:
            colors.append(PLOT_ORANGE)
        else:
            colors.append("#2a2a3e")

    fig, ax = plt.subplots(figsize=(8, max(3, len(genres) * 0.52)))
    bars = ax.barh(genres, values, color=colors, height=0.55,
                   edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color=PLOT_TEXT, fontsize=8, fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=8, color=PLOT_TEXT)
    ax.grid(axis="x", color=PLOT_GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(axis="y", visible=False)
    style_ax(ax, f"{model_name}")
    fig.tight_layout()
    return fig

def plot_ensemble_radar(probas):
    """Promedio de confianza entre modelos por género."""
    all_genres = list(list(probas.values())[0].keys())
    avg_proba  = {g: np.mean([probas[m][g] for m in probas]) for g in all_genres}
    sorted_g   = sorted(avg_proba.items(), key=lambda x: x[1], reverse=True)[:8]
    labels     = [g.capitalize() for g, _ in sorted_g]
    values     = [v * 100 for _, v in sorted_g]

    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(labels))
    gradient_colors = [
        matplotlib.colors.to_rgba(PLOT_GOLD,    alpha=1.0),
        matplotlib.colors.to_rgba(PLOT_ORANGE,  alpha=0.85),
        *[matplotlib.colors.to_rgba("#2a2a3e", alpha=0.7)] * 6
    ]
    bars = ax.bar(x, values, color=gradient_colors[:len(labels)],
                  width=0.6, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8, color=PLOT_TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=PLOT_TEXT)
    ax.set_ylabel("Avg Confidence (%)", fontsize=8, color=PLOT_TEXT)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", color=PLOT_GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    style_ax(ax, "Ensemble — Top Genre Probabilities")
    fig.tight_layout()
    return fig

# ==================================================
# MODEL METRICS (hardcoded — actualiza con los tuyos)
# ==================================================

MODEL_METRICS = {
    "Logistic Regression": {
        "Accuracy":  0.715,
        "F1 Score":  0.708,
        "Precision": 0.720,
        "Recall":    0.715,
    },
    "Random Forest": {
        "Accuracy":  0.730,
        "F1 Score":  0.724,
        "Precision": 0.735,
        "Recall":    0.730,
    },
    "XGBoost": {
        "Accuracy":  0.735,
        "F1 Score":  0.730,
        "Precision": 0.738,
        "Recall":    0.735,
    },
}

# ==================================================
# HEADER
# ==================================================

st.markdown('<div class="hero-title">Music Genre<br>Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Logistic Regression · Random Forest · XGBoost</div>', unsafe_allow_html=True)

# ==================================================
# SIDEBAR — Model Metrics
# ==================================================

with st.sidebar:
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

    colors_sidebar = {"Logistic Regression": "#a09c91",
                      "Random Forest": PLOT_ORANGE,
                      "XGBoost": PLOT_GOLD}

    for model_name, metrics in MODEL_METRICS.items():
        st.markdown(f"""
        <div class="card card-accent" style="border-left-color:{colors_sidebar[model_name]}; margin-bottom:1rem;">
            <div style="font-family:'Syne',sans-serif; font-size:0.78rem; font-weight:700;
                        color:{colors_sidebar[model_name]}; margin-bottom:0.7rem; letter-spacing:0.05em;">
                {model_name}
            </div>
        """, unsafe_allow_html=True)

        for metric, val in metrics.items():
            pct = int(val * 100)
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-label">{metric}</span>
                <div class="metric-bar-bg">
                    <div class="metric-bar-fill" style="width:{pct}%;
                         background:linear-gradient(90deg,{colors_sidebar[model_name]},#ff6b35);"></div>
                </div>
                <span class="metric-value">{val:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# MAIN — File Upload
# ==================================================

col_up, col_info = st.columns([1.6, 1])

with col_up:
    st.markdown('<div class="section-label">Upload Audio</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a WAV file here",
        type=["wav", "mp3", "ogg", "flac"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class="card" style="height:100%;">
        <div class="section-label">How it works</div>
        <div style="font-size:0.72rem; color:#6b6659; line-height:1.8;">
            1 · Upload an audio file (WAV/MP3)<br>
            2 · Features are extracted with Librosa<br>
            3 · Three models vote on the genre<br>
            4 · Results + probabilities are displayed
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# MAIN — Results
# ==================================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    song_name = uploaded_file.name

    # Load models
    with st.spinner("Loading models…"):
        lr_model, rf_model, xgb_model, scaler, encoder = load_models()

    # Predict
    with st.spinner("Extracting features & predicting…"):
        predictions, probas, y, sr = predict_genre(
            audio_path, lr_model, rf_model, xgb_model, scaler, encoder
        )

    # Final vote
    final_pred = Counter(predictions.values()).most_common(1)[0][0]

    # ── Hero result ──────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    col_res, col_models = st.columns([1, 1.6])

    with col_res:
        st.markdown(f"""
        <div class="card card-accent">
            <div class="section-label">Final Prediction (Ensemble Vote)</div>
            <div class="prediction-badge">{final_pred.upper()}</div>
            <div class="prediction-sub" style="margin-top:0.4rem;">{song_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Audio player
        st.audio(audio_path, format="audio/wav")

    with col_models:
        st.markdown('<div class="section-label">Per-Model Predictions</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        model_colors = [("#a09c91", "LR"), (PLOT_ORANGE, "RF"), (PLOT_GOLD, "XGB")]
        for col, (mname, mpred) in zip(cols, predictions.items()):
            color = {"Logistic Regression": "#a09c91",
                     "Random Forest": PLOT_ORANGE,
                     "XGBoost": PLOT_GOLD}[mname]
            col.markdown(f"""
            <div class="card" style="text-align:center; padding:1rem;">
                <div style="font-size:0.62rem; letter-spacing:0.15em; color:{color};
                            text-transform:uppercase; margin-bottom:0.5rem;">{mname}</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.25rem;
                            font-weight:800; color:{color}; text-transform:uppercase;">
                    {mpred}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    tab_audio, tab_proba, tab_ensemble = st.tabs([
        "🎛  Audio Analysis",
        "📊  Model Probabilities",
        "🧩  Ensemble Overview",
    ])

    # ─ Tab 1: Audio ─
    with tab_audio:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Waveform</div>', unsafe_allow_html=True)
            st.pyplot(plot_waveform(y, sr), use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Spectrogram</div>', unsafe_allow_html=True)
            st.pyplot(plot_spectrogram(y, sr), use_container_width=True)

    # ─ Tab 2: Per-model probas ─
    with tab_proba:
        st.markdown('<div class="section-label">Confidence per Genre — each model</div>',
                    unsafe_allow_html=True)
        for model_name in predictions:
            st.pyplot(plot_model_probas(probas, model_name), use_container_width=True)

    # ─ Tab 3: Ensemble ─
    with tab_ensemble:
        st.markdown('<div class="section-label">Average Ensemble Confidence</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_ensemble_radar(probas), use_container_width=True)

        # Top 5 table
        all_genres   = list(list(probas.values())[0].keys())
        avg_proba    = {g: np.mean([probas[m][g] for m in probas]) * 100 for g in all_genres}
        top5         = sorted(avg_proba.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_df      = pd.DataFrame(top5, columns=["Genre", "Avg Confidence (%)"])
        top5_df["Genre"] = top5_df["Genre"].str.capitalize()
        top5_df["Avg Confidence (%)"] = top5_df["Avg Confidence (%)"].map(lambda x: f"{x:.2f}%")
        top5_df.index = range(1, len(top5_df) + 1)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Top 5 Genres</div>',
                    unsafe_allow_html=True)
        st.dataframe(top5_df, use_container_width=True)

    # Cleanup
    os.unlink(audio_path)

else:
    # ── Placeholder state ─────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem; margin-top:1.5rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🎵</div>
        <div style="font-family:'Syne',sans-serif; font-size:1rem; color:#6b6659;
                    letter-spacing:0.05em;">
            Upload an audio file to start the analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
