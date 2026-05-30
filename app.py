import streamlit as st
import pandas as pd
import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    lr_model = joblib.load(
        "lr_model_custom.pkl"
    )

    rf_model = joblib.load(
        "rf_model_custom.pkl"
    )

    xgb_model = joblib.load(
        "xgb_model_custom.pkl"
    )

    scaler = joblib.load(
        "scaler_custom.pkl"
    )

    encoder = joblib.load(
        "encoder_custom.pkl"
    )

    return (
        lr_model,
        rf_model,
        xgb_model,
        scaler,
        encoder
    )

(
    lr_model,
    rf_model,
    xgb_model,
    scaler,
    encoder
) = load_models()

# =====================================================
# MODEL ACCURACIES
# =====================================================

lr_acc = 0.715
rf_acc = 0.730
xgb_acc = 0.735

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.hero {

    background: linear-gradient(
        135deg,
        #1a1a2e 0%,
        #16213e 50%,
        #0f3460 100%
    );

    border-radius: 20px;

    padding: 35px;

    margin-bottom: 25px;
}

.hero-title {

    color: white;

    font-size: 2.5rem;

    font-weight: 800;
}

.hero-sub {

    color: #bfc7d5;

    font-size: 1rem;
}

.metric-card {

    background-color: #161b22;

    border-radius: 15px;

    padding: 20px;

    text-align: center;
}

.metric-value {

    font-size: 2rem;

    font-weight: bold;

    color: #e94560;
}

.metric-label {

    color: #9ca3af;

    font-size: .8rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FEATURE EXTRACTION
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

    return (
        predictions,
        rf_features,
        scaled_features
    )

# =====================================================
# ENSEMBLE
# =====================================================

def ensemble_prediction(predictions):

    weights = {

        "Logistic Regression": lr_acc,
        "Random Forest": rf_acc,
        "XGBoost": xgb_acc
    }

    scores = {}

    for model, genre in predictions.items():

        scores[genre] = (

            scores.get(
                genre,
                0
            )

            +

            weights[model]
        )

    final_prediction = max(
        scores,
        key=scores.get
    )

    confidence = (

        scores[
            final_prediction
        ]

        /

        sum(
            scores.values()
        )
    )

    return (
        final_prediction,
        confidence,
        scores
    )

# =====================================================
# HERO
# =====================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎵 Music Genre Classification
        </div>

        <div class="hero-sub">

            Upload a WAV file and classify
            its musical genre using

            Logistic Regression,
            Random Forest
            and XGBoost.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MODEL METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {lr_acc:.3f}
            </div>

            <div class="metric-label">
                Logistic Regression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {rf_acc:.3f}
            </div>

            <div class="metric-label">
                Random Forest
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-value">
                {xgb_acc:.3f}
            </div>

            <div class="metric-label">
                XGBoost
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# AUDIO UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# =====================================================
# AUDIO PROCESSING
# =====================================================

if uploaded_file:

    with open(
        "temp.wav",
        "wb"
    ) as f:

        f.write(
            uploaded_file.read()
        )

    song_path = "temp.wav"

    st.subheader(
        "🎧 Audio Player"
    )

    st.audio(
        song_path
    )

    y, sr = librosa.load(
        song_path,
        duration=30
    )

    # =====================================
    # WAVEFORM + SPECTROGRAM
    # =====================================

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

    # =====================================
    # PREDICTIONS
    # =====================================

    predictions, rf_features, scaled_features = \
        predict_song_genre(
            song_path
        )

    final_prediction, confidence, scores = \
        ensemble_prediction(
            predictions
        )

    # =====================================
    # RESULTS TABLE
    # =====================================

    st.subheader(
        "🤖 Model Predictions"
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

            round(
                lr_acc,
                3
            ),

            round(
                rf_acc,
                3
            ),

            round(
                xgb_acc,
                3
            )
        ]
    })

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # =====================================
    # FINAL PREDICTION
    # =====================================

    st.success(
        f"🏆 Final Prediction: {final_prediction.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )

    # =====================================
    # GENRE PROBABILITIES
    # =====================================

    rf_prob = rf_model.predict_proba(
        rf_features
    )[0]

    xgb_prob = xgb_model.predict_proba(
        rf_features
    )[0]

    lr_prob = lr_model.predict_proba(
        scaled_features
    )[0]

    total_weight = (
        rf_acc +
        xgb_acc +
        lr_acc
    )

    genre_probs = {}

    for genre, rf_p, xgb_p, lr_p in zip(

        encoder.classes_,

        rf_prob,

        xgb_prob,

        lr_prob
    ):

        genre_probs[genre] = (

            rf_p * rf_acc +

            xgb_p * xgb_acc +

            lr_p * lr_acc

        ) / total_weight

    prob_df = pd.DataFrame({

        "Genre":
            list(
                genre_probs.keys()
            ),

        "Probability":
            list(
                genre_probs.values()
            )
    })

    prob_df = prob_df.sort_values(

        "Probability",

        ascending=False
    )

    # =====================================
    # PROBABILITY CHART
    # =====================================

    st.subheader(
        "📊 Genre Probability Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.barplot(

        data=prob_df,

        x="Probability",

        y="Genre",

        palette="magma",

        ax=ax
    )

    ax.set_title(

        "Genre Probability Distribution",

        fontsize=16,

        fontweight='bold'
    )

    ax.set_xlabel(
        "Probability"
    )

    ax.set_ylabel(
        "Genre"
    )

    st.pyplot(fig)

    # =====================================
    # TOP 3 GENRES
    # =====================================

    st.subheader(
        "🎯 Top 3 Genres"
    )

    top3 = prob_df.head(3)

    col1, col2, col3 = st.columns(3)

    for col, (_, row) in zip(

        [col1, col2, col3],

        top3.iterrows()
    ):

        with col:

            st.metric(

                row["Genre"].upper(),

                f"{row['Probability']:.2%}"
            )

    # =====================================
    # RAW SCORES
    # =====================================

    with st.expander(
        "Show Ensemble Scores"
    ):

        st.json(
            scores
        )
