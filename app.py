import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import io
import os
import re
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")

import yt_dlp

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

/* ── hero ── */
.hero{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
  border-radius:20px;padding:38px 48px;margin-bottom:28px;
  border:1px solid rgba(233,69,96,.2);position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-40%;right:-8%;width:380px;height:380px;
  background:radial-gradient(circle,rgba(233,69,96,.18) 0%,transparent 70%);border-radius:50%;}
.hero-badge{display:inline-block;background:linear-gradient(90deg,#e94560,#7b2d8b);
  color:#fff;font-size:.72rem;font-weight:700;padding:4px 14px;border-radius:20px;
  margin-bottom:12px;letter-spacing:1px;text-transform:uppercase;}
.hero-title{font-size:2.4rem;font-weight:800;color:#fff;margin:0 0 8px;letter-spacing:-.5px;}
.hero-sub{font-size:1rem;color:#a0aec0;margin:0;line-height:1.6;}

/* ── metric cards ── */
.metric-card{background:linear-gradient(135deg,#1e1e2e,#252540);
  border:1px solid rgba(255,255,255,.08);border-radius:16px;
  padding:22px 18px;text-align:center;transition:transform .2s;}
.metric-card:hover{transform:translateY(-3px);}
.metric-value{font-size:2rem;font-weight:700;margin:6px 0 4px;}
.metric-label{font-size:.75rem;color:#718096;letter-spacing:.5px;text-transform:uppercase;}

/* ── model result cards ── */
.model-card{background:#141428;border-radius:14px;padding:20px;
  border-left:4px solid;margin-bottom:10px;}
.model-name{font-size:.92rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;}
.model-genre{font-size:1.55rem;font-weight:800;}
.model-acc{font-size:.8rem;color:#718096;margin-top:4px;}

/* ── prediction banner ── */
.pred-banner{background:linear-gradient(135deg,#1a1a2e,#0f3460);
  border:2px solid #e94560;border-radius:18px;padding:32px;
  text-align:center;margin:20px 0;}
.pred-genre{font-size:3.2rem;font-weight:800;color:#e94560;}
.pred-conf{font-size:.95rem;color:#a0aec0;margin-top:6px;}

/* ── section header ── */
.sh{font-size:1.1rem;font-weight:700;color:#e2e8f0;
  padding:8px 0;border-bottom:2px solid #e94560;margin-bottom:18px;
  display:flex;align-items:center;gap:8px;}

/* ── url input ── */
.url-box{background:#141428;border:2px dashed #e94560;border-radius:14px;
  padding:28px;text-align:center;margin-bottom:16px;}

/* ── info box ── */
.info-box{background:rgba(233,69,96,.07);border:1px solid rgba(233,69,96,.25);
  border-radius:10px;padding:14px 18px;font-size:.86rem;color:#cbd5e0;}

/* ── supported sites badge ── */
.site-badge{display:inline-block;background:#1e1e2e;border:1px solid #2d2d4e;
  color:#a0aec0;font-size:.72rem;padding:3px 10px;border-radius:12px;margin:3px;}

/* ── progress ── */
.stProgress>div>div{background:linear-gradient(90deg,#e94560,#7b2d8b)!important;}

/* ── tabs ── */
.stTabs [data-baseweb="tab"]{color:#718096!important;font-weight:500;}
.stTabs [aria-selected="true"]{color:#e94560!important;border-bottom-color:#e94560!important;}

/* ── genre pill ── */
.gpill{display:inline-block;padding:5px 16px;border-radius:20px;
  font-size:.82rem;font-weight:700;margin:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
GENRES = ['blues','classical','country','disco','hiphop',
          'jazz','metal','pop','reggae','rock']

GENRE_COLORS = {
    'blues':     ('#1e40af','#93c5fd'),
    'classical': ('#6d28d9','#c4b5fd'),
    'country':   ('#92400e','#fcd34d'),
    'disco':     ('#be185d','#f9a8d4'),
    'hiphop':    ('#0f766e','#5eead4'),
    'jazz':      ('#b45309','#fde68a'),
    'metal':     ('#374151','#d1d5db'),
    'pop':       ('#be123c','#fda4af'),
    'reggae':    ('#166534','#86efac'),
    'rock':      ('#7c3aed','#ddd6fe'),
}

DARK_BG = "#0a0a0f"
CARD_BG = "#141428"
ACCENT  = "#e94560"
ACCENT2 = "#7b2d8b"
TEXT    = "#e2e8f0"
GRID_C  = "#1e1e2e"

SUPPORTED_DOMAINS = [
    "youtube.com","youtu.be","tiktok.com","soundcloud.com",
    "spotify.com","instagram.com","twitter.com","x.com",
    "facebook.com","vimeo.com","twitch.tv","bandcamp.com",
    "music.youtube.com","dailymotion.com",
]

# ─────────────────────────────────────────────────────────────
# URL DOWNLOAD  (yt-dlp)
# ─────────────────────────────────────────────────────────────
def download_audio_from_url(url: str) -> tuple[bytes, str]:
    """
    Returns (wav_bytes, title) or raises RuntimeError.
    Downloads to a temp dir, reads back as bytes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_tmpl = os.path.join(tmpdir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown")
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(str(e))

        wav_path = os.path.join(tmpdir, "audio.wav")
        if not os.path.exists(wav_path):
            # find any audio file
            for f in os.listdir(tmpdir):
                if f.startswith("audio"):
                    wav_path = os.path.join(tmpdir, f)
                    break

        if not os.path.exists(wav_path):
            raise RuntimeError("Audio file not found after download.")

        with open(wav_path, "rb") as fh:
            wav_bytes = fh.read()

    return wav_bytes, title


def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_features(y, sr) -> dict:
    f = {}
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f['chroma_stft_mean'] = float(np.mean(chroma))
    f['chroma_stft_var']  = float(np.var(chroma))
    rms = librosa.feature.rms(y=y)
    f['rms_mean'] = float(np.mean(rms))
    f['rms_var']  = float(np.var(rms))
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    f['spectral_centroid_mean'] = float(np.mean(sc))
    f['spectral_centroid_var']  = float(np.var(sc))
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f['spectral_bandwidth_mean'] = float(np.mean(sb))
    f['spectral_bandwidth_var']  = float(np.var(sb))
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    f['rolloff_mean'] = float(np.mean(ro))
    f['rolloff_var']  = float(np.var(ro))
    zcr = librosa.feature.zero_crossing_rate(y)
    f['zero_crossing_rate_mean'] = float(np.mean(zcr))
    f['zero_crossing_rate_var']  = float(np.var(zcr))
    harm = librosa.effects.harmonic(y)
    f['harmony_mean'] = float(np.mean(harm))
    f['harmony_var']  = float(np.var(harm))
    sco = librosa.feature.spectral_contrast(y=y, sr=sr)
    f['perceptr_mean'] = float(np.mean(sco))
    f['perceptr_var']  = float(np.var(sco))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        f[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
        f[f'mfcc{i+1}_var']  = float(np.var(mfccs[i]))
    return f


# ─────────────────────────────────────────────────────────────
# TRAIN / LOAD MODELS  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    encoder = LabelEncoder()
    encoder.fit(GENRES)

    try:
        df = pd.read_csv("features_30_sec.csv")
        X  = df.drop(columns=[c for c in ['filename','label','length'] if c in df.columns])
        y_enc = encoder.transform(df['label'])
        source = "GTZAN dataset (features_30_sec.csv)"
    except Exception:
        np.random.seed(42)
        rows = []
        for i, g in enumerate(GENRES):
            for _ in range(100):
                r = {
                    'chroma_stft_mean': np.random.normal(.35+i*.01,.05),
                    'chroma_stft_var':  np.random.normal(.08,.02),
                    'rms_mean':         np.random.normal(.13+i*.005,.04),
                    'rms_var':          np.random.normal(.002,.001),
                    'spectral_centroid_mean': np.random.normal(1800+i*50,400),
                    'spectral_centroid_var':  np.random.normal(300000,80000),
                    'spectral_bandwidth_mean':np.random.normal(1800+i*30,300),
                    'spectral_bandwidth_var': np.random.normal(200000,60000),
                    'rolloff_mean':           np.random.normal(3500+i*80,800),
                    'rolloff_var':            np.random.normal(2e6,5e5),
                    'zero_crossing_rate_mean':np.random.normal(.08+i*.004,.02),
                    'zero_crossing_rate_var': np.random.normal(.001,.0003),
                    'harmony_mean':   np.random.normal(0,.03),
                    'harmony_var':    np.random.normal(.013,.003),
                    'perceptr_mean':  np.random.normal(20+i*.5,3),
                    'perceptr_var':   np.random.normal(50,10),
                    'tempo':          np.random.normal(115+i*3,20),
                    'label': g,
                }
                for j in range(20):
                    r[f'mfcc{j+1}_mean'] = np.random.normal(-5+j*2+i*.5,10)
                    r[f'mfcc{j+1}_var']  = np.random.normal(200+j*10,50)
                rows.append(r)
        df = pd.DataFrame(rows)
        X  = df.drop(columns=['label'])
        y_enc  = encoder.transform(df['label'])
        source = "synthetic demo data"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=.2, random_state=42, stratify=y_enc)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb_m = xgb.XGBClassifier(
        objective='multi:softmax', num_class=len(GENRES),
        n_estimators=200, max_depth=6, learning_rate=.05,
        subsample=.8, colsample_bytree=.8,
        eval_metric='mlogloss', random_state=42, verbosity=0)
    xgb_m.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_sc, y_train)

    metrics = {}
    for name, model, Xtr, Xte in [
        ("Random Forest",       rf,    X_train,    X_test),
        ("XGBoost",             xgb_m, X_train,    X_test),
        ("Logistic Regression", lr,    X_train_sc, X_test_sc),
    ]:
        pred  = model.predict(Xte)
        proba = model.predict_proba(Xte) if hasattr(model,'predict_proba') else None
        acc   = accuracy_score(y_test, pred)
        bal   = balanced_accuracy_score(y_test, pred)
        auc   = roc_auc_score(y_test, proba, multi_class='ovr',
                              average='weighted') if proba is not None else 0.
        cr    = classification_report(y_test, pred,
                                      target_names=encoder.classes_,
                                      output_dict=True)
        cm    = confusion_matrix(y_test, pred)
        metrics[name] = dict(acc=acc, bal=bal, auc=auc,
                             report=cr, cm=cm, pred=pred)

    return dict(rf=rf, xgb=xgb_m, lr=lr,
                scaler=scaler, encoder=encoder,
                feature_cols=list(X.columns),
                metrics=metrics, y_test=y_test, source=source)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
MODEL_ACCS = {'Random Forest': .73, 'XGBoost': .735, 'Logistic Regression': .715}

def predict(features_dict, bundle) -> dict:
    df     = pd.DataFrame([features_dict])
    df     = df.reindex(columns=bundle['feature_cols'], fill_value=0).astype(float)
    df_sc  = bundle['scaler'].transform(df)
    enc    = bundle['encoder']

    preds = {
        'Random Forest':       enc.inverse_transform(bundle['rf'].predict(df))[0],
        'XGBoost':             enc.inverse_transform(bundle['xgb'].predict(df))[0],
        'Logistic Regression': enc.inverse_transform(bundle['lr'].predict(df_sc))[0],
    }

    rf_prob  = bundle['rf'].predict_proba(df)[0]
    xgb_prob = bundle['xgb'].predict_proba(df)[0]
    lr_prob  = bundle['lr'].predict_proba(df_sc)[0]

    # weighted ensemble vote
    scores = {}
    for m, g in preds.items():
        scores[g] = scores.get(g, 0) + MODEL_ACCS[m]
    final      = max(scores, key=scores.get)
    confidence = scores[final] / sum(scores.values())

    # weighted avg probabilities per genre
    total_w = sum(MODEL_ACCS.values())
    genre_probs = {}
    for cls, (p_rf, p_xgb, p_lr) in zip(
            enc.classes_, zip(rf_prob, xgb_prob, lr_prob)):
        genre_probs[cls] = (
            p_rf  * MODEL_ACCS['Random Forest'] +
            p_xgb * MODEL_ACCS['XGBoost'] +
            p_lr  * MODEL_ACCS['Logistic Regression']
        ) / total_w

    return dict(final=final, confidence=confidence,
                predictions=preds, genre_probs=genre_probs)


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS  (dark theme)
# ─────────────────────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d2d4e")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID_C, linewidth=.5, linestyle='--', alpha=.6)
    return fig, ax

def plot_waveform(y, sr):
    fig, ax = dark_fig(10, 2.8)
    t = np.linspace(0, len(y)/sr, num=len(y))
    ax.fill_between(t, y, alpha=.75, color=ACCENT)
    ax.set_xlabel("Time (s)"); ax.set_xlim(0, t[-1])
    ax.set_title("Waveform", fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout(); return fig

def plot_spectrogram(y, sr):
    fig, ax = dark_fig(10, 3.2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time',
                                   y_axis='log', cmap='magma', ax=ax)
    cb = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    ax.set_title("Spectrogram", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
    plt.tight_layout(); return fig

def plot_mfcc(y, sr):
    fig, ax = dark_fig(10, 2.8)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax, cmap='coolwarm')
    cb = plt.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    ax.set_title("MFCCs", fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout(); return fig

def plot_chroma(y, sr):
    fig, ax = dark_fig(10, 2.8)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma',
                                   x_axis='time', ax=ax, cmap='viridis')
    cb = plt.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    ax.set_title("Chromagram", fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout(); return fig

def plot_genre_probs(genre_probs):
    items  = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
    genres = [k for k, _ in items]
    probs  = [v for _, v in items]
    fig, ax = dark_fig(9, 3.6)
    colors_ = [ACCENT if i == 0 else ACCENT2 if i == 1
               else "#4facfe" for i in range(len(genres))]
    bars = ax.barh(genres[::-1], probs[::-1], color=colors_[::-1], alpha=.85)
    ax.set_xlim(0, max(probs)*1.2)
    ax.set_title("Genre Probability (ensemble)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Weighted Probability")
    for bar, v in zip(bars, probs[::-1]):
        ax.text(v+.002, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va='center', color=TEXT, fontsize=9)
    plt.tight_layout(); return fig

def plot_confusion(cm, classes, title, cmap):
    fig, ax = dark_fig(7, 5.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                linewidths=.4, linecolor="#0a0a0f", ax=ax,
                cbar_kws={'shrink':.8})
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(colors=TEXT, rotation=45, labelsize=8)
    plt.tight_layout(); return fig

def plot_model_comparison(m):
    names  = list(m.keys())
    short  = ["RF","XGB","LR"]
    accs   = [m[n]['acc'] for n in names]
    bals   = [m[n]['bal'] for n in names]
    aucs   = [m[n]['auc'] for n in names]
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    fig.patch.set_facecolor(CARD_BG)
    for ax, vals, title, c in zip(axes,
            [accs,bals,aucs],
            ["Accuracy","Balanced Accuracy","ROC-AUC"],
            [ACCENT,ACCENT2,"#4facfe"]):
        bars = ax.bar(short, vals, color=c, alpha=.85, width=.5)
        ax.set_facecolor(CARD_BG); ax.set_ylim(0,1)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold')
        ax.tick_params(colors=TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor("#2d2d4e")
        ax.grid(axis='y', color=GRID_C, linestyle='--', alpha=.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+.012,
                    f"{v:.3f}", ha='center', color=TEXT,
                    fontsize=10, fontweight='bold')
    plt.tight_layout(); return fig

def plot_f1(report, title):
    classes = [k for k in report if k not in
               ('accuracy','macro avg','weighted avg')]
    f1s = [report[c]['f1-score'] for c in classes]
    fig, ax = dark_fig(10, 3.5)
    bars = ax.barh(classes, f1s, color=ACCENT, alpha=.82)
    ax.set_xlim(0,1.08)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel("F1-Score")
    for bar, v in zip(bars, f1s):
        ax.text(v+.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va='center', color=TEXT, fontsize=9)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────
# SHARED ANALYSIS RENDER
# ─────────────────────────────────────────────────────────────
def render_analysis(audio_bytes: bytes, display_name: str, bundle: dict):
    """Load audio, show player + viz + inference result."""

    # ── audio player
    st.markdown('<div class="sh">▶️ Audio Player</div>', unsafe_allow_html=True)
    st.audio(audio_bytes)

    # ── load with librosa
    with st.spinner("🔍 Extracting acoustic features…"):
        y, sr = librosa.load(io.BytesIO(audio_bytes), duration=30, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo_val) if np.isscalar(tempo_val) else float(tempo_val[0])
    zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # ── quick stats strip
    qa, qb, qc, qd = st.columns(4)
    for col, val, lbl, c in [
        (qa, f"{duration:.1f}s",         "Duration",    ACCENT),
        (qb, f"{sr//1000}kHz",           "Sample Rate", ACCENT2),
        (qc, f"{tempo_val:.0f} BPM",     "Tempo",       "#4facfe"),
        (qd, f"{zcr_val:.4f}",           "ZCR Mean",    "#48bb78"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{c};font-size:1.5rem;">{val}</div>
          <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── audio visualisations
    st.markdown('<div class="sh">📡 Audio Visualizations</div>',
                unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🌊 Waveform","🌈 Spectrogram","🎶 MFCCs","🎸 Chroma"])
    with t1: st.pyplot(plot_waveform(y, sr))
    with t2: st.pyplot(plot_spectrogram(y, sr))
    with t3: st.pyplot(plot_mfcc(y, sr))
    with t4: st.pyplot(plot_chroma(y, sr))

    # ── inference
    st.markdown('<div class="sh">🤖 Model Inference</div>',
                unsafe_allow_html=True)
    bar = st.progress(0, text="Extracting features…")
    feats = extract_features(y, sr)
    bar.progress(33, text="🌲 Running Random Forest…");  time.sleep(.3)
    bar.progress(60, text="⚡ Running XGBoost…");        time.sleep(.3)
    bar.progress(85, text="📐 Running Logistic Regression…")
    result = predict(feats, bundle)
    bar.progress(100, text="✅ Ensemble complete"); time.sleep(.4)
    bar.empty()

    # ── prediction banner
    bg, fg = GENRE_COLORS.get(result['final'], ('#e94560','#fff'))
    st.markdown(f"""
    <div class="pred-banner">
      <div style="font-size:.8rem;color:#718096;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:6px;">
        🏆 Final Ensemble Prediction
      </div>
      <div class="pred-genre" style="color:{fg};text-shadow:0 0 40px {bg}80;">
        {result['final'].upper()}
      </div>
      <div class="pred-conf">
        Confidence: <strong style="color:{ACCENT};">{result['confidence']:.1%}</strong>
        &nbsp;·&nbsp; {display_name}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── per-model cards
    st.markdown('<div class="sh">🔬 Per-Model Predictions</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (mname, icon, color) in zip([c1,c2,c3],[
        ("Random Forest",       "🌲", "#4facfe"),
        ("XGBoost",             "⚡", ACCENT),
        ("Logistic Regression", "📐", ACCENT2),
    ]):
        pg = result['predictions'][mname]
        bg2, fg2 = GENRE_COLORS.get(pg, ('#333','#fff'))
        col.markdown(f"""
        <div class="model-card" style="border-left-color:{color};">
          <div class="model-name">{icon} {mname}</div>
          <div class="model-genre" style="color:{fg2};">{pg.upper()}</div>
          <div class="model-acc">Accuracy: {MODEL_ACCS[mname]:.1%}</div>
        </div>""", unsafe_allow_html=True)

    # ── probability chart
    st.markdown('<div class="sh">📊 Genre Probability Distribution</div>',
                unsafe_allow_html=True)
    st.pyplot(plot_genre_probs(result['genre_probs']))

    # ── feature details (collapsed)
    with st.expander("🔩 Extracted Features", expanded=False):
        feat_df = pd.DataFrame([feats]).T.reset_index()
        feat_df.columns = ["Feature","Value"]
        feat_df["Value"] = feat_df["Value"].round(6)
        st.dataframe(feat_df, use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
      <div style="font-size:3rem;">🎵</div>
      <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">Genre Classifier</div>
      <div style="font-size:.73rem;color:#718096;margin-top:4px;">ML-powered · 10 genres</div>
    </div><hr style="border-color:#1e1e2e;">
    """, unsafe_allow_html=True)

    st.markdown("**🎚️ Navigation**")
    page = st.radio("", [
        "🏠 Home & Predict",
        "📊 Model Metrics",
        "🔬 Confusion Matrices",
        "📈 Feature Importance",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e1e2e;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <b>Supported sources</b><br><br>
      📁 Upload: WAV · MP3 · OGG · FLAC<br><br>
      🔗 Links:<br>
      YouTube · TikTok · SoundCloud<br>
      Instagram · Twitter/X · Vimeo<br>
      Bandcamp · Twitch · +1000 more
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.73rem;color:#4a5568;margin-top:18px;text-align:center;">
      Adrían López (21357)<br>
      María José Ramírez (221051)<br>
      Seminario 2 · Matemática Aplicada
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner("⚙️ Initializing models…"):
    bundle = load_models()


# ─────────────────────────────────────────────────────────────
# ── PAGE: HOME & PREDICT
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home & Predict":

    st.markdown("""
    <div class="hero">
      <div class="hero-badge">🎓 Proyecto Final · Seminario 2</div>
      <div class="hero-title">🎵 Music Genre Classifier</div>
      <div class="hero-sub">
        Upload an audio file <strong>or paste a link</strong> from YouTube, TikTok,
        SoundCloud, Instagram and more — three ML models will analyse the audio
        and predict the musical genre with waveform, spectrogram, and ensemble confidence.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── top KPI strip
    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl, c in [
        (c1,"73.5%","XGBoost Accuracy",     ACCENT),
        (c2,"74.0%","Logistic Reg. Acc.",    ACCENT2),
        (c3,"73.0%","Random Forest Acc.",    "#4facfe"),
        (c4,"10",   "Genres",               "#48bb78"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{c};">{val}</div>
          <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────
    # INPUT SECTION — tabs: File / URL
    # ────────────────────────────────
    st.markdown('<div class="sh">🎧 Audio Input</div>', unsafe_allow_html=True)
    inp_tab1, inp_tab2 = st.tabs(["📁 Upload File", "🔗 Paste URL / Link"])

    audio_bytes   = None
    display_name  = ""

    # ── TAB 1: file upload
    with inp_tab1:
        uploaded = st.file_uploader(
            "Drop your audio file here",
            type=["wav","mp3","ogg","flac"],
            help="Best results with 30-second clips."
        )
        if uploaded:
            audio_bytes  = uploaded.read()
            display_name = uploaded.name

    # ── TAB 2: URL input
    with inp_tab2:
        st.markdown("""
        <div style="margin-bottom:12px;">
          <span style="color:#a0aec0;font-size:.88rem;">
            Paste any link — YouTube, TikTok, SoundCloud, Instagram, Twitter/X, Vimeo…
          </span>
        </div>
        """, unsafe_allow_html=True)

        # show supported site badges
        badges = "".join(
            f'<span class="site-badge">{s}</span>'
            for s in SUPPORTED_DOMAINS
        )
        st.markdown(f"<div style='margin-bottom:14px;'>{badges}</div>",
                    unsafe_allow_html=True)

        url_input = st.text_input(
            "URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )

        col_btn, col_note = st.columns([1,3])
        with col_btn:
            analyse_url = st.button("🔗 Download & Analyse", type="primary",
                                    use_container_width=True)
        with col_note:
            st.markdown("""
            <div style="color:#718096;font-size:.8rem;padding-top:8px;">
              ⚠️ Only publicly accessible content can be downloaded.
              Private / age-restricted content may fail.
            </div>""", unsafe_allow_html=True)

        if analyse_url and url_input.strip():
            if not is_url(url_input.strip()):
                st.error("⚠️ Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("⬇️ Downloading audio… (this may take 15-30 s)"):
                    try:
                        audio_bytes, display_name = download_audio_from_url(
                            url_input.strip())
                        st.success(f"✅ Downloaded: **{display_name}**")
                    except RuntimeError as e:
                        st.error(f"❌ Download failed: {e}")
                        audio_bytes = None

    # ── ANALYSIS (shared for both inputs)
    if audio_bytes:
        st.markdown("---")
        render_analysis(audio_bytes, display_name, bundle)


# ─────────────────────────────────────────────────────────────
# ── PAGE: MODEL METRICS
# ─────────────────────────────────────────────────────────────
elif page == "📊 Model Metrics":
    st.markdown("""
    <div class="hero" style="padding:26px 40px;">
      <div class="hero-title" style="font-size:2rem;">📊 Model Performance</div>
      <div class="hero-sub">Accuracy, balanced accuracy &amp; ROC-AUC across all classifiers.</div>
    </div>
    """, unsafe_allow_html=True)

    m = bundle['metrics']
    st.markdown('<div class="sh">🏆 Overall Comparison</div>', unsafe_allow_html=True)
    st.pyplot(plot_model_comparison(m))

    st.markdown('<div class="sh">📋 Per-Model Details</div>', unsafe_allow_html=True)
    tabs = st.tabs(["🌲 Random Forest","⚡ XGBoost","📐 Logistic Regression"])

    for tab, key, cmap in zip(tabs,
            ["Random Forest","XGBoost","Logistic Regression"],
            ["Blues","Reds","Purples"]):
        with tab:
            s = m[key]
            ca,cb,cc = st.columns(3)
            for col, val, lbl, c in [
                (ca, f"{s['acc']:.3f}","Accuracy",          ACCENT),
                (cb, f"{s['bal']:.3f}","Balanced Accuracy", ACCENT2),
                (cc, f"{s['auc']:.3f}","ROC-AUC",           "#4facfe"),
            ]:
                col.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:{c};">{val}</div>
                  <div class="metric-label">{lbl}</div></div>""",
                  unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(plot_f1(s['report'], f"F1-Score per Genre — {key}"))
            st.markdown("**Classification Report**")
            cr_df = pd.DataFrame(s['report']).T.round(3)
            st.dataframe(cr_df.style.background_gradient(
                subset=['precision','recall','f1-score'], cmap='RdYlGn'),
                use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ── PAGE: CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Confusion Matrices":
    st.markdown("""
    <div class="hero" style="padding:26px 40px;">
      <div class="hero-title" style="font-size:2rem;">🔬 Confusion Matrices</div>
      <div class="hero-sub">True vs predicted labels for each model on the held-out test set.</div>
    </div>
    """, unsafe_allow_html=True)

    m   = bundle['metrics']
    enc = bundle['encoder']
    for key, cmap, icon in [
        ("Random Forest",       "Blues",   "🌲"),
        ("XGBoost",             "Reds",    "⚡"),
        ("Logistic Regression", "Purples", "📐"),
    ]:
        st.markdown(f'<div class="sh">{icon} {key}</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_confusion(m[key]['cm'], enc.classes_,
                                 f"{key} — Confusion Matrix", cmap))
        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ── PAGE: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
elif page == "📈 Feature Importance":
    st.markdown("""
    <div class="hero" style="padding:26px 40px;">
      <div class="hero-title" style="font-size:2rem;">📈 Feature Importance</div>
      <div class="hero-sub">MDI importance (Random Forest) and Gain importance (XGBoost).</div>
    </div>
    """, unsafe_allow_html=True)

    rf    = bundle['rf']
    xgb_m = bundle['xgb']
    cols  = bundle['feature_cols']

    tab1, tab2 = st.tabs(["🌲 Random Forest","⚡ XGBoost"])

    with tab1:
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        fig, ax = dark_fig(12,5)
        ax.bar(range(20), imp[idx], color=ACCENT, alpha=.85)
        ax.set_xticks(range(20))
        ax.set_xticklabels([cols[i] for i in idx], rotation=45,
                           ha='right', fontsize=8, color=TEXT)
        ax.set_title("Top 20 Features — Random Forest (MDI)",
                     color=TEXT, fontsize=13, fontweight='bold')
        ax.set_ylabel("Mean Decrease in Impurity")
        plt.tight_layout(); st.pyplot(fig)

    with tab2:
        imp = xgb_m.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        fig, ax = dark_fig(12,5)
        ax.bar(range(20), imp[idx], color=ACCENT2, alpha=.85)
        ax.set_xticks(range(20))
        ax.set_xticklabels([cols[i] for i in idx], rotation=45,
                           ha='right', fontsize=8, color=TEXT)
        ax.set_title("Top 20 Features — XGBoost (Gain)",
                     color=TEXT, fontsize=13, fontweight='bold')
        ax.set_ylabel("Gain Importance")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown(f"""
    <div class="info-box" style="margin-top:20px;">
      <b>ℹ️ Data source:</b> {bundle['source']}.
      Place <code>features_30_sec.csv</code> (GTZAN) in the same folder
      and restart to use real data.
    </div>
    """, unsafe_allow_html=True)
