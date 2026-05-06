"""
╔══════════════════════════════════════════════════════════╗
║   Zomato IPO — AI Sentiment Analysis Dashboard           ║
║   Streamlit Deployment | NLP Project 2024-25             ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import os
import time

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato IPO Sentiment Analyzer",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Zomato Brand Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #282C3F 0%, #1a1e2e 100%);
}
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #cccccc !important; font-size: 13px; }

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #E23744 0%, #FC8019 100%);
    padding: 28px 36px;
    border-radius: 16px;
    color: white;
    margin-bottom: 28px;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
.main-header p  { margin: 6px 0 0; font-size: 1rem; opacity: 0.9; }

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #E23744;
    text-align: center;
}
.metric-card .val  { font-size: 2rem; font-weight: 800; color: #E23744; }
.metric-card .lbl  { font-size: 0.82rem; color: #777; margin-top: 4px; font-weight: 500; }

/* ── Sentiment result cards ── */
.result-positive { background: #e8f5e9; border-left: 6px solid #2e7d32; border-radius: 12px; padding: 20px 24px; }
.result-negative { background: #fce4ec; border-left: 6px solid #E23744; border-radius: 12px; padding: 20px 24px; }
.result-neutral  { background: #fff8e1; border-left: 6px solid #FC8019; border-radius: 12px; padding: 20px 24px; }
.result-positive h2, .result-negative h2, .result-neutral h2 { margin: 0 0 6px; }
.result-positive p,  .result-negative p,  .result-neutral p  { margin: 0; font-size: 0.9rem; color: #555; }

/* ── Section titles ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #282C3F;
    border-left: 4px solid #E23744;
    padding-left: 12px;
    margin-bottom: 16px;
}

/* ── Feature badge ── */
.feature-pill {
    display: inline-block;
    background: #f0f4ff;
    color: #3949ab;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}

/* ── Info box ── */
.info-box {
    background: #f5f7ff;
    border: 1px solid #c5cae9;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.88rem;
    color: #444;
}

/* ── Hide default header ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BULLISH_KEYWORDS = [
    "profit", "growth", "revenue", "subscribe", "listing gains", "multibagger",
    "bullish", "oversubscribed", "strong", "buy", "invest", "opportunity",
    "returns", "gain", "surge", "rally", "upside", "positive", "recommend",
    "subscription", "allotment", "grey market premium", "gmp", "good", "great",
    "excellent", "amazing", "fantastic", "best", "top", "leading", "dominant",
    "market leader", "high demand", "solid", "fundamentals", "promising",
]
BEARISH_KEYWORDS = [
    "loss", "overvalued", "expensive", "avoid", "debt", "risky", "bearish",
    "burn rate", "never profitable", "fraud", "scam", "poor", "bad", "worst",
    "crash", "decline", "fall", "drop", "risky", "concern", "warning",
    "sell", "short", "negative", "disappointing", "unprofitable", "deficit",
    "liability", "problem", "issue", "weak", "fail", "collapse", "bubble",
    "inflated", "valuation concern", "red flag", "loss making",
]
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","that","this","these",
    "those","it","its","he","she","they","we","you","i","me","him","her",
    "them","us","my","your","his","our","their","what","which","who","when",
    "where","why","how","all","each","every","both","more","most","other",
    "such","no","not","only","same","so","than","too","very","just","can",
}
LABEL_MAP   = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI_MAP   = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
COLOR_MAP   = {"Positive": "#2e7d32", "Negative": "#E23744", "Neutral": "#FC8019"}
BG_MAP      = {"Positive": "#e8f5e9",  "Negative": "#fce4ec",  "Neutral": "#fff8e1"}
MODEL_STATS = {
    "Logistic Regression": {"cv": 0.7462, "test": 0.6875, "f1": 0.6976, "prec": 0.746,  "rec": 0.6889},
    "Naive Bayes":         {"cv": 0.7769, "test": 0.7500, "f1": 0.7556, "prec": 0.7905, "rec": 0.7556},
    "Random Forest ★":    {"cv": 0.7936, "test": 0.8750, "f1": 0.8783, "prec": 0.9167, "rec": 0.8667},
    "Linear SVC":          {"cv": 0.7782, "test": 0.8125, "f1": 0.8194, "prec": 0.8381, "rec": 0.8111},
    "Gradient Boosting":   {"cv": 0.7141, "test": 0.8750, "f1": 0.8611, "prec": 0.9048, "rec": 0.8667},
}

# ─────────────────────────────────────────────
# LOAD MODELS  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load saved ML artifacts from disk."""
    base = os.path.dirname(__file__)
    artifacts = {}
    for name, fname in [("model", "best_model.pkl"),
                         ("tfidf",  "tfidf_vectorizer.pkl"),
                         ("le",     "label_encoder.pkl")]:
        path = os.path.join(base, "models", fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)
        else:
            artifacts[name] = None
    return artifacts

# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Reproduce the exact preprocessing applied during training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # strip URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # strip handles/hashtags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)          # strip non-ASCII
    text = re.sub(r"\d+", "", text)                      # strip digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def extract_features(raw_text: str) -> dict:
    """Extract all engineered numerical features from raw text."""
    words     = raw_text.split()
    clean     = clean_text(raw_text)
    c_words   = clean.split()
    bull      = sum(1 for kw in BULLISH_KEYWORDS if kw in raw_text.lower())
    bear      = sum(1 for kw in BEARISH_KEYWORDS if kw in raw_text.lower())
    total     = max(len(words), 1)
    return {
        "word_count":              len(words),
        "clean_word_count":        len(c_words),
        "char_length":             len(clean),
        "avg_word_len":            np.mean([len(w) for w in c_words]) if c_words else 0,
        "unique_word_ratio":       len(set(c_words)) / max(len(c_words), 1),
        "exclamation_count":       raw_text.count("!"),
        "question_count":          raw_text.count("?"),
        "upper_word_count":        sum(1 for w in words if w.isupper() and len(w) > 1),
        "bullish_keyword_count":   bull,
        "bearish_keyword_count":   bear,
        "sentiment_keyword_ratio": (bull - bear) / total,
    }
from scipy.sparse import hstack, csr_matrix

NUM_COLS_ORDER = [
    "word_count",
    "clean_word_count",
    "char_length",
    "avg_word_len",
    "unique_word_ratio",
    "bullish_keyword_count",
    "bearish_keyword_count",
    "sentiment_keyword_ratio",
    "exclamation_count",
    "question_count",
]
def predict_sentiment(text: str, artifacts: dict):
    feats = extract_features(text)
    clean = clean_text(text)

    if artifacts.get("model") and artifacts.get("tfidf"):

        # Step A: TF-IDF on cleaned text → shape (1, 3000)
        tfidf_vec = artifacts["tfidf"].transform([clean])

        # Step B: 10 numeric features in exact training order → shape (1, 10)
        num_vals = np.array(
            [[feats[col] for col in NUM_COLS_ORDER]],
            dtype=np.float64
        )

        # Step C: combine both → shape (1, 3010) ← what the model expects
        X_final = hstack([tfidf_vec, csr_matrix(num_vals)])

        # Step D: predict
        proba      = artifacts["model"].predict_proba(X_final)[0]
        pred_idx   = int(np.argmax(proba))
        label      = LABEL_MAP[pred_idx]
        proba_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)}

    else:
        score = feats["sentiment_keyword_ratio"]
        if score > 0.02:
            label, proba_dict = "Positive", {"Positive": 0.75, "Neutral": 0.18, "Negative": 0.07}
        elif score < -0.02:
            label, proba_dict = "Negative", {"Positive": 0.08, "Neutral": 0.22, "Negative": 0.70}
        else:
            label, proba_dict = "Neutral",  {"Positive": 0.25, "Neutral": 0.55, "Negative": 0.20}

    return label, proba_dict, feats, clean
# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def confidence_bar_chart(proba_dict: dict):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    labels = ["Negative", "Neutral", "Positive"]
    values = [proba_dict.get(l, 0) * 100 for l in labels]
    colors = [COLOR_MAP[l] for l in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.45)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold", color="#333")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#555")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig

def model_comparison_chart():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    models  = list(MODEL_STATS.keys())
    cv_vals = [MODEL_STATS[m]["cv"]   * 100 for m in models]
    ts_vals = [MODEL_STATS[m]["test"] * 100 for m in models]
    f1_vals = [MODEL_STATS[m]["f1"]   * 100 for m in models]
    x     = np.arange(len(models))
    w     = 0.25
    ax.bar(x - w, cv_vals, w, label="CV Acc (%)",   color="#282C3F", alpha=0.85)
    ax.bar(x,     ts_vals, w, label="Test Acc (%)", color="#E23744", alpha=0.85)
    ax.bar(x + w, f1_vals, w, label="F1 Macro (%)", color="#FC8019", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ★", "") for m in models], rotation=12, ha="right", fontsize=8)
    ax.set_ylim(55, 100)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Model Performance Comparison", fontsize=11, fontweight="bold", color="#282C3F", pad=8)
    ax.axvline(2, color="#E23744", linewidth=1.5, linestyle="--", alpha=0.4, label="Best Model")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

def feature_radar_chart(feats: dict):
    categories = ["Bullish\nKeywords", "Bearish\nKeywords", "Exclamations",
                  "Questions", "CAPS Words", "Unique\nWord Ratio"]
    max_vals   = [10, 10, 5, 5, 10, 1.0]
    raw_vals   = [
        feats["bullish_keyword_count"],
        feats["bearish_keyword_count"],
        feats["exclamation_count"],
        feats["question_count"],
        feats["upper_word_count"],
        feats["unique_word_ratio"],
    ]
    norm_vals  = [min(v / m, 1.0) for v, m in zip(raw_vals, max_vals)]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    norm_vals += norm_vals[:1]
    angles    += angles[:1]
    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    ax.fill(angles, norm_vals, color="#E23744", alpha=0.25)
    ax.plot(angles, norm_vals, color="#E23744", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, color="#333")
    ax.set_yticklabels([])
    ax.set_title("Feature Profile", fontsize=10, fontweight="bold", color="#282C3F", pad=14)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <span style='font-size:2.5rem;'>🍕</span><br>
      <span style='font-size:1.2rem; font-weight:800; color:#FC8019;'>Zomato IPO</span><br>
      <span style='font-size:0.85rem; color:#aaa;'>Sentiment Analyzer</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🎯  Sentiment Analyzer",
        "📦  Batch Analysis",
        "📊  Model Dashboard",
        "ℹ️  About the Project",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#888; line-height:1.7;'>
      <b style='color:#FC8019;'>Model</b> &nbsp; Random Forest<br>
      <b style='color:#FC8019;'>Accuracy</b> &nbsp; 87.5%<br>
      <b style='color:#FC8019;'>F1-Macro</b> &nbsp; 0.878<br>
      <b style='color:#FC8019;'>Features</b> &nbsp; TF-IDF + 11 engineered<br>
      <b style='color:#FC8019;'>Classes</b> &nbsp; Positive / Neutral / Negative
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#666; text-align:center;'>
      NLP Project 2024–25 &nbsp;|&nbsp; Team Zomato<br>
      <span style='color:#E23744;'>♥</span> Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading model artifacts…"):
    artifacts = load_models()

model_loaded = artifacts.get("model") is not None

# ─────────────────────────────────────────────
# PAGE 1 — SENTIMENT ANALYZER
# ─────────────────────────────────────────────
if "Sentiment Analyzer" in page:
    st.markdown("""
    <div class="main-header">
      <h1>🎯 Zomato IPO Sentiment Analyzer</h1>
      <p>Paste any financial news headline, investor comment, or review below and get an instant AI-powered sentiment score.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("""
        ⚠️ **Model files not found** — Place `best_model.pkl`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl`
        inside the `models/` folder next to `app.py`. Running in **heuristic fallback mode** until then.
        """)

    # ── Sample buttons ──
    st.markdown('<div class="section-title">💡 Try a sample or write your own</div>', unsafe_allow_html=True)
    samples = {
        "📈 Bullish":   "Zomato IPO has strong fundamentals, massive market share, and listing gains are expected. Subscribe for solid returns!",
        "📉 Bearish":   "Zomato is a loss-making company with a risky valuation. Avoid this overpriced IPO — it's a bubble waiting to burst.",
        "⚖️ Neutral":   "Zomato IPO was oversubscribed 38 times and raised Rs 9375 crore, making it one of the largest Indian tech listings.",
        "🗣️ Hinglish":  "Zomato ka IPO subscribe karna chahiye ya nahi? Loss-making company hai but market leader bhi hai bhai.",
        "⭐ Review":    "Best food delivery app! Great delivery speed, supportive staff, and amazing variety of restaurants. Highly recommend!",
    }
    cols = st.columns(len(samples))
    selected_sample = ""
    for col, (label, text) in zip(cols, samples.items()):
        if col.button(label, use_container_width=True):
            selected_sample = text

    # ── Text input ──
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Enter financial text to analyze",
        value=selected_sample,
        height=130,
        placeholder="Type or paste a news headline, YouTube comment, investor opinion, or Zomato review…",
        label_visibility="collapsed",
    )

    analyze_btn = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn and user_input.strip():
        with st.spinner("Analyzing…"):
            time.sleep(0.4)  # UX breathing room
            label, proba, feats, clean = predict_sentiment(user_input, artifacts)

        st.markdown("<br>", unsafe_allow_html=True)
        # ── Result header ──
        confidence = proba[label] * 100
        css_class  = f"result-{label.lower()}"
        emoji      = EMOJI_MAP[label]
        color      = COLOR_MAP[label]
        st.markdown(f"""
        <div class="{css_class}">
          <h2 style="color:{color};">{emoji} &nbsp; {label} Sentiment</h2>
          <p>Confidence: <strong style="color:{color};">{confidence:.1f}%</strong> &nbsp;|&nbsp;
             Model: <strong>Random Forest</strong> &nbsp;|&nbsp;
             Words analyzed: <strong>{feats['word_count']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns([2, 1.8, 1.4])

        with col_a:
            st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)
            st.pyplot(confidence_bar_chart(proba))

        with col_b:
            st.markdown('<div class="section-title">Feature Profile</div>', unsafe_allow_html=True)
            st.pyplot(feature_radar_chart(feats))

        with col_c:
            st.markdown('<div class="section-title">Extracted Signals</div>', unsafe_allow_html=True)
            st.metric("Bullish Keywords",  feats["bullish_keyword_count"])
            st.metric("Bearish Keywords",  feats["bearish_keyword_count"])
            st.metric("Sentiment Ratio",   f"{feats['sentiment_keyword_ratio']:.3f}")
            st.metric("Unique Word Ratio", f"{feats['unique_word_ratio']:.2f}")
            st.metric("Word Count",        feats["word_count"])

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔎 View cleaned text & all engineered features"):
            st.markdown("**Cleaned Text:**")
            st.code(clean if clean.strip() else "(empty after cleaning)", language=None)
            st.markdown("**All Engineered Features:**")
            feat_df = pd.DataFrame([feats]).T.reset_index()
            feat_df.columns = ["Feature", "Value"]
            feat_df["Value"] = feat_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ─────────────────────────────────────────────
# PAGE 2 — BATCH ANALYSIS
# ─────────────────────────────────────────────
elif "Batch Analysis" in page:
    st.markdown("""
    <div class="main-header">
      <h1>📦 Batch Sentiment Analysis</h1>
      <p>Upload a CSV file with a text column to score an entire dataset at once. Download the results with predictions appended.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      📋 <strong>How to use:</strong> Upload a CSV that has at least one text column (named <code>text</code>, <code>review</code>,
      or <code>comment</code>). The app will run every row through the full preprocessing and inference pipeline
      and append <code>predicted_sentiment</code> and <code>confidence</code> columns to your file.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Preview — {len(df_upload):,} rows × {len(df_upload.columns)} columns**")
        st.dataframe(df_upload.head(5), use_container_width=True)

        # Detect text column
        text_col_options = [c for c in df_upload.columns
                            if any(k in c.lower() for k in ["text", "review", "comment", "content", "body"])]
        if not text_col_options:
            text_col_options = list(df_upload.columns)

        text_col = st.selectbox("Select the text column to analyze", text_col_options)

        if st.button("🚀  Run Batch Analysis", type="primary"):
            progress = st.progress(0)
            preds, confs, labels_neg, labels_neu, labels_pos = [], [], [], [], []

            for i, row_text in enumerate(df_upload[text_col].fillna("").astype(str)):
                lbl, proba, _, _ = predict_sentiment(row_text, artifacts)
                preds.append(lbl)
                confs.append(round(proba[lbl] * 100, 1))
                labels_neg.append(round(proba.get("Negative", 0) * 100, 1))
                labels_neu.append(round(proba.get("Neutral", 0) * 100, 1))
                labels_pos.append(round(proba.get("Positive", 0) * 100, 1))
                progress.progress((i + 1) / len(df_upload))

            df_upload["predicted_sentiment"] = preds
            df_upload["confidence_%"]        = confs
            df_upload["prob_negative_%"]     = labels_neg
            df_upload["prob_neutral_%"]      = labels_neu
            df_upload["prob_positive_%"]     = labels_pos

            st.success(f"✅ Done! Analyzed {len(df_upload):,} records.")
            st.dataframe(df_upload.head(20), use_container_width=True)

            # ── Distribution pie ──
            counts = pd.Series(preds).value_counts()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
            clrs = [COLOR_MAP.get(l, "#ccc") for l in counts.index]
            ax1.pie(counts.values, labels=counts.index, colors=clrs,
                    autopct="%1.1f%%", startangle=140, textprops={"fontsize": 10})
            ax1.set_title("Sentiment Distribution", fontweight="bold", color="#282C3F")
            ax2.bar(counts.index, counts.values, color=clrs, edgecolor="white")
            ax2.set_ylabel("Count")
            ax2.spines[["top","right"]].set_visible(False)
            ax2.set_title("Sentiment Counts", fontweight="bold", color="#282C3F")
            st.pyplot(fig)

            # ── Download ──
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Results CSV", csv_out,
                               "zomato_sentiment_results.csv", "text/csv",
                               use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — MODEL DASHBOARD
# ─────────────────────────────────────────────
elif "Model Dashboard" in page:
    st.markdown("""
    <div class="main-header">
      <h1>📊 Model Performance Dashboard</h1>
      <p>Full evaluation metrics, confusion matrices, ROC curves, and feature importance for all five trained classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown('<div class="metric-card"><div class="val">87.5%</div><div class="lbl">Best Test Accuracy<br>(Random Forest)</div></div>', unsafe_allow_html=True)
    k2.markdown('<div class="metric-card"><div class="val">0.878</div><div class="lbl">Best F1-Macro<br>(Random Forest)</div></div>', unsafe_allow_html=True)
    k3.markdown('<div class="metric-card"><div class="val">79.4%</div><div class="lbl">CV Mean Accuracy<br>(Random Forest)</div></div>', unsafe_allow_html=True)
    k4.markdown('<div class="metric-card"><div class="val">3,002</div><div class="lbl">Total Features<br>(TF-IDF + Engineered)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison ──
    st.markdown('<div class="section-title">Model Comparison — All 5 Classifiers</div>', unsafe_allow_html=True)
    st.pyplot(model_comparison_chart())

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ──
    st.markdown('<div class="section-title">Detailed Results Table</div>', unsafe_allow_html=True)
    df_results = pd.DataFrame([
        {
            "Model": m,
            "CV Acc (%)":    f"{v['cv']*100:.1f}",
            "Test Acc (%)":  f"{v['test']*100:.1f}",
            "Precision":     f"{v['prec']:.4f}",
            "Recall":        f"{v['rec']:.4f}",
            "F1-Macro":      f"{v['f1']:.4f}",
        }
        for m, v in MODEL_STATS.items()
    ])
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrices ──
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_map = {
        "Logistic Regression": "models/confusion_matrix_Logistic_Regression.png",
        "Naive Bayes":         "models/confusion_matrix_Naive_Bayes.png",
        "Random Forest":       "models/confusion_matrix_Random_Forest.png",
        "Linear SVC":          "models/confusion_matrix_Linear_SVC.png",
        "Gradient Boosting":   "models/confusion_matrix_Gradient_Boosting.png",
    }
    selected_cm = st.selectbox("Select model", list(cm_map.keys()))
    cm_path = os.path.join(os.path.dirname(__file__), cm_map[selected_cm])
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix — {selected_cm}", use_container_width=False, width=500)
    else:
        st.info("Place confusion matrix PNG files inside the `models/` folder to display them here.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC + Feature importance ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        roc_path = os.path.join(os.path.dirname(__file__), "models/roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Place `roc_curve.png` in the `models/` folder.")
    with c2:
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fi_path = os.path.join(os.path.dirname(__file__), "models/feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Place `feature_importance.png` in the `models/` folder.")

# ─────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────
elif "About" in page:
    st.markdown("""
    <div class="main-header">
      <h1>ℹ️ About the Project</h1>
      <p>Zomato IPO Sentiment Analysis — NLP Project 2024–25</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Problem Statement
    Retail investors participating in high-profile Indian IPOs like Zomato lack access to a real-time,
    NLP-powered system that can aggregate unstructured sentiment data from multiple sources, analyze it
    intelligently, and deliver actionable, easy-to-understand market insights.

    ### 🗂️ Datasets Used
    | Dataset | Samples | Source |
    |---|---|---|
    | News Articles | 221 | Moneycontrol, ET Markets, Livemint |
    | App Reviews | 5,479 | Zomato Platform |
    | YouTube Comments | 16,487 | Akshat Shrivastava, CA Rachana Ranade, Pranjal Kamra |
    | YouTube Videos | 200+ | Financial YouTube channels |

    ### 🔧 Pipeline Overview
    1. **Data Collection** — Multi-source scraping (news portals, YouTube API, Zomato app)
    2. **Preprocessing** — URL stripping, Hinglish-aware cleaning, stopword removal, date parsing
    3. **Feature Engineering** — TF-IDF (3,000 features, bigrams) + 11 engineered numerical signals
    4. **Model Training** — 5 classifiers evaluated via 5-fold stratified cross-validation
    5. **Deployment** — Best model (Random Forest) serialized and served via this Streamlit app

    ### 🤖 Engineered Features
    """)

    feat_pills = [
        "word_count", "clean_word_count", "char_length", "avg_word_len",
        "unique_word_ratio", "exclamation_count", "question_count", "upper_word_count",
        "bullish_keyword_count", "bearish_keyword_count", "sentiment_keyword_ratio",
        "source_encoded", "channel_freq", "like_count_log", "reply_count_log", "is_reply",
    ]
    pills_html = " ".join([f'<span class="feature-pill">{f}</span>' for f in feat_pills])
    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("""
    <br>

    ### 📁 Project File Structure
    ```
    streamlit_app/
    ├── app.py                          ← This file (main Streamlit app)
    ├── requirements.txt                ← Python dependencies
    └── models/
        ├── best_model.pkl              ← Trained Random Forest classifier
        ├── tfidf_vectorizer.pkl        ← Fitted TF-IDF vectorizer
        ├── label_encoder.pkl           ← Label encoder (0=Neg, 1=Neu, 2=Pos)
        ├── confusion_matrix_*.png      ← Confusion matrix images (5 models)
        ├── roc_curve.png               ← ROC curve image
        └── feature_importance.png      ← Feature importance image
    ```

    ### 🚀 How to Run Locally
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Place your .pkl and .png files inside models/

    # 3. Run the app
    streamlit run app.py
    ```

    ### ☁️ How to Deploy on Streamlit Cloud
    1. Push this folder to a **GitHub repository**
    2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
    3. Click **New App** → select your repo → set **Main file path** to `app.py`
    4. Click **Deploy** — your app will be live in ~2 minutes!

    > **Note on large .pkl files:** If your model files exceed 100MB, use [Git LFS](https://git-lfs.github.com/) or
    > store them in Google Drive / HuggingFace Hub and load them dynamically.
    """, unsafe_allow_html=True)
