import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Agent · Zomato IPO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.main { background: #f7f5f0; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    color: #0d0d0d; letter-spacing: -0.03em; line-height: 1.1;
}
.hero-sub {
    font-size: 1rem; color: #666; font-weight: 300; margin-top: 0.4rem;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700; color: #0d0d0d;
    border-left: 4px solid #f7c948;
    padding-left: 0.75rem; margin: 1.5rem 0 0.8rem 0;
}
.metric-card {
    background: #0d0d0d; border-radius: 12px;
    padding: 1.2rem 1.4rem; color: white;
}
.metric-card .label {
    font-size: 0.7rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #777; margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 700; color: #f7c948;
}
.metric-card .delta { font-size: 0.78rem; color: #aaa; margin-top: 0.2rem; }

.pred-positive {
    background: #d4f5e2; border: 2px solid #1a6b3c;
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.pred-negative {
    background: #fde8e8; border: 2px solid #9b1c1c;
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.pred-neutral {
    background: #fff3cd; border: 2px solid #856404;
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.pred-label {
    font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800;
}
.pred-confidence { font-size: 0.9rem; color: #555; margin-top: 0.3rem; }
.thin-divider { border: none; border-top: 1px solid #ddd; margin: 1.2rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for fname, key in [("best_model.pkl","classifier"),
                       ("tfidf_vectorizer.pkl","vectorizer"),
                       ("label_encoder.pkl","label_encoder")]:
        try:
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            models[key] = None
    return models

models = load_models()

# ─── Load CSVs ────────────────────────────────────────────────────────────────
DATA_DIR = "cleaned_data"

@st.cache_data
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_all_data():
    return {
        "ipo_tweets":     load_csv("ipo_tweets_clean.csv"),
        "reviews":        load_csv("reviews_clean.csv"),
        "zomato_ipo":     load_csv("zomato-ipo.csv"),
        "news":           load_csv("news_zomato.csv"),
        "yt_comments":    load_csv("youtube_comments_zoma.csv"),
        "yt_videos":      load_csv("youtube_videos_zomato.csv"),
        "zomato_reviews": load_csv("zomato_reviews.csv"),
    }

data = load_all_data()

# ─── Text Preprocessing ───────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text: str):
    if not models["vectorizer"] or not models["classifier"]:
        return None, None, None
    cleaned = clean_text(text)
    vec  = models["vectorizer"].transform([cleaned])
    pred = models["classifier"].predict(vec)[0]
    proba = (models["classifier"].predict_proba(vec)[0]
             if hasattr(models["classifier"], "predict_proba") else None)
    label = (models["label_encoder"].inverse_transform([pred])[0]
             if models["label_encoder"] else pred)
    return label, proba, cleaned

LABEL_MAP = {1:"Positive", -1:"Negative", 0:"Neutral",
             "positive":"Positive", "negative":"Negative", "neutral":"Neutral",
             "Positive":"Positive", "Negative":"Negative", "Neutral":"Neutral"}

def norm_label(v):
    return LABEL_MAP.get(v, str(v))

def fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    color:#f7c948;letter-spacing:-0.02em;'>📈 AI Financial</div>
        <div style='font-size:0.72rem;color:#555;letter-spacing:0.1em;
                    text-transform:uppercase;margin-top:2px;'>Zomato IPO · Market Agent</div>
    </div>
    """, unsafe_allow_html=True)

    clf_ok = "✅" if models["classifier"]    else "❌"
    vec_ok = "✅" if models["vectorizer"]    else "❌"
    enc_ok = "✅" if models["label_encoder"] else "❌"
    st.markdown(f"""
    <div style='background:#1a1a1a;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1.2rem;'>
        <div style='font-size:0.7rem;color:#777;letter-spacing:0.1em;text-transform:uppercase;
                    margin-bottom:0.5rem;'>Model Status</div>
        <div style='font-size:0.82rem;'>{clf_ok} best_model.pkl</div>
        <div style='font-size:0.82rem;'>{vec_ok} tfidf_vectorizer.pkl</div>
        <div style='font-size:0.82rem;'>{enc_ok} label_encoder.pkl</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["🏠  Overview", "🔮  Live Prediction", "📊  Data Explorer",
                     "🤖  Model Performance", "💬  Sentiment Analysis", "📰  News & Social"],
                    label_visibility="collapsed")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class='hero-title'>Zomato IPO<br><span style='color:#f7c948'>Financial Intelligence</span></div>
    <div class='hero-sub'>NLP-powered sentiment · Real-time market insights · Multi-source data</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [("IPO Tweets", data["ipo_tweets"], "Twitter signals"),
             ("App Reviews", data["reviews"],    "User reviews"),
             ("News Articles", data["news"],     "Media coverage"),
             ("YT Comments", data["yt_comments"],"YouTube engagement")]
    for col, (name, df, desc) in zip([col1,col2,col3,col4], cards):
        rows = fmt(len(df)) if df is not None else "N/A"
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{name}</div>
                <div class='value'>{rows}</div>
                <div class='delta'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression", "50.0%",  "Dataset 2")
    c2.metric("SVM Classifier",      "58.3%",  "Dataset 2 ↑")
    c3.metric("Best Saved Model",    "Active" if models["classifier"] else "Missing",
              "best_model.pkl ✅" if models["classifier"] else "Not loaded ❌")

    st.markdown("<div class='section-header'>Dataset Sizes</div>", unsafe_allow_html=True)
    lmap = {"ipo_tweets":"IPO Tweets","reviews":"App Reviews","news":"News",
            "yt_comments":"YT Comments","yt_videos":"YT Videos",
            "zomato_reviews":"Zomato Reviews","zomato_ipo":"IPO Data"}
    names, rows = [], []
    for k, df in data.items():
        if df is not None:
            names.append(lmap.get(k, k)); rows.append(len(df))

    fig = go.Figure(go.Bar(x=names, y=rows, marker_color="#f7c948",
                           marker_line_color="#0d0d0d", marker_line_width=1.5))
    fig.update_layout(plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                      font=dict(family="DM Sans"), height=300,
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e5e5e5"),
                      margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 · LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":
    st.markdown("""
    <div class='hero-title'>Live <span style='color:#f7c948'>Sentiment Predictor</span></div>
    <div class='hero-sub'>Enter any text — get an instant prediction from your trained model</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    if not models["classifier"] or not models["vectorizer"]:
        st.error("⚠️ Model files not found. Ensure `best_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder as `app.py`.")
    else:
        # ── Single prediction ──
        st.markdown("<div class='section-header'>Single Text Prediction</div>", unsafe_allow_html=True)

        ex1, ex2, ex3 = st.columns(3)
        example_text = st.session_state.get("example_text", "")
        if ex1.button("📈 Bullish example"):
            st.session_state["example_text"] = "Zomato IPO is a great investment! Strong growth potential and solid fundamentals."
            st.rerun()
        if ex2.button("📉 Bearish example"):
            st.session_state["example_text"] = "Zomato IPO is way overvalued. Too risky at current price levels."
            st.rerun()
        if ex3.button("😐 Neutral example"):
            st.session_state["example_text"] = "Zomato files for IPO worth Rs 9375 crore. Proceeds to be used for expansion."
            st.rerun()

        user_input = st.text_area("Enter text to analyze",
                                  value=st.session_state.get("example_text", ""),
                                  height=130,
                                  placeholder="Type a tweet, review, or headline about Zomato IPO...")

        if st.button("🔮  Predict Sentiment", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    label, proba, cleaned = predict_sentiment(user_input)

                label_str = norm_label(label)
                css_cls = {"Positive":"pred-positive","Negative":"pred-negative",
                           "Neutral":"pred-neutral"}.get(label_str, "pred-neutral")
                emoji   = {"Positive":"📈","Negative":"📉","Neutral":"➡️"}.get(label_str, "❓")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div class='{css_cls}'>
                        <div class='pred-label'>{emoji} {label_str}</div>
                        <div class='pred-confidence'>Predicted sentiment</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    if proba is not None:
                        le = models["label_encoder"]
                        clf = models["classifier"]
                        class_names = ([norm_label(c) for c in le.classes_]
                                       if le else [str(c) for c in clf.classes_])
                        proba_df = pd.DataFrame({"Class": class_names, "Probability": proba})
                        bar_colors = ["#d4f5e2" if c=="Positive"
                                      else "#fde8e8" if c=="Negative"
                                      else "#fff3cd" for c in proba_df["Class"]]
                        fig = go.Figure(go.Bar(
                            x=proba_df["Class"], y=proba_df["Probability"],
                            marker_color=bar_colors,
                            marker_line_color="#0d0d0d", marker_line_width=1.5,
                            text=[f"{p:.1%}" for p in proba_df["Probability"]],
                            textposition="outside"
                        ))
                        fig.update_layout(
                            plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                            font=dict(family="DM Sans"), height=240,
                            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                            margin=dict(t=10,b=10,l=10,r=10),
                            title=dict(text="Confidence per class", font=dict(size=13))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("This model doesn't support probability estimates (no predict_proba).")

                st.markdown(f"**Cleaned text used:** `{cleaned}`")
            else:
                st.warning("Please enter some text first.")

        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

        # ── Batch prediction ──
        st.markdown("<div class='section-header'>Batch Prediction — Upload CSV</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a CSV file with a text column", type=["csv"])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
            if text_cols:
                text_col = st.selectbox("Select the text column", text_cols)
                if st.button("▶️  Run Batch Prediction"):
                    with st.spinner(f"Predicting {len(batch_df)} rows..."):
                        preds, confs = [], []
                        for txt in batch_df[text_col].fillna(""):
                            lbl, prob, _ = predict_sentiment(str(txt))
                            preds.append(norm_label(lbl))
                            confs.append(f"{max(prob):.1%}" if prob is not None else "N/A")
                        batch_df["predicted_sentiment"] = preds
                        batch_df["confidence"]          = confs

                    st.success(f"✅ Predicted {len(batch_df)} rows!")
                    st.dataframe(batch_df, use_container_width=True)

                    counts = batch_df["predicted_sentiment"].value_counts()
                    fig2 = px.pie(values=counts.values, names=counts.index,
                                  color_discrete_sequence=["#d4f5e2","#fde8e8","#fff3cd"],
                                  hole=0.45, title="Batch Sentiment Distribution")
                    fig2.update_layout(paper_bgcolor="#f7f5f0",
                                       font=dict(family="DM Sans"), height=300)
                    st.plotly_chart(fig2, use_container_width=True)

                    st.download_button("⬇️ Download Results as CSV",
                                       batch_df.to_csv(index=False).encode("utf-8"),
                                       "predictions.csv", "text/csv")
            else:
                st.error("No text columns found in your CSV.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 · DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Explorer":
    st.markdown("""
    <div class='hero-title'>Data <span style='color:#f7c948'>Explorer</span></div>
    <div class='hero-sub'>Browse, filter and inspect all cleaned datasets</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    lmap = {"ipo_tweets":"IPO Tweets","reviews":"App Reviews","news":"News Articles",
            "yt_comments":"YouTube Comments","yt_videos":"YouTube Videos",
            "zomato_reviews":"Zomato Reviews","zomato_ipo":"IPO Data"}
    available = {k: v for k, v in data.items() if v is not None}

    if not available:
        st.warning(f"No CSV files found in `{DATA_DIR}/`.")
    else:
        key = st.selectbox("Dataset", list(available.keys()),
                           format_func=lambda k: lmap.get(k, k))
        df = available[key]

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", fmt(len(df)))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", fmt(int(df.isnull().sum().sum())))

        st.markdown("<div class='section-header'>Preview</div>", unsafe_allow_html=True)
        n = st.slider("Rows to show", 5, min(200, len(df)), 10)
        st.dataframe(df.head(n), use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

            st.markdown("<div class='section-header'>Column Distribution</div>", unsafe_allow_html=True)
            col = st.selectbox("Select column", num_cols)
            fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#f7c948"])
            fig.update_layout(plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                              font=dict(family="DM Sans"), height=280,
                              margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 · MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown("""
    <div class='hero-title'>Model <span style='color:#f7c948'>Performance</span></div>
    <div class='hero-sub'>Classification results — Dataset 2 (IPO Tweets / Reviews)</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    results = {
        "Logistic Regression": {
            "accuracy": 0.50,
            "report": pd.DataFrame({
                "Class":     ["-1 Negative","0 Neutral","1 Positive","Macro Avg","Weighted Avg"],
                "Precision": [0.45,1.00,0.00,0.48,0.52],
                "Recall":    [1.00,0.25,0.00,0.42,0.50],
                "F1-Score":  [0.62,0.40,0.00,0.34,0.39],
                "Support":   [5,4,3,12,12],
            })
        },
        "SVM": {
            "accuracy": 0.5833,
            "report": pd.DataFrame({
                "Class":     ["-1 Negative","0 Neutral","1 Positive","Macro Avg","Weighted Avg"],
                "Precision": [0.50,1.00,1.00,0.83,0.75],
                "Recall":    [1.00,0.25,0.33,0.53,0.58],
                "F1-Score":  [0.67,0.40,0.50,0.52,0.56],
                "Support":   [5,4,3,12,12],
            })
        },
    }

    model_name = st.selectbox("Select Model", list(results.keys()))
    res = results[model_name]
    acc = res["accuracy"]
    rep = res["report"]

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=acc * 100,
            number={"suffix":"%","font":{"size":36,"family":"Syne"}},
            title={"text":"Accuracy","font":{"size":14,"family":"DM Sans"}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#f7c948"},
                   "steps":[{"range":[0,50],"color":"#fde8e8"},
                             {"range":[50,70],"color":"#fff3cd"},
                             {"range":[70,100],"color":"#d4f5e2"}],
                   "threshold":{"line":{"color":"#0d0d0d","width":3},"value":acc*100}}
        ))
        fig.update_layout(height=260, paper_bgcolor="#f7f5f0",
                          margin=dict(t=30,b=10,l=20,r=20),
                          font=dict(family="DM Sans"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Classification Report</div>", unsafe_allow_html=True)
        st.dataframe(
            rep.set_index("Class").style.format("{:.2f}",
                subset=["Precision","Recall","F1-Score"]),
            use_container_width=True
        )

    st.markdown("<div class='section-header'>Precision · Recall · F1 by Class</div>", unsafe_allow_html=True)
    core = rep[~rep["Class"].str.contains("Avg")]
    fig2 = go.Figure()
    for metric, color in [("Precision","#f7c948"),("Recall","#0d0d0d"),("F1-Score","#aaa")]:
        fig2.add_trace(go.Bar(name=metric, x=core["Class"], y=core[metric],
                              marker_color=color))
    fig2.update_layout(barmode="group", plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                       font=dict(family="DM Sans"), height=300,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Engineering Summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", "60")
    c2.metric("TF-IDF Features", "5,000")
    c3.metric("Train Set", "48 (80%)")
    c4.metric("Test Set", "12 (20%)")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 · SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💬  Sentiment Analysis":
    st.markdown("""
    <div class='hero-title'>Sentiment <span style='color:#f7c948'>Analysis</span></div>
    <div class='hero-sub'>Explore sentiment distributions across data sources</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    lmap = {"ipo_tweets":"IPO Tweets","reviews":"App Reviews","news":"News Articles",
            "yt_comments":"YouTube Comments","yt_videos":"YouTube Videos",
            "zomato_reviews":"Zomato Reviews","zomato_ipo":"IPO Data"}
    available = {k: v for k, v in data.items() if v is not None}

    key = st.selectbox("Data Source", list(available.keys()),
                       format_func=lambda k: lmap.get(k, k))
    df  = available[key]
    sent_col = next((c for c in df.columns
                     if "sentiment" in c.lower() or "label" in c.lower()), None)

    if sent_col:
        counts = df[sent_col].value_counts().reset_index()
        counts.columns = ["Sentiment","Count"]
        counts["Sentiment"] = counts["Sentiment"].apply(norm_label)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(counts, values="Count", names="Sentiment", hole=0.45,
                         color_discrete_sequence=["#f7c948","#0d0d0d","#aaa"])
            fig.update_layout(paper_bgcolor="#f7f5f0", font=dict(family="DM Sans"),
                              height=320, margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure(go.Bar(
                x=counts["Sentiment"], y=counts["Count"],
                marker_color=["#f7c948","#0d0d0d","#aaa"][:len(counts)],
                marker_line_color="#0d0d0d", marker_line_width=1
            ))
            fig2.update_layout(plot_bgcolor="#f7f5f0", paper_bgcolor="#f7f5f0",
                               font=dict(family="DM Sans"), height=320,
                               margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)

        filt = st.selectbox("Filter by sentiment",
                            ["All"] + df[sent_col].unique().tolist())
        filtered = df if filt == "All" else df[df[sent_col] == filt]
        st.dataframe(filtered.head(30), use_container_width=True)
    else:
        st.info(f"No sentiment column detected. Columns available: {', '.join(df.columns[:10])}")
        st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 · NEWS & SOCIAL
# ════════════════════════════════════════════════════════════════════════════
elif page == "📰  News & Social":
    st.markdown("""
    <div class='hero-title'>News & <span style='color:#f7c948'>Social Media</span></div>
    <div class='hero-sub'>Media and social signals around the Zomato IPO</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    tabs = st.tabs(["📰 News", "🐦 IPO Tweets", "▶️ YouTube"])

    with tabs[0]:
        if data["news"] is not None:
            df = data["news"]
            st.metric("Articles", len(df))
            text_c = next((c for c in df.columns
                           if "title" in c.lower() or "text" in c.lower()), df.columns[0])
            search = st.text_input("Search news", "", key="news_s")
            disp = df[df[text_c].astype(str).str.contains(search, case=False)] if search else df
            st.dataframe(disp.head(30), use_container_width=True)
        else:
            st.warning("`news_zomato.csv` not found in `cleaned_data/`")

    with tabs[1]:
        if data["ipo_tweets"] is not None:
            df = data["ipo_tweets"]
            c1, c2 = st.columns(2)
            c1.metric("Tweets", len(df))
            c2.metric("Columns", len(df.columns))
            text_c = next((c for c in df.columns
                           if "text" in c.lower() or "tweet" in c.lower()), df.columns[0])
            search = st.text_input("Search tweets", "", key="tweet_s")
            disp = df[df[text_c].astype(str).str.contains(search, case=False)] if search else df
            st.dataframe(disp.head(30), use_container_width=True)
        else:
            st.warning("`ipo_tweets_clean.csv` not found in `cleaned_data/`")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Videos**")
            if data["yt_videos"] is not None:
                st.metric("Videos", len(data["yt_videos"]))
                st.dataframe(data["yt_videos"].head(15), use_container_width=True)
            else:
                st.warning("`youtube_videos_zomato.csv` not found")
        with c2:
            st.markdown("**Comments**")
            if data["yt_comments"] is not None:
                st.metric("Comments", len(data["yt_comments"]))
                st.dataframe(data["yt_comments"].head(15), use_container_width=True)
            else:
                st.warning("`youtube_comments_zomato.csv` not found")
                st.warning("`youtube_comments_zomato.csv` not found")
