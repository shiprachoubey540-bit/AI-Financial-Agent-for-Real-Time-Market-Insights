"""
============================================================
AI FINANCIAL AGENT — ZOMATO IPO SENTIMENT ANALYSIS
DEPLOYMENT: app.py  (Flask REST API + Web UI)
============================================================
Run:
    python app.py

Endpoints:
    GET  /              → Web dashboard UI
    POST /predict       → Single text prediction (JSON)
    POST /predict_batch → Batch prediction (JSON list)
    GET  /health        → Health check
    GET  /models        → List loaded models
============================================================
"""

from flask import Flask, request, jsonify, render_template_string
import pickle, os, re, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Optional: VADER for confidence scores ─────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except ImportError:
    VADER_OK = False

app = Flask(__name__)

# ── Model registry ─────────────────────────────────────────────────────────
MODEL_DIR   = "models"
LABEL_MAP   = {0: "Negative 📉", 1: "Neutral ➡️", 2: "Positive 📈"}
LABEL_COLOR = {0: "#E74C3C",     1: "#F39C12",    2: "#27AE60"}

loaded_models     = {}
tfidf_vectorizer  = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_models():
    """Load all .pkl models and the TF-IDF vectorizer at startup."""
    global tfidf_vectorizer

    if not os.path.exists(MODEL_DIR):
        print(f"[WARN] {MODEL_DIR}/ not found — run save_models_notebook_cell.py first")
        return

    # Load vectorizer
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        print("✅ TF-IDF vectorizer loaded")
    else:
        print("[WARN] tfidf_vectorizer.pkl not found")

    # Load all classifier pkl files
    model_files = {
        "logistic_regression": "logistic_regression.pkl",
        "svm":                 "svm_model.pkl",
        "naive_bayes":         "naive_bayes.pkl",
        "random_forest":       "random_forest.pkl",
        "gradient_boosting":   "gradient_boosting.pkl",
    }

    for name, fname in model_files.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                loaded_models[name] = pickle.load(f)
            print(f"✅ Loaded: {name}")
        else:
            print(f"[SKIP] Not found: {fname}")

    print(f"\n📦 {len(loaded_models)} model(s) ready for inference")


def predict_sentiment(text: str, model_name: str = "logistic_regression"):
    """Core prediction function — returns label, confidence, vader score."""
    if not loaded_models:
        return {"error": "No models loaded. Run save_models_notebook_cell.py first."}

    if model_name not in loaded_models:
        model_name = list(loaded_models.keys())[0]

    clean   = clean_text(text)
    clf     = loaded_models[model_name]

    if tfidf_vectorizer:
        X = tfidf_vectorizer.transform([clean])
    else:
        return {"error": "TF-IDF vectorizer not loaded."}

    pred    = int(clf.predict(X)[0])
    label   = LABEL_MAP[pred]
    color   = LABEL_COLOR[pred]

    # Confidence
    if hasattr(clf, "predict_proba"):
        proba   = clf.predict_proba(X)[0].tolist()
        confidence = round(max(proba) * 100, 1)
    elif hasattr(clf, "decision_function"):
        scores  = clf.decision_function(X)[0]
        exp     = np.exp(scores - scores.max())
        proba   = (exp / exp.sum()).tolist()
        confidence = round(max(proba) * 100, 1)
    else:
        proba, confidence = [], 0.0

    # VADER supplementary
    vader_compound = 0.0
    if VADER_OK:
        vader_compound = round(vader.polarity_scores(text)["compound"], 4)

    # Financial signal
    if pred == 2:
        signal = "BULLISH 🟢 — Positive market sentiment detected"
    elif pred == 0:
        signal = "BEARISH 🔴 — Negative market sentiment detected"
    else:
        signal = "NEUTRAL ⚪ — Market sentiment is mixed"

    return {
        "input_text":     text,
        "cleaned_text":   clean,
        "model_used":     model_name,
        "sentiment_code": pred,
        "sentiment":      label,
        "color":          color,
        "confidence":     confidence,
        "probabilities":  {
            "negative": round(proba[0] * 100, 1) if proba else 0,
            "neutral":  round(proba[1] * 100, 1) if proba else 0,
            "positive": round(proba[2] * 100, 1) if proba else 0,
        },
        "vader_compound": vader_compound,
        "financial_signal": signal,
    }


# ==========================================================================
# ROUTES
# ==========================================================================

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": list(loaded_models.keys()),
        "vectorizer":    tfidf_vectorizer is not None,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/models")
def list_models():
    return jsonify({
        "available_models": list(loaded_models.keys()),
        "default_model":    list(loaded_models.keys())[0] if loaded_models else None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "text": "...", "model": "logistic_regression" }
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    model_name = data.get("model", "logistic_regression")
    result     = predict_sentiment(data["text"], model_name)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    POST /predict_batch
    Body: { "texts": ["...", "..."], "model": "logistic_regression" }
    """
    data = request.get_json(force=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field (list) in request body"}), 400

    model_name = data.get("model", "logistic_regression")
    results    = [predict_sentiment(t, model_name) for t in data["texts"]]

    # Summary stats
    sentiments = [r["sentiment_code"] for r in results if "sentiment_code" in r]
    summary = {
        "total":    len(results),
        "positive": sentiments.count(2),
        "neutral":  sentiments.count(1),
        "negative": sentiments.count(0),
        "overall_signal": (
            "BULLISH 🟢" if sentiments.count(2) > sentiments.count(0)
            else "BEARISH 🔴" if sentiments.count(0) > sentiments.count(2)
            else "NEUTRAL ⚪"
        )
    }
    return jsonify({"summary": summary, "results": results})


# ==========================================================================
# WEB DASHBOARD UI
# ==========================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Financial Agent — Zomato IPO Sentiment</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0A0E1A;
    --card: #111827;
    --border: #1F2937;
    --accent: #F59E0B;
    --green: #10B981;
    --red: #EF4444;
    --yellow: #F59E0B;
    --text: #F9FAFB;
    --muted: #6B7280;
    --font: 'Space Grotesk', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
  }

  /* NAV */
  nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    background: rgba(17,24,39,0.8);
    backdrop-filter: blur(8px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { font-weight: 700; font-size: 1.1rem; letter-spacing: -0.02em; }
  .logo span { color: var(--accent); }
  .status-pill {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.8rem; color: var(--muted);
    background: var(--border); padding: 0.4rem 0.8rem; border-radius: 999px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
         animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* LAYOUT */
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }

  /* HERO */
  .hero { text-align: center; padding: 3rem 0 2rem; }
  .hero h1 {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700; letter-spacing: -0.03em;
    line-height: 1.1;
  }
  .hero h1 em { font-style: normal; color: var(--accent); }
  .hero p { color: var(--muted); margin-top: 1rem; font-size: 1.05rem; }

  /* CARDS */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }
  .card-title {
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 1rem;
  }

  /* GRID */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2rem; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-top: 1.5rem; }
  @media(max-width:700px){ .grid-2,.grid-3{ grid-template-columns:1fr; } }

  /* INPUT AREA */
  .input-section { margin-top: 2rem; }
  .input-row { display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; }
  textarea {
    flex: 1; min-width: 250px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text);
    font-family: var(--font); font-size: 0.95rem;
    padding: 1rem; resize: vertical; min-height: 100px;
    outline: none; transition: border-color 0.2s;
  }
  textarea:focus { border-color: var(--accent); }
  select {
    background: var(--card); border: 1px solid var(--border);
    color: var(--text); font-family: var(--font);
    padding: 0.6rem 1rem; border-radius: 8px;
    font-size: 0.9rem; cursor: pointer;
  }
  .btn {
    background: var(--accent); color: #000;
    border: none; border-radius: 8px;
    padding: 0.75rem 2rem; font-family: var(--font);
    font-weight: 700; font-size: 0.95rem; cursor: pointer;
    transition: transform 0.1s, opacity 0.2s;
    white-space: nowrap;
  }
  .btn:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* RESULT */
  #result-card {
    margin-top: 1.5rem; display: none;
    border-radius: 12px; overflow: hidden;
  }
  .result-header {
    padding: 1.2rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
  }
  .sentiment-badge {
    font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
  }
  .confidence {
    font-family: var(--mono); font-size: 0.9rem;
    background: rgba(0,0,0,0.3); padding: 0.4rem 0.8rem; border-radius: 6px;
  }
  .result-body { padding: 1.2rem 1.5rem; background: rgba(0,0,0,0.25); }
  .signal-box {
    font-weight: 600; font-size: 1rem;
    padding: 0.8rem 1rem; border-radius: 8px;
    background: rgba(255,255,255,0.06);
    margin-bottom: 1rem;
  }

  /* PROB BARS */
  .prob-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.4rem 0; }
  .prob-label { width: 70px; font-size: 0.82rem; color: var(--muted); }
  .prob-track {
    flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden;
  }
  .prob-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .prob-val { width: 42px; font-family: var(--mono); font-size: 0.78rem; text-align: right; }

  /* STAT CHIPS */
  .chip {
    background: var(--border); border-radius: 8px;
    padding: 1rem; text-align: center;
  }
  .chip-val { font-size: 1.5rem; font-weight: 700; }
  .chip-lbl { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }

  /* API DOCS */
  .api-section { margin-top: 3rem; }
  pre {
    background: #0D1117; border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem;
    font-family: var(--mono); font-size: 0.8rem;
    color: #79C0FF; overflow-x: auto; white-space: pre-wrap;
  }
  .method { font-size: 0.7rem; font-weight: 700; font-family: var(--mono);
            padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.5rem; }
  .post { background: #10B981; color: #000; }
  .get  { background: #3B82F6; color: #fff; }
  .endpoint-row { display: flex; align-items: center; margin: 0.5rem 0;
                  font-family: var(--mono); font-size: 0.85rem; }

  /* LOADER */
  .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid #000; border-top-color: transparent;
    border-radius: 50%; animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer {
    text-align: center; padding: 3rem 1rem 2rem;
    color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border);
    margin-top: 4rem;
  }
</style>
</head>
<body>

<nav>
  <div class="logo">AI <span>Financial</span> Agent</div>
  <div class="status-pill"><span class="dot"></span> Live · Zomato IPO Sentiment</div>
</nav>

<div class="container">

  <div class="hero">
    <h1>Real-Time <em>Market Sentiment</em><br>from Social Signals</h1>
    <p>Analyse tweets, reviews & news to generate financial market signals for Zomato IPO</p>
  </div>

  <!-- Stats row -->
  <div class="grid-3">
    <div class="chip">
      <div class="chip-val" id="stat-models">—</div>
      <div class="chip-lbl">Models Loaded</div>
    </div>
    <div class="chip">
      <div class="chip-val">3</div>
      <div class="chip-lbl">Sentiment Classes</div>
    </div>
    <div class="chip">
      <div class="chip-val" id="stat-vec">—</div>
      <div class="chip-lbl">Vectorizer</div>
    </div>
  </div>

  <!-- Input section -->
  <div class="input-section card" style="margin-top:1.5rem;">
    <div class="card-title">📊 Analyse Text</div>
    <div class="input-row">
      <textarea id="input-text"
        placeholder="Enter a tweet, review, or news headline about Zomato IPO...&#10;&#10;e.g. 'Zomato IPO subscribed 38x — huge investor confidence!'"></textarea>
      <div style="display:flex;flex-direction:column;gap:0.8rem;">
        <div>
          <label style="font-size:0.78rem;color:var(--muted);display:block;margin-bottom:0.4rem;">Model</label>
          <select id="model-select">
            <option value="logistic_regression">Logistic Regression</option>
            <option value="svm">Linear SVM</option>
            <option value="naive_bayes">Naive Bayes</option>
            <option value="random_forest">Random Forest</option>
            <option value="gradient_boosting">Gradient Boosting</option>
          </select>
        </div>
        <button class="btn" id="predict-btn" onclick="predictSentiment()">
          Analyse →
        </button>
      </div>
    </div>

    <!-- Result -->
    <div id="result-card">
      <div class="result-header" id="result-header">
        <div class="sentiment-badge" id="result-label">—</div>
        <div class="confidence" id="result-confidence">—</div>
      </div>
      <div class="result-body">
        <div class="signal-box" id="result-signal">—</div>
        <div style="font-size:0.8rem;color:var(--muted);margin-bottom:0.8rem;">Probability Distribution</div>
        <div class="prob-row">
          <span class="prob-label">Negative</span>
          <div class="prob-track"><div class="prob-fill" id="bar-neg" style="background:#EF4444;width:0%"></div></div>
          <span class="prob-val" id="val-neg">0%</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Neutral</span>
          <div class="prob-track"><div class="prob-fill" id="bar-neu" style="background:#F59E0B;width:0%"></div></div>
          <span class="prob-val" id="val-neu">0%</span>
        </div>
        <div class="prob-row">
          <span class="prob-label">Positive</span>
          <div class="prob-track"><div class="prob-fill" id="bar-pos" style="background:#10B981;width:0%"></div></div>
          <span class="prob-val" id="val-pos">0%</span>
        </div>
        <div style="margin-top:1rem;font-size:0.78rem;color:var(--muted);">
          VADER Score: <span id="result-vader" style="font-family:var(--mono);color:var(--text);">—</span>
          &nbsp;|&nbsp; Model: <span id="result-model" style="font-family:var(--mono);color:var(--text);">—</span>
        </div>
      </div>
    </div>
  </div>

  <!-- API Docs -->
  <div class="api-section">
    <div class="card-title" style="margin-bottom:1rem;">🔌 API Endpoints</div>
    <div class="card">
      <div class="endpoint-row"><span class="method post">POST</span>/predict</div>
      <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Zomato IPO oversubscribed!", "model": "logistic_regression"}'</pre>

      <div class="endpoint-row" style="margin-top:1rem;"><span class="method post">POST</span>/predict_batch</div>
      <pre>curl -X POST http://localhost:5000/predict_batch \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Great IPO!", "Bad listing day"], "model": "svm"}'</pre>

      <div class="endpoint-row" style="margin-top:1rem;"><span class="method get">GET</span>/health</div>
      <pre>curl http://localhost:5000/health</pre>
    </div>
  </div>

</div><!-- /container -->

<footer>AI Financial Agent · Zomato IPO Sentiment Analysis · Group Project</footer>

<script>
  // Load health on page load
  fetch('/health').then(r => r.json()).then(d => {
    document.getElementById('stat-models').textContent = d.models_loaded.length;
    document.getElementById('stat-vec').textContent    = d.vectorizer ? '✅ TF-IDF' : '❌ Missing';

    // Populate model select with actual loaded models
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    (d.models_loaded.length ? d.models_loaded : ['logistic_regression']).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
      sel.appendChild(opt);
    });
  }).catch(() => {
    document.getElementById('stat-models').textContent = '?';
    document.getElementById('stat-vec').textContent = '?';
  });

  async function predictSentiment() {
    const text  = document.getElementById('input-text').value.trim();
    const model = document.getElementById('model-select').value;
    const btn   = document.getElementById('predict-btn');

    if (!text) { alert('Please enter some text to analyse.'); return; }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text, model})
      });
      const d = await res.json();

      if (d.error) { alert(d.error); return; }

      // Update UI
      const card   = document.getElementById('result-card');
      const header = document.getElementById('result-header');
      card.style.display = 'block';
      header.style.background = d.color + '22';
      header.style.borderBottom = `2px solid ${d.color}`;

      document.getElementById('result-label').textContent      = d.sentiment;
      document.getElementById('result-confidence').textContent = `${d.confidence}% confidence`;
      document.getElementById('result-signal').textContent     = d.financial_signal;
      document.getElementById('result-vader').textContent      = d.vader_compound;
      document.getElementById('result-model').textContent      = d.model_used;

      const p = d.probabilities;
      ['neg','neu','pos'].forEach((k,i) => {
        const keys = ['negative','neutral','positive'];
        const val  = p[keys[i]] || 0;
        document.getElementById(`bar-${k}`).style.width = val + '%';
        document.getElementById(`val-${k}`).textContent = val + '%';
      });

    } catch(e) {
      alert('Prediction failed. Is the Flask server running?');
    } finally {
      btn.disabled = false;
      btn.innerHTML = 'Analyse →';
    }
  }

  // Allow Ctrl+Enter to submit
  document.getElementById('input-text').addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter') predictSentiment();
  });
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


# ==========================================================================
# STARTUP
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("  AI FINANCIAL AGENT — DEPLOYMENT SERVER")
    print("🚀"*30)
    load_models()
    print("\n  Dashboard → http://localhost:5000")
    print("  API Docs  → http://localhost:5000/health\n")
    app.run(debug=True, host="0.0.0.0", port=5000)