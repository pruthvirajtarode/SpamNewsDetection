# streamlit run app_spam.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import joblib, os, time

# Page config
st.set_page_config(page_title="Spam News Detection", page_icon="📰", layout="wide")

# --- Footer details (update with your links) ---
AUTHOR_NAME = "Pruthviraj Tarode"
LINKEDIN_URL = "https://www.linkedin.com/"   # replace with your LinkedIn
GITHUB_URL   = "https://github.com/pruthvirajtarode"

# --- CSS Styling ---
st.markdown("""
<style>
    .main { padding: 0 2rem; }
    .hero-wrap {
        position: relative; overflow: hidden;
        border-radius: 0 0 28px 28px;
        background: linear-gradient(120deg,#2563eb 0%, #7c3aed 100%);
        color: white; padding: 38px 20px; margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(79,70,229,.35);
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .hero-sub { color: #e9eafc; margin-top: 6px; font-size: 1.05rem; }
    .glow { height: 3px; width: 100%; background: linear-gradient(90deg,#93c5fd, #c4b5fd, #93c5fd);
            filter: blur(0.6px); opacity: .9; border-radius: 999px; margin-top: 14px; }
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, rgba(248,250,252,.9) 0%, rgba(238,242,255,.9) 100%);
        backdrop-filter: blur(6px); border-right: 1px solid #e5e7eb;
    }
    .stButton>button {
        background: linear-gradient(90deg,#3b82f6,#7c3aed);
        color:#fff; border:none; padding:12px 20px;
        border-radius:14px; font-weight:700; font-size:1rem;
        box-shadow: 0 6px 15px rgba(79,70,229,.35); transition: all .25s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); }
    .card { border-radius: 16px; padding: 18px; background: #ffffff;
            border: 1px solid #e5e7eb; box-shadow: 0 8px 20px rgba(0,0,0,.06); }
    .metric {
        border-radius: 16px; padding: 18px; background: #ffffff;
        border: 2px solid transparent; background-clip: padding-box, border-box; background-origin: border-box;
        background-image: linear-gradient(#fff,#fff), linear-gradient(90deg,#3b82f6,#7c3aed);
        box-shadow: 0 6px 18px rgba(0,0,0,.06); text-align:center;
    }
    .metric .label { color:#6b7280; font-size:.9rem; margin-bottom:6px; }
    .metric .value { font-size:1.9rem; font-weight:800; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding:10px 20px; border-radius:12px 12px 0 0;
        background:#f3f4f6; color:#374151; font-weight:600;
    }
    .stTabs [aria-selected="true"] {
        background:#fff !important; border-bottom: 3px solid;
        border-image: linear-gradient(90deg,#3b82f6,#7c3aed) 1; color:#111827;
    }
    .footer {
        background: linear-gradient(90deg,#1e3a8a,#6d28d9);
        color:white; text-align:center; padding:12px;
        border-radius:12px; margin-top:20px; font-size:.9rem;
    }
    .footer a { color: #c7d2fe; text-decoration: none; margin: 0 .5rem; }
    .footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Hero Banner ---
st.markdown('<div class="hero-wrap"><div class="hero-title">📰 Spam News Detection</div><div class="hero-sub">Interactive training, evaluation & live predictions with TF-IDF + Logistic Regression</div><div class="glow"></div></div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Controls")
uploaded_csv = st.sidebar.file_uploader("Upload Dataset (CSV with 'text' and 'label')", type=["csv"])
test_size = st.sidebar.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
ngram_max = st.sidebar.selectbox("Max n-gram", [1, 2])
max_features = st.sidebar.select_slider("Max features", options=[5000,10000,20000,30000,50000], value=20000)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# --- Load data ---
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    default_path = os.path.join("spam_news_1000.csv")
    df = pd.read_csv(default_path) if os.path.exists(default_path) else pd.DataFrame({"text":[],"label":[]})

tab1, tab2, tab3 = st.tabs(["🧪 Train & Evaluate","🔮 Predictions","ℹ️ About"])

def animate_value(target: float, decimals: int = 3, duration: float = 0.6):
    ph = st.empty(); steps = 20
    for i in range(steps+1):
        val = target * (i/steps)
        ph.markdown(f'<div class="value">{val:.{decimals}f}</div>', unsafe_allow_html=True)
        time.sleep(duration/steps)
    return ph

# --- Tab 1: Training ---
with tab1:
    st.subheader("Dataset Preview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Train / Re-train"):
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("Dataset must contain 'text' and 'label' columns.")
        else:
            data = df.dropna(subset=['text','label']).copy()
            data['text'] = data['text'].astype(str); data['label'] = data['label'].astype(int)
            Xtr, Xval, ytr, yval = train_test_split(data['text'], data['label'],
                                                    test_size=test_size, random_state=random_state, stratify=data['label'])
            pipe = Pipeline([('tfidf',TfidfVectorizer(max_features=max_features, ngram_range=(1,ngram_max))),
                             ('clf',LogisticRegression(max_iter=400))])
            with st.spinner("Training model..."):
                pipe.fit(Xtr,ytr); preds = pipe.predict(Xval)

            acc = accuracy_score(yval,preds)
            prec, rec, f1,_ = precision_recall_fscore_support(yval,preds,average='binary',zero_division=0)
            cm = confusion_matrix(yval,preds)

            os.makedirs("models",exist_ok=True); os.makedirs("outputs",exist_ok=True)
            joblib.dump(pipe,"models/model.joblib")
            with open("outputs/metrics.txt","w") as f:
                f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
                f.write("Confusion Matrix:\n"+str(cm))

            c1,c2,c3,c4 = st.columns(4)
            for col,label,val in zip([c1,c2,c3,c4],["Accuracy","Precision","Recall","F1-score"],[acc,prec,rec,f1]):
                with col:
                    st.markdown('<div class="metric"><div class="label">'+label+'</div>', unsafe_allow_html=True)
                    animate_value(val, decimals=3, duration=0.8)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Confusion Matrix")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real(0)','Spam(1)'])
            disp.plot(ax=ax)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Predictions ---
with tab2:
    st.subheader("Live Predictions")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Manual text input
    txt = st.text_area("Enter one text per line",
                       "Congratulations! You have won a free smartphone\nGovernment announces new policy on digital education",
                       height=150)

    # File upload for bulk predictions
    uploaded_pred_file = st.file_uploader("Or upload a CSV file with a 'text' column for bulk predictions", type=["csv"])

    if st.button("Predict"):
        if not os.path.exists("models/model.joblib"):
            st.warning("Train a model first in the Train & Evaluate tab.")
        else:
            pipe = joblib.load("models/model.joblib")

            results = pd.DataFrame()

            # Case 1: Manual text input
            if txt.strip():
                lines = [t.strip() for t in txt.splitlines() if t.strip()]
                preds = pipe.predict(lines)
                results = pd.DataFrame({"text": lines,
                                        "prediction": preds}).replace({"prediction": {0:"REAL (0)",1:"SPAM/FAKE (1)"}})

            # Case 2: CSV upload
            if uploaded_pred_file is not None:
                pred_df = pd.read_csv(uploaded_pred_file)
                if "text" not in pred_df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    preds = pipe.predict(pred_df['text'].astype(str))
                    pred_df["prediction"] = preds
                    pred_df = pred_df.replace({"prediction": {0:"REAL (0)",1:"SPAM/FAKE (1)"}})
                    results = pd.concat([results, pred_df], ignore_index=True)

            # Show results
            if not results.empty:
                st.dataframe(results, use_container_width=True)
                st.download_button("Download predictions CSV",
                                   results.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: About ---
with tab3:
    st.markdown("""
**How it works**  
This app uses TF-IDF features and a Logistic Regression classifier to detect spammy or fake news text.
- Use the packaged dataset (`spam_news_1000.csv`) or upload your own (`text`,`label`).
- Tune n-grams and feature size.
- Train, view metrics & confusion matrix, then try live predictions.
""")

# --- Footer ---
footer_html = """
<div class='footer'>
  © 2025 {AUTHOR_NAME} — Spam News Detection •
  <a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a> •
  <a href='{GITHUB_URL}' target='_blank'>GitHub</a>
</div>
""".format(AUTHOR_NAME=AUTHOR_NAME, LINKEDIN_URL=LINKEDIN_URL, GITHUB_URL=GITHUB_URL)

st.markdown(footer_html, unsafe_allow_html=True)
