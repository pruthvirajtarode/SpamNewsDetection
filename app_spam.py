import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import joblib, os

st.set_page_config(page_title="Spam News Detection", page_icon="📰", layout="wide")

st.title("📰 Spam News Detection")

uploaded_csv = st.sidebar.file_uploader("Upload Dataset (CSV with 'text' and 'label')", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    df = pd.read_csv("spam_news_1000.csv")

tab1, tab2 = st.tabs(["Train & Evaluate","Predictions"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Train / Re-train"):
        if 'text' in df.columns and 'label' in df.columns:
            Xtr, Xval, ytr, yval = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
            pipe = Pipeline([('tfidf',TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
                             ('clf',LogisticRegression(max_iter=400))])
            pipe.fit(Xtr,ytr); preds = pipe.predict(Xval)
            acc = accuracy_score(yval,preds)
            prec, rec, f1,_ = precision_recall_fscore_support(yval,preds,average='binary')
            cm = confusion_matrix(yval,preds)
            st.success(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real','Spam'])
            disp.plot(ax=ax)
            st.pyplot(fig)
            os.makedirs("models", exist_ok=True)
            joblib.dump(pipe,"models/model.joblib")
        else:
            st.error("Dataset must have 'text' and 'label' columns.")

with tab2:
    st.subheader("Live Predictions")
    txt = st.text_area("Enter one text per line",
                       "Congratulations! You won a prize\nGovernment launches new education policy")
    uploaded_pred_file = st.file_uploader("Or upload a CSV with 'text' column", type=["csv"], key="pred")

    if st.button("Predict"):
        if os.path.exists("models/model.joblib"):
            pipe = joblib.load("models/model.joblib")
            results = pd.DataFrame()

            if txt.strip():
                lines = [t.strip() for t in txt.splitlines() if t.strip()]
                preds = pipe.predict(lines)
                results = pd.DataFrame({"text": lines, "prediction": preds}).replace({"prediction":{0:"REAL (0)",1:"SPAM (1)"}})

            if uploaded_pred_file is not None:
                pred_df = pd.read_csv(uploaded_pred_file)
                if "text" in pred_df.columns:
                    preds = pipe.predict(pred_df['text'].astype(str))
                    pred_df["prediction"] = preds
                    pred_df = pred_df.replace({"prediction":{0:"REAL (0)",1:"SPAM (1)"}})
                    results = pd.concat([results, pred_df], ignore_index=True)
                else:
                    st.error("CSV must contain a 'text' column.")

            if not results.empty:
                st.dataframe(results)
                st.download_button("Download predictions CSV",
                                   results.to_csv(index=False).encode(),
                                   "predictions.csv","text/csv")
        else:
            st.warning("Train a model first in the Train & Evaluate tab.")
