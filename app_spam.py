import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spam News Detection", page_icon="📰", layout="wide")

st.markdown("""
    <style>
        body {background-color: #0f172a; color: white;}
        .main {background: linear-gradient(135deg, #1e3a8a, #6d28d9); color: white; padding: 20px; border-radius: 12px;}
        h1, h2, h3, h4, h5 {color: white !important;}
        .stButton>button {
            background: linear-gradient(135deg, #3b82f6, #9333ea);
            color: white; border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: bold; border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #9333ea, #3b82f6);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Spam News Detection")

st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (with 'text' and 'label')", type=["csv"])
validation_split = st.sidebar.slider("Validation split", 0.1, 0.5, 0.2, step=0.05)
max_features = st.sidebar.slider("Max features", 5000, 50000, 20000, step=5000)
ngram_range = st.sidebar.selectbox("Max n-gram", [1, 2, 3])
random_state = st.sidebar.number_input("Random state", value=42)

tab1, tab2, tab3 = st.tabs(["📊 Train & Evaluate", "🤖 Predictions", "ℹ️ About"])

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None
    st.session_state.df = None

def load_and_train(data):
    X = data['text']
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state, stratify=y)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_range))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer
    st.session_state.df = data
    return acc, prec, rec, f1, preds, y_val

with tab1:
    st.subheader("📊 Dataset Preview")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        if "text" in df.columns and "label" in df.columns:
            st.success("✅ Dataset loaded. Model will train automatically.")
            acc, prec, rec, f1, preds, y_val = load_and_train(df)
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("Precision", f"{prec:.3f}")
            st.metric("Recall", f"{rec:.3f}")
            st.metric("F1-score", f"{f1:.3f}")
            cm = confusion_matrix(y_val, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real (0)", "Spam (1)"], yticklabels=["Real (0)", "Spam (1)"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
        else:
            st.error("❌ CSV must have 'text' and 'label' columns.")
    else:
        st.info("Upload a dataset in sidebar to start training.")

with tab2:
    st.subheader("🤖 Live Predictions")
    if st.session_state.model:
        user_input = st.text_area("Enter news text to check if it's SPAM or REAL:")
        if st.button("Predict"):
            if user_input.strip():
                vect = st.session_state.vectorizer.transform([user_input])
                pred = st.session_state.model.predict(vect)[0]
                result = "🚨 SPAM / FAKE (1)" if pred == 1 else "✅ REAL (0)"
                st.success(f"Prediction: {result}")
    else:
        st.warning("⚠️ Please upload a dataset first in 'Train & Evaluate'.")

with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
        This project detects **Spam/Fake News vs Real News** using:
        - TF-IDF (Term Frequency–Inverse Document Frequency) for feature extraction  
        - Logistic Regression for classification  

        Built with **Python, Scikit-learn, Pandas, and Streamlit**.

        Features:
        - 📊 Interactive training & evaluation
        - 📈 Performance visualization
        - 🤖 Live predictions  

        🔗 Links:  
        - [GitHub Repo](https://github.com/pruthvirajtarode/SpamNewsDetection)  
        - [Live App](https://spamnewsdetection-99auakb9rgcjtevzkh8xa7.streamlit.app/)  
        - [LinkedIn](https://www.linkedin.com/in/pruthviraj-tarode-616ab1258/)  
    """)
