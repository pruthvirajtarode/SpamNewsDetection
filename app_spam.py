import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Spam News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #1e3c72, #2a5298);
            color: white !important;
        }
        .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: white !important;
            font-weight: bold;
        }
        .stAlert {
            background-color: #fff3cd !important;
            color: black !important;
            border: 1px solid #ffeeba;
        }
        .custom-footer {
            text-align: center;
            color: white !important;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        }
        .custom-footer a {
            color: white !important;
            text-decoration: none;
            font-weight: bold;
        }
        .custom-footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Default Dataset
# -------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("spam_news_1000.csv")

# -------------------------------
# Train Model
# -------------------------------
def train_model(data, max_features=5000, ngram_range=(1,1)):
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    return model, vectorizer

# -------------------------------
# App Layout
# -------------------------------
st.title("📰 Spam News Detection")

tab1, tab2, tab3 = st.tabs(["📊 Train & Evaluate", "🤖 Predictions", "ℹ️ About"])

# -------------------------------
# Shared Variables
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = load_default_data()
    st.session_state.model, st.session_state.vectorizer = train_model(st.session_state.data)

# -------------------------------
# Tab 1: Train & Evaluate
# -------------------------------
with tab1:
    st.subheader("Upload Dataset (optional)")
    uploaded_file = st.file_uploader("Upload CSV with 'text' and 'label' columns", type="csv")

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        if "text" in new_data.columns and "label" in new_data.columns:
            st.session_state.data = new_data
            st.session_state.model, st.session_state.vectorizer = train_model(new_data)
            st.success("✅ Model retrained on uploaded dataset!")
        else:
            st.error("❌ CSV must contain 'text' and 'label' columns")

    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.data.head())

    X = st.session_state.data['text']
    y = st.session_state.data['label']
    preds = st.session_state.model.predict(st.session_state.vectorizer.transform(X))

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)

    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "SPAM"], yticklabels=["REAL", "SPAM"], ax=ax)
    st.pyplot(fig)

    # -------------------------------
    # Download Trained Model
    # -------------------------------
    st.subheader("📥 Download Trained Model")
    model_bytes = pickle.dumps({
        "model": st.session_state.model,
        "vectorizer": st.session_state.vectorizer
    })
    st.download_button(
        label="💾 Download Model (.pkl)",
        data=model_bytes,
        file_name="spam_news_model.pkl",
        mime="application/octet-stream"
    )

# -------------------------------
# Tab 2: Predictions
# -------------------------------
with tab2:
    st.subheader("Live Predictions")
    user_input = st.text_area("✍️ Enter news text to check if it's SPAM or REAL:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            input_features = st.session_state.vectorizer.transform([user_input])
            prediction = st.session_state.model.predict(input_features)[0]
            if prediction == 1:
                st.error("🚨 This news looks like **SPAM/FAKE** ❌")
            else:
                st.success("✅ This news looks **REAL** 📰")

# -------------------------------
# Tab 3: About
# -------------------------------
with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    This project uses **TF-IDF + Logistic Regression** to classify news text as REAL (0) or SPAM/FAKE (1).  
    Built with **Python, Scikit-learn, Pandas, and Streamlit**, it allows:
    - 📊 Interactive training and evaluation  
    - 📈 Performance visualization  
    - 🤖 Live predictions  
    - 📂 Upload new dataset for retraining  
    - 💾 Download trained model for reuse  

    🔗 Links:  
    - [🌐 Live App](https://spamnewsdetection-99auakb9rgcjtevzkh8xa7.streamlit.app/)  
    - [💻 GitHub Repository](https://github.com/pruthvirajtarode/SpamNewsDetection)  
    - [👨‍💼 LinkedIn](https://www.linkedin.com/in/pruthviraj-tarode-616ab1258/)  
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="custom-footer">
© 2025 Pruthviraj Tarode — Spam News Detection | 
<a href="https://github.com/pruthvirajtarode/SpamNewsDetection" target="_blank">GitHub</a> • 
<a href="https://www.linkedin.com/in/pruthviraj-tarode-616ab1258/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
