import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Spam News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# Load Default Dataset
# -------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("spam_news_1000.csv")

# -------------------------------
# Train Model Function
# -------------------------------
@st.cache_resource
def train_model(df, test_size=0.2, max_features=20000, ngram_range=(1,1), random_state=42):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred)
    }

    return model, vectorizer, metrics

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Upload Dataset (CSV with 'text' and 'label')")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

validation_split = st.sidebar.slider("Validation split", 0.1, 0.5, 0.2, 0.05)
ngram_max = st.sidebar.selectbox("Max n-gram", [1, 2])
max_features = st.sidebar.slider("Max features", 1000, 20000, 20000, 1000)
random_state = st.sidebar.number_input("Random state", value=42)

# -------------------------------
# Load Dataset
# -------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

st.session_state["df"] = df

# -------------------------------
# Auto Train on Startup
# -------------------------------
if "model" not in st.session_state:
    model, vectorizer, metrics = train_model(
        df, test_size=validation_split, max_features=max_features,
        ngram_range=(1, ngram_max), random_state=random_state
    )
    st.session_state["model"] = model
    st.session_state["vectorizer"] = vectorizer
    st.session_state["metrics"] = metrics

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["📊 Train & Evaluate", "🔮 Predictions", "ℹ️ About"])

# -------------------------------
# Train & Evaluate Tab
# -------------------------------
with tabs[0]:
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Performance Metrics (Auto-trained)")
    metrics = st.session_state["metrics"]
    st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
    st.write(f"**Precision:** {metrics['precision']:.3f}")
    st.write(f"**Recall:** {metrics['recall']:.3f}")
    st.write(f"**F1-score:** {metrics['f1']:.3f}")

    # Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics["confusion"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------------------
# Predictions Tab
# -------------------------------
with tabs[1]:
    st.header("Live Predictions")
    input_text = st.text_area("Enter news text to check if it's SPAM or REAL:", height=150)

    if st.button("Predict"):
        if input_text.strip():
            vectorizer = st.session_state["vectorizer"]
            model = st.session_state["model"]

            input_tfidf = vectorizer.transform([input_text])
            prediction = model.predict(input_tfidf)[0]

            label = "🚨 SPAM/FAKE (1)" if prediction == 1 else "✅ REAL (0)"
            st.success(f"Prediction: {label}")
        else:
            st.warning("Please enter text for prediction.")

# -------------------------------
# About Tab
# -------------------------------
with tabs[2]:
    st.header("ℹ️ About This Project")
    st.markdown("""
    This project uses **TF-IDF + Logistic Regression** to classify news text as REAL (0) or SPAM/FAKE (1).  
    Built with **Python, Scikit-learn, Pandas, and Streamlit**, it allows:

    - 📊 Interactive training and evaluation  
    - 📈 Performance visualization  
    - 🔮 Live predictions  

    ✨ The UI is styled with modern gradient theme + custom layout.  

    🔗 **GitHub Repo:** [Spam News Detection](https://github.com/pruthvirajtarode/SpamNewsDetection)  
    🔗 **Live App:** [Click Here](https://spamnewsdetection-99auakb9rgcjtevzkh8xa7.streamlit.app/)  
    """)
