import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Page Config -------------------
st.set_page_config(page_title="Spam News Detection", page_icon="📰", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    /* Header */
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Card Style */
    .card {
        background-color: white;
        color: black;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }
    /* Footer */
    footer {
        text-align: center;
        margin-top: 3rem;
        color: #ddd;
    }
    footer a {
        color: #fff;
        text-decoration: none;
        margin: 0 8px;
    }
    footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- App Title -------------------
st.markdown("<div class='main-header'>📰 Spam News Detection</div>", unsafe_allow_html=True)

# ------------------- Sidebar Controls -------------------
st.sidebar.header("Upload Dataset (CSV with 'text' and 'label')")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

val_size = st.sidebar.slider("Validation split", 0.1, 0.5, 0.2, 0.05)
ngram = st.sidebar.selectbox("Max n-gram", [1, 2])
max_features = st.sidebar.slider("Max features", 5000, 30000, 20000, 1000)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# ------------------- Load Data -------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "text": [
            "Congratulations! You have won a free iPhone",
            "Government announces new digital education policy",
            "Urgent: Verify your bank account immediately",
            "NASA successfully launches satellite for weather monitoring",
            "Claim your free gift card before midnight"
        ],
        "label": [1, 0, 1, 0, 1]
    })

# ------------------- Tabs -------------------
tab1, tab2, tab3 = st.tabs(["📊 Train & Evaluate", "🤖 Predictions", "ℹ️ About"])

# ------------------- Train & Evaluate -------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Train / Re-train"):
        X_train, X_val, y_train, y_val = train_test_split(
            df['text'], df['label'], test_size=val_size, random_state=random_state
        )
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        st.markdown("### 📈 Performance Metrics")
        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**Precision:** {prec:.3f}")
        st.write(f"**Recall:** {rec:.3f}")
        st.write(f"**F1 Score:** {f1:.3f}")

        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "SPAM"], yticklabels=["REAL", "SPAM"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Predictions -------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live Predictions")
    user_input = st.text_area("Enter news text to check if it's SPAM or REAL:")

    if st.button("Predict"):
        if 'model' in locals():
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]
            label = "🚨 SPAM / FAKE" if prediction == 1 else "✅ REAL"
            st.markdown(f"### Prediction: {label}")
        else:
            st.warning("⚠️ Please train the model first in the 'Train & Evaluate' tab.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- About -------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This Project")
    st.write("""
    This project uses **TF-IDF + Logistic Regression** to classify news text as **REAL (0)** or **SPAM/FAKE (1)**.  
    Built with **Python, Scikit-learn, Pandas, and Streamlit**, it allows:  
    - 📊 Interactive training and evaluation  
    - 📈 Performance visualization  
    - 🤖 Live predictions  

    ✨ The UI is styled with custom CSS (gradient background, modern buttons, card layout).
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown(
    """
    <footer>
        © 2025 Pruthviraj Tarode — Spam News Detection |
        <a href="https://www.linkedin.com/in/pruthviraj-tarode-616ab1258/" target="_blank">LinkedIn</a> •
        <a href="https://github.com/pruthvirajtarode/SpamNewsDetection" target="_blank">GitHub</a>
    </footer>
    """,
    unsafe_allow_html=True
)
