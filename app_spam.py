import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------ CSS Styling ------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        font-family: 'Poppins', sans-serif;
        color: #222;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
    }

    /* Card */
    .card {
        background: #ffffff;
        color: #222;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.25);
        margin-top: 1rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.07);
        box-shadow: 0px 4px 14px rgba(0,0,0,0.5);
    }

    /* Table */
    .dataframe {
        background: white;
        color: black;
        border-radius: 12px;
    }

    /* Footer */
    footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #eee;
        font-size: 0.9rem;
    }
    footer a { color: #fff; text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown("<div class='main-header'>📰 Spam News Detection</div>", unsafe_allow_html=True)

# ------------------ Tabs ------------------
tabs = st.tabs(["📊 Train & Evaluate", "🔮 Predictions", "ℹ️ About"])

# ------------------ Train & Evaluate ------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")

    uploaded_file = st.file_uploader("Upload Dataset (CSV with 'text' and 'label')", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("spam_news_1000.csv")

    st.dataframe(data.head())

    validation_split = st.slider("Validation split", 0.1, 0.5, 0.2)
    max_features = st.slider("Max features", 500, 20000, 5000)
    ngram_range = st.selectbox("Max n-gram", [1, 2])

    if st.button("🚀 Train / Re-train"):
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['label'], test_size=validation_split, random_state=42
        )
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_range))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        st.success("✅ Model trained successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
        col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
        col3.metric("Recall", round(recall_score(y_test, y_pred), 3))
        col4.metric("F1-score", round(f1_score(y_test, y_pred), 3))

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Predictions ------------------
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Live Predictions")

    user_input = st.text_area("✍️ Enter one text per line")

    if st.button("✨ Predict"):
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv("spam_news_1000.csv")

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(data['text'])
        model = LogisticRegression().fit(X, data['label'])

        if user_input.strip():
            lines = user_input.split("\n")
            preds = model.predict(vectorizer.transform(lines))
            results = pd.DataFrame({"text": lines, "prediction": preds})
            results['prediction'] = results['prediction'].map({
                0: "✅ REAL (0)",
                1: "❌ SPAM/FAKE (1)"
            })
            st.table(results)
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ About ------------------
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This Project")
    st.write("""
        This project uses **TF-IDF + Logistic Regression** to classify news text as **REAL (0)** or **SPAM/FAKE (1)**.  
        Built with **Python, Scikit-learn, Pandas, and Streamlit**, it allows:
        - 📊 Interactive training and evaluation  
        - 📈 Performance visualization  
        - 🔮 Live predictions  

        ✨ The UI is styled with custom CSS (gradient background, white cards, modern buttons).
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown(
    """
    <footer>
        © 2025 Pruthviraj Tarode — Spam News Detection |
        <a href="https://www.linkedin.com/in/pruthvirajtarode" target="_blank">LinkedIn</a> •
        <a href="https://github.com/pruthvirajtarode" target="_blank">GitHub</a>
    </footer>
    """,
    unsafe_allow_html=True
)
