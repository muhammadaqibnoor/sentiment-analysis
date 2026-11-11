import streamlit as st
import joblib
import re
import html
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Intelligence",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Project Paths & Model Loading
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "logistic_regression_model.pkl"
)

tfidf = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

# -----------------------------
# Text Cleaning & Prediction
# -----------------------------
def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(review):
    cleaned = clean_text(review)
    features = tfidf.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probabilities) * 100

    return sentiment, confidence


# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #8b949e;
        font-size: 16px;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 15px;
    }
    .result-positive {
        background-color: rgba(40, 167, 69, 0.10);
    }
    .result-negative {
        background-color: rgba(220, 53, 69, 0.10);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🎬 Sentiment AI")
    st.caption("IMDb Movie Review Analysis")

    st.divider()

    st.subheader("About the Project")
    st.write(
        "An NLP machine learning application "
        "that classifies movie reviews as Positive or Negative."
    )

    st.subheader("Dataset")
    st.write("IMDb Large Movie Review Dataset")
    st.write("50,000 labeled movie reviews")

    st.divider()

    st.subheader("Best Model")
    st.success("Logistic Regression")

    st.caption("TF-IDF text feature extraction")

# -----------------------------
# Main Dashboard
# -----------------------------
st.markdown(
    '<div class="main-title">IMDb Movie Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand movie reviews through Natural Language Processing and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "89.93%")
col2.metric("Precision", "89.59%")
col3.metric("Recall", "90.36%")
col4.metric("ROC-AUC", "96.30%")

st.divider()

# -----------------------------
# Prediction Interface
# -----------------------------
st.subheader("🔍 Analyze a Movie Review")

st.write(
    "Enter a movie review below. The trained model will predict its sentiment."
)

review = st.text_area(
    "Your Movie Review",
    height=170,
    placeholder="Example: The movie was fantastic, entertaining and beautifully acted..."
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):

    if review.strip():

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction Result")

        if sentiment == "Positive":
            st.markdown(
                '<div class="result-card result-positive">'
                '<h2>😊 Positive Sentiment</h2>'
                '<p>The model predicts that this review expresses a positive sentiment.</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card result-negative">'
                '<h2>😞 Negative Sentiment</h2>'
                '<p>The model predicts that this review expresses a negative sentiment.</p>'
                '</div>',
                unsafe_allow_html=True
            )

        st.metric(
            "Model Confidence (Predicted Class Probability)",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

    else:
        st.warning("Please enter a movie review first.")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Built with Python, Scikit-learn, TF-IDF and Streamlit | "
    "IMDb Sentiment Analysis Portfolio Project"
)

st.caption(
    "Note: Model confidence is a predicted probability and is not necessarily calibrated."
)