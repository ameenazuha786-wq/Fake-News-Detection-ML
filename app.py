
import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Application title
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news headline or article below "
    "to predict whether it is Fake or Real."
)

# User input
news_text = st.text_area(
    "Enter News Article:",
    height=200,
    placeholder="Paste your news article here..."
)

# Prediction button
if st.button("Predict News", type="primary"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        # Convert input into TF-IDF features
        news_vector = vectorizer.transform([news_text])

        # Predict result
        prediction = model.predict(news_vector)[0]

        # Get prediction probability
        probabilities = model.predict_proba(news_vector)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == 0:
            st.error("⚠️ Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "This is a model prediction, not independent "
            "verification of the news."
        )

st.divider()

st.caption("Developed using Python, TF-IDF and Logistic Regression.")
