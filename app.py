import streamlit as st
import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis - Review Predictor",
    page_icon="💬",
    layout="centered"
)

# Text preprocessing function (same as in notebook)
def preprocess_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('logistic_regression.pkl')
        svm_model = joblib.load('svm_model.pkl')
        # Try to load vectorizer if exists
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
        except:
            # If vectorizer doesn't exist, create a new one
            # You'll need to retrain or save the vectorizer from your notebook
            vectorizer = None
        return lr_model, svm_model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

lr_model, svm_model, vectorizer = load_models()

# Header
st.title("🎯 Sentiment Analysis Review")
st.markdown("### Predict Review: Positive or Negative")
st.markdown("---")

# Input area
st.subheader("📝 Enter Your Review")
review_input = st.text_area(
    "Type your review here:",
    height=150,
    placeholder="Example: This product is amazing and high quality!"
)

# Prediction button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Predict Sentiment", use_container_width=True)

# Prediction process
if predict_button:
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing sentiment..."):
            if lr_model is None or svm_model is None:
                st.error("❌ Models could not be loaded. Make sure logistic_regression.pkl and svm_model.pkl files exist.")
            elif vectorizer is None:
                st.warning("⚠️ Vectorizer not found. Need to save from notebook.")
                st.info("""
                In your notebook, add this code after training the model:
                ```python
                import joblib
                joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
                ```
                """)
            else:
                try:
                    # Preprocess input
                    cleaned_text = preprocess_text(review_input)
                    
                    # Vectorize
                    vectorized_input = vectorizer.transform([cleaned_text])
                    
                    # Predict with both models
                    lr_prediction = lr_model.predict(vectorized_input)[0]
                    svm_prediction = svm_model.predict(vectorized_input)[0]
                    
                    # Get confidence scores
                    try:
                        # Logistic Regression probability
                        lr_proba = lr_model.predict_proba(vectorized_input)[0]
                        lr_confidence = lr_proba[1] if lr_prediction == 1 else lr_proba[0]
                    except:
                        lr_confidence = 0.85
                    
                    try:
                        # SVM decision function
                        svm_decision = svm_model.decision_function(vectorized_input)[0]
                        svm_confidence = 1 / (1 + np.exp(-svm_decision))  # Sigmoid
                        if svm_prediction == 0:
                            svm_confidence = 1 - svm_confidence
                    except:
                        svm_confidence = 0.85
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results - Model Comparison")
                    
                    # Create two columns for side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔵 Logistic Regression")
                        st.markdown("**F1-Score: 91.30%**")
                        if lr_prediction == 1:  # Positive
                            st.success("### ✅ POSITIVE")
                            st.markdown(f"**Confidence:** {lr_confidence:.2%}")
                        else:  # Negative
                            st.error("### ❌ NEGATIVE")
                            st.markdown(f"**Confidence:** {lr_confidence:.2%}")
                    
                    with col2:
                        st.markdown("#### 🟠 SVM")
                        st.markdown("**F1-Score: 91.09%**")
                        if svm_prediction == 1:  # Positive
                            st.success("### ✅ POSITIVE")
                            st.markdown(f"**Confidence:** {svm_confidence:.2%}")
                        else:  # Negative
                            st.error("### ❌ NEGATIVE")
                            st.markdown(f"**Confidence:** {svm_confidence:.2%}")
                    
                    # Check if predictions differ
                    if lr_prediction != svm_prediction:
                        st.warning("⚠️ **Models Disagree!** This review is challenging to classify.")
                        st.info(f"LR predicts: {'Positive' if lr_prediction == 1 else 'Negative'} | SVM predicts: {'Positive' if svm_prediction == 1 else 'Negative'}")
                    else:
                        st.success("✅ **Both models agree!** High confidence prediction.")
                        if lr_prediction == 1:
                            st.balloons()
                    
                    # Display the review
                    with st.expander("📄 Analyzed Review"):
                        st.markdown(f"**Original:** {review_input}")
                        st.markdown(f"**Cleaned:** {cleaned_text}")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Make sure models and vectorizer are compatible.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **How to Use:**
    1. Type or paste review in text area
    2. Click "Predict Sentiment" button
    3. View prediction results
    
    **Sentiment Classes:**
    - ✅ **Positive**: Positive review
    - ❌ **Negative**: Negative review
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Status")
    if lr_model is None or svm_model is None:
        st.error("❌ Models could not be loaded")
    elif vectorizer is None:
        st.warning("⚠️ Vectorizer not available")
        st.info("Need to save vectorizer from notebook")
    else:
        st.success("✅ Models & Vectorizer Ready!")
        st.info("**Logistic Regression**\nF1-Score: 91.30%")
        st.info("**SVM (LinearSVC)**\nF1-Score: 91.09%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Sentiment Analysis - NLP Project</div>",
    unsafe_allow_html=True
)
