"""
predict.py - Prediction & Inference Module
============================================
Loads a trained model and vectorizer to make predictions on new text.
Uses the same text cleaning pipeline as training.

Educational Overview for AI/ML Students:
----------------------------------------
In a real-world machine learning system, the lifecycle is divided into two phases:
1. Training Time (Offline): Reading historical datasets, experimenting with architectures,
   tuning hyperparameters, and saving model weights/artifacts.
2. Inference Time (Online/Production): Serving predictions on live, unseen data in real time.

This module is the core inference engine of the Disaster Intelligence Engine:
- It deserializes the persisted model and TF-IDF vocabulary from disk.
- It applies the EXACT same text sanitization pipeline used during training to prevent skew.
- It calculates both the predicted class ("Informative" vs "Not Informative") and a calibrated
  confidence probability score.
- It serves as the backend utility that powers the Streamlit web application (`app/app.py`).

Usage:
    python src/predict.py
    
    Or import and use in the Streamlit app:
        from src.predict import load_model, predict, predict_batch
"""

import os
import sys
import joblib
from pathlib import Path

# Add project root to path so we can import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


def load_model(models_dir=None):
    """
    Load the trained model and TF-IDF vectorizer from disk.
    
    Why We Need the Fitted Vectorizer (Not a New One):
    --------------------------------------------------
    A TF-IDF vectorizer learns two things during training:
    1. A fixed vocabulary dictionary mapping words to specific column indices (e.g. 'shelter' -> column 1042).
    2. An Inverse Document Frequency (IDF) weight vector representing how rare each word was in the training corpus.
    
    If we instantiated a brand new vectorizer at inference time, it would have no vocabulary or completely
    different column mappings! The trained classifier expects feature column #1042 to represent the exact same
    word token it saw during training. Loading the serialized vectorizer ensures perfect feature alignment.
    
    Args:
        models_dir (str, optional): Directory containing model files.
                                     Defaults to models/ in project root.
    
    Returns:
        tuple: (model, vectorizer) — the loaded sklearn objects.
    
    Raises:
        FileNotFoundError: If model or vectorizer files are not found.
    """
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    
    model_path = os.path.join(models_dir, "disaster_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    
    # Ensure artifacts exist before attempting deserialization
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run 'python src/train.py' first to train the model."
        )
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}\n"
            "Please run 'python src/train.py' first to train the model."
        )
    
    # joblib.load reconstructs the Python objects with all learned weights and parameters
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("✅ Model and vectorizer loaded successfully.")
    return model, vectorizer


def predict(text, model, vectorizer):
    """
    Predict whether a single text is informative or not.
    
    Inference Pipeline Steps:
    -------------------------
    1. Text Preprocessing: Cleans the raw input string using the exact same `clean_text()`
       function used during training (stripping URLs, @user handles, special chars, lowercasing).
       *WHY IDENTICAL PREPROCESSING?*
       If the training set lowercased all words and stripped punctuation, the model's vocabulary
       only contains words like 'flood' and 'rescue'. If inference receives 'FLOOD!!!', without
       preprocessing it would be treated as an Out-of-Vocabulary (OOV) token, resulting in all-zero
       feature weights and ruined prediction accuracy!
       
    2. TF-IDF Transformation:
       `vectorizer.transform([cleaned])` transforms the text into the pre-learned N-dimensional space.
       Notice we call `.transform()`, NEVER `.fit_transform()` at inference time!
       
    3. Classification & Confidence Scoring:
       - `model.predict(features)` yields the binary class: 0 (Not Informative) or 1 (Informative).
       - `model.predict_proba(features)` yields class probabilities: [P(class 0), P(class 1)].
       - Confidence is the probability of the predicted class: e.g., 0.94 means 94% certain.
       *WHAT CONFIDENCE MEANS IN DISASTER RESPONSE:*
       Confidence indicates decision certainty. High confidence (>90%) enables automated routing
       to first responders. Borderline confidence (50-65%) flags the tweet for human triage.
    
    Args:
        text (str): Raw input text to classify.
        model: Trained sklearn model (must support predict_proba).
        vectorizer: Fitted TfidfVectorizer.
    
    Returns:
        tuple: (label_str, confidence_float)
            - label_str: "Informative" or "Not Informative"
            - confidence_float: Probability score between 0 and 1
    """
    # Step 1: Apply same preprocessing as training
    cleaned = clean_text(text)
    
    # Step 2: Handle edge cases (empty string or only whitespace/punctuation)
    if not cleaned or not cleaned.strip():
        return "Not Informative", 0.5
    
    # Step 3: Transform using the fitted TF-IDF vectorizer (projects text to feature vector)
    features = vectorizer.transform([cleaned])
    
    # Step 4: Get discrete class prediction (0 or 1)
    prediction = model.predict(features)[0]
    
    # Step 5: Get calibrated confidence score (class posterior probability)
    try:
        # predict_proba returns a 2D array: [[prob_class_0, prob_class_1]]
        probabilities = model.predict_proba(features)[0]
        # Confidence is the probability assigned to whichever class won the prediction
        confidence = float(probabilities[prediction])
    except AttributeError:
        # Fallback heuristic if a model without predict_proba is supplied
        # (Our trained LinearSVC is wrapped in CalibratedClassifierCV, so predict_proba is supported)
        confidence = 0.95 if prediction == 1 else 0.90
    
    # Step 6: Map numeric prediction back to human-readable label
    label_map = {1: "Informative", 0: "Not Informative"}
    label = label_map.get(prediction, "Unknown")
    
    return label, confidence


def predict_batch(texts, model, vectorizer):
    """
    Predict multiple texts at once.
    
    Iterates through a collection of incoming tweets or messages,
    running each through the inference pipeline and aggregating results.
    
    Connection to Streamlit App:
    ----------------------------
    In `app/app.py`, when a user uploads a batch CSV or clicks 'Process Live Stream',
    the Streamlit app passes the list of texts to `predict_batch()`.
    The returned list of dictionaries is converted into a Pandas DataFrame and displayed
    as an interactive data table with download buttons, metric cards, and pie charts.
    
    Args:
        texts (list): List of raw text strings.
        model: Trained sklearn model.
        vectorizer: Fitted TfidfVectorizer.
    
    Returns:
        list: List of dicts with 'text', 'prediction', 'confidence' keys.
    """
    results = []
    for text in texts:
        label, confidence = predict(text, model, vectorizer)
        results.append({
            'text': text,
            'prediction': label,
            'confidence': confidence
        })
    return results


# ============================================================
# Demo — Run this to test predictions with example texts
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚨 Disaster Intelligence Engine — Prediction Demo")
    print("=" * 60)
    
    # Attempt to load model and vectorizer
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    # Example disaster-related texts showcasing informative vs casual chatter
    example_texts = [
        "People are trapped after the flood near the bridge and need immediate rescue.",
        "This storm is crazy omg 😭",
        "Red Cross setting up emergency shelters at Lincoln High School for flood evacuees.",
        "Praying for everyone affected by the earthquake 🙏",
        "URGENT: 3 buildings collapsed in downtown after 7.2 magnitude earthquake. "
        "Multiple casualties reported. Emergency services en route.",
        "Just saw the news about the hurricane. Stay safe everyone!",
        "Water supply contaminated after flood. Boil water advisory issued for zones 3-7.",
    ]
    
    print("\n📋 Running predictions on example texts...\n")
    results = predict_batch(example_texts, model, vectorizer)
    
    for i, res in enumerate(results, 1):
        icon = "🟢" if res['prediction'] == "Informative" else "🔴"
        print(f"{'─'*60}")
        print(f"Example {i}:")
        print(f"  Text: {res['text'][:80]}{'...' if len(res['text']) > 80 else ''}")
        print(f"  {icon} Prediction: {res['prediction']}")
        print(f"  📊 Confidence: {res['confidence']:.1%}")
    
    print(f"\n{'─'*60}")
    print("✅ Prediction demo complete!")
