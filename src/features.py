"""
features.py - Feature Engineering Module
==========================================
Extracts TF-IDF features from cleaned text data.
TF-IDF (Term Frequency - Inverse Document Frequency) converts text
into numerical vectors that ML models can process.

=============================================================================
🎓 STUDENT GUIDE - TEXT VECTORIZATION & TF-IDF EXPLAINED:
-----------------------------------------------------------------------------
This module is Stage 3 of the ML pipeline:
  [preprocessed_data.csv] ---> [features.py] ---> [X_train_tfidf, X_test_tfidf]
                                                         |
                                                         v
                                                    [train.py]

1. Why do we need Feature Engineering for Text?
   Machine Learning algorithms (like Logistic Regression, SVM, and Naive Bayes)
   are mathematical models that perform linear algebra operations. They cannot
   read words or sentences directly. We must convert text strings into numerical
   feature vectors — a process known as Vectorization.

2. What is TF-IDF?
   Simple word counting (Bag of Words) has a fatal flaw: common filler words
   like "the", "is", and "at" appear constantly, dominating word counts even
   though they contain zero disaster information.
   TF-IDF solves this by balancing two terms:
     - TF (Term Frequency): How often a word appears in a specific tweet.
       If "evacuation" appears 3 times in a tweet, it is important to that tweet.
     - IDF (Inverse Document Frequency): How rare a word is across ALL tweets:
         IDF(word) = log(Total number of tweets / Number of tweets with word)
       If a word appears in every tweet (like "disaster"), IDF ≈ 0.
       If a word is rare and specific (like "tsunami" or "trapped"), IDF is high.
     - TF-IDF = TF × IDF: Assigns high weights to words that are frequent in a
       given tweet, but distinctive across the entire corpus.

3. Why fit ONLY on Training Data? (Preventing Data Leakage)
   In real life, your model will be deployed to predict on unseen incoming tweets.
   The test dataset simulates this unseen future.
   If we fit the vectorizer on the test data (or on the combined dataset):
     - The vocabulary and IDF weights learn information from the test set.
     - This is DATA LEAKAGE: the model cheats by peeking at test statistics.
     - Consequence: Overly optimistic evaluation scores that fail in production.
   Rule: Always fit on Train ONLY, then transform both Train and Test.

4. What is a Sparse Matrix?
   With a vocabulary of 10,000 words, each tweet becomes a vector of length 10,000.
   However, a typical tweet has only 15 to 30 words. Over 99% of entries are 0.
   A standard (dense) matrix would waste gigabytes storing zeros.
   A SciPy Compressed Sparse Row (CSR) matrix stores ONLY non-zero values and
   their coordinates, drastically saving RAM and accelerating training.
=============================================================================

Usage:
    Called by train.py during the training pipeline.
    
    Or import:
        from src.features import create_tfidf_features
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Project root directory for consistent file paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_tfidf_features(train_texts, test_texts):
    """
    Create TF-IDF feature matrices from text data.
    
    IMPORTANT: The vectorizer is FIT ONLY on training data to prevent 
    data leakage. Test data is only transformed.
    
    TF-IDF Settings:
        - max_features=10000: Keep top 10,000 features
        - ngram_range=(1,2): Use both unigrams and bigrams
        - min_df=3: Ignore terms appearing in fewer than 3 documents
        - max_df=0.95: Ignore terms appearing in more than 95% of documents
        - sublinear_tf=True: Apply sublinear TF scaling (1 + log(tf))
    
    Args:
        train_texts (array-like): Training text data.
        test_texts (array-like): Test text data.
    
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
            - X_train_tfidf: Sparse matrix of training features
            - X_test_tfidf: Sparse matrix of test features
            - vectorizer: Fitted TfidfVectorizer object
    """
    print("📝 Extracting TF-IDF features...")
    
    # -----------------------------------------------------------------------
    # Initializing TfidfVectorizer with carefully chosen hyperparameters:
    # -----------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        # max_features=10000:
        # Caps vocabulary size to the 10,000 most informative terms.
        # Why? Prevents the "curse of dimensionality" and limits memory usage
        # while retaining sufficient vocabulary for accurate classification.
        max_features=10000,
        
        # ngram_range=(1, 2):
        # Uses unigrams (single words like 'flood') AND bigrams (word pairs like 'flash flood').
        # Why? Context matters! In disaster situations:
        #   - 'power' + 'outage' -> 'power outage' (specific infrastructure failure)
        #   - 'not' + 'safe' -> 'not safe' (negation capture; unigrams alone lose this!)
        ngram_range=(1, 2),
        
        # min_df=3:
        # Minimum document frequency: A word/bigram must appear in at least 3 tweets.
        # Why? Removes one-off typos, rare usernames, or unique noise that cannot
        # help the model generalize to new, unseen tweets.
        min_df=3,
        
        # max_df=0.95:
        # Maximum document frequency: Ignores terms appearing in >95% of tweets.
        # Why? Words that appear in virtually every tweet cannot help distinguish
        # informative from non-informative content (acts like an automatic stop-word filter).
        max_df=0.95,
        
        # sublinear_tf=True:
        # Replaces raw term frequency (tf) with sublinear scaling: 1 + log(tf).
        # Why? A word appearing 20 times in a tweet is not 20 times more important
        # than a word appearing once; log scaling dampens the effect of repetition.
        sublinear_tf=True,
        
        # strip_accents='unicode':
        # Normalizes accented characters (e.g., 'résumé' -> 'resume').
        # Useful for multilingual social media text.
        strip_accents='unicode',
        
        # token_pattern=r'(?u)\b\w\w+\b':
        # Regular expression defining what is considered a word token.
        # Selects words with 2 or more alphanumeric characters, dropping single letters.
        token_pattern=r'(?u)\b\w\w+\b'
    )
    
    # -----------------------------------------------------------------------
    # Step 1: FIT on training data ONLY (prevents data leakage)
    # fit_transform() does two things:
    #   1. fit(): Learns the vocabulary dictionary and computes IDF weights
    #      from train_texts exclusively.
    #   2. transform(): Encodes train_texts into a numerical sparse matrix.
    # -----------------------------------------------------------------------
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    
    # -----------------------------------------------------------------------
    # Step 2: TRANSFORM test data using the fitted vectorizer
    # transform() uses the ALREADY LEARNED vocabulary and IDF weights from
    # training. It does NOT learn new words from test_texts.
    # Words in test_texts that were never seen during training are safely ignored.
    # -----------------------------------------------------------------------
    X_test_tfidf = vectorizer.transform(test_texts)
    
    # Display dimensions and vocabulary statistics:
    # The output matrices are scipy.sparse.csr_matrix (Compressed Sparse Row)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"   Vocabulary size: {vocab_size:,} terms")
    print(f"   Train matrix: {X_train_tfidf.shape}")  # (n_train_samples, n_features)
    print(f"   Test matrix:  {X_test_tfidf.shape}")   # (n_test_samples, n_features)
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def compute_simple_features(texts):
    """
    Compute basic text statistics as additional features.
    
    These simple features can complement TF-IDF by capturing
    document-level properties.
    
    Why text length and word count matter in disaster intelligence:
    - Informative tweets (e.g., emergency alerts, shelter locations, casualty reports)
      tend to contain structured sentences, addresses, contact numbers, and higher word counts.
    - Non-informative tweets (e.g., emotional reactions like "OMG!", "stay safe everyone")
      are often much shorter with lower word counts.
    
    Args:
        texts (array-like): Text data.
    
    Returns:
        pd.DataFrame: DataFrame with text_length and word_count columns.
    """
    text_series = pd.Series(texts)
    
    features = pd.DataFrame({
        # Character count: Total number of characters in the cleaned tweet
        'text_length': text_series.str.len().fillna(0).astype(int),
        # Word count: Total number of whitespace-separated tokens
        'word_count': text_series.str.split().str.len().fillna(0).astype(int),
    })
    
    return features


def save_vectorizer(vectorizer, model_dir=None):
    """
    Save the fitted TF-IDF vectorizer to disk.
    
    Why serialize the vectorizer?
    In production (e.g., our Streamlit web dashboard or live API):
    Incoming new disaster tweets must be transformed using the EXACT same
    vocabulary indices and IDF weights that the model was trained on!
    Without the saved vectorizer, newly received tweets cannot be processed.
    We use joblib for fast and efficient serialization of NumPy/SciPy objects.
    
    Args:
        vectorizer: Fitted TfidfVectorizer object.
        model_dir (str, optional): Directory to save. Defaults to models/.
    """
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, "models")
    
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    # Save the vectorizer object to disk
    joblib.dump(vectorizer, output_path)
    print(f"   ✅ Saved TF-IDF vectorizer → {output_path}")


def load_vectorizer(model_dir=None):
    """
    Load a saved TF-IDF vectorizer from disk.
    
    Used during inference / testing to vectorize new tweets without retraining.
    
    Args:
        model_dir (str, optional): Directory to load from. Defaults to models/.
    
    Returns:
        TfidfVectorizer or None: Loaded vectorizer, or None if not found.
    """
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, "models")
    
    input_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    # Check if the serialized file exists before attempting to load
    if os.path.exists(input_path):
        return joblib.load(input_path)
    
    print(f"⚠️ Vectorizer not found: {input_path}")
    return None
