"""
preprocessing.py - Text Preprocessing Module
===============================================
Cleans disaster-related text data while preserving important information.
Handles URLs, mentions, hashtags, punctuation, and whitespace.

=============================================================================
🎓 STUDENT GUIDE - WHY TEXT PREPROCESSING IS CRITICAL FOR ML:
-----------------------------------------------------------------------------
This module is Stage 2 of the ML pipeline:
  [processed_data.csv] ---> [preprocessing.py] ---> [preprocessed_data.csv]
                                                          |
                                                          v
                                                    [features.py]

1. Why do we need preprocessing?
   In Machine Learning, the rule is "Garbage In, Garbage Out". Raw social media
   posts (tweets) are filled with noise: web links, user handles, emojis, irregular
   capitalization, and duplicated retweets. If fed directly to an ML model:
     - Vocabulary explodes with meaningless tokens (e.g., bit.ly links).
     - The model overfits to usernames or specific URL patterns.
     - Identical words in different cases ("Flood" vs "flood") are treated as
       completely different features.

2. Why keep disaster-related words?
   Unlike generic text cleaning where hashtags might be stripped entirely,
   disaster tweets rely heavily on hashtags (e.g., #flood, #earthquake, #rescue)
   to convey urgent situational awareness. We strip the '#' symbol but
   PRESERVE the underlying word so the model learns these critical signals.

3. Why do label encoding?
   Computers and ML algorithms perform mathematical matrix multiplications
   and compute loss functions. They cannot calculate gradients on strings like
   "informative". Converting 'informative' -> 1 and 'not_informative' -> 0
   provides the standard binary targets (y ∈ {0, 1}) needed for classification.

4. Why remove duplicates?
   On Twitter, retweets cause the exact same message to appear multiple times.
   Removing duplicates prevents data leakage between train/test splits and
   stops the model from becoming biased toward frequently repeated tweets.
=============================================================================

Usage:
    python src/preprocessing.py
    
    Or import functions:
        from src.preprocessing import clean_text, preprocess_dataframe
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path

# Add project root to path so modules can import from the root directory cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def clean_text(text):
    """
    Clean a single text string for ML processing.
    
    Removes noise (URLs, mentions) while PRESERVING disaster-related
    vocabulary that is important for classification.
    
    Args:
        text (str): Raw input text.
    
    Returns:
        str: Cleaned text.
    """
    # Guard clause: Handle missing values (NaN) or non-string inputs safely
    # If text is None or float (like np.nan), return an empty string to avoid AttributeError
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text)
    
    # -----------------------------------------------------------------------
    # Step 1: Lowercase
    # WHY: Words like "Flood", "flood", and "FLOOD" carry identical meaning.
    # Without lowercasing, the vocabulary would count them as 3 separate tokens,
    # unnecessarily inflating vocabulary size and diluting frequency statistics.
    # -----------------------------------------------------------------------
    text = text.lower()
    
    # -----------------------------------------------------------------------
    # Step 2: Remove URLs (http/https/www links)
    # WHY: URLs (e.g., 'https://t.co/xyz123') are unique identifiers generated
    # by link shorteners. They do not carry general disaster semantics and
    # would only add random, uninformative noise to our feature matrix.
    # -----------------------------------------------------------------------
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # -----------------------------------------------------------------------
    # Step 3: Remove @mentions
    # WHY: Mentions like '@RedCross' or '@user123' identify specific accounts.
    # Removing them prevents the model from overfitting to individual usernames
    # that won't appear in future, unseen disaster events.
    # -----------------------------------------------------------------------
    text = re.sub(r'@\w+', '', text)
    
    # -----------------------------------------------------------------------
    # Step 4: Clean hashtags — keep the word, remove just the # symbol
    # WHY: In crisis communication, users convey urgent keywords via hashtags
    # (e.g., #earthquake, #trapped, #shelter, #hurricane).
    # If we deleted the whole hashtag, we would destroy crucial disaster signal!
    # By using regex capturing group r'\1', '#earthquake' becomes 'earthquake'.
    # -----------------------------------------------------------------------
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # -----------------------------------------------------------------------
    # Step 5: Remove special characters but keep basic punctuation
    # WHY: Emojis, strange unicode symbols, and formatting characters are noise.
    # We keep letters, numbers (\w), whitespace (\s), and basic punctuation
    # (.,!?) because exclamation marks and question marks often convey urgency
    # or emergency appeals in disaster tweets (e.g., "Need rescue now!").
    # -----------------------------------------------------------------------
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # -----------------------------------------------------------------------
    # Step 6: Remove extra whitespace
    # WHY: After removing URLs, mentions, and special characters, multiple
    # consecutive spaces and newlines are left behind. Collapsing them to a
    # single space ensures uniform word separation.
    # -----------------------------------------------------------------------
    text = re.sub(r'\s+', ' ', text)
    
    # -----------------------------------------------------------------------
    # Step 7: Strip leading/trailing whitespace
    # WHY: Cleanly trims whitespace from the beginning and end of the string.
    # -----------------------------------------------------------------------
    text = text.strip()
    
    return text


def preprocess_dataframe(df):
    """
    Apply full preprocessing pipeline to a DataFrame.
    
    Steps:
    1. Remove rows with missing text or label
    2. Apply text cleaning
    3. Remove empty texts after cleaning
    4. Remove duplicate texts
    5. Encode labels (informative=1, not_informative=0)
    
    Args:
        df (pd.DataFrame): Raw DataFrame with 'text' and 'label' columns.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with added 'clean_text' and 'label_encoded' columns.
    """
    print("🧹 Starting preprocessing pipeline...")
    initial_size = len(df)
    
    # -----------------------------------------------------------------------
    # Step 1: Remove missing values (NaNs)
    # WHY: Supervised ML models require both input features (X) and labels (y).
    # A row with missing text has no features to learn from; a row with missing
    # label has no target to verify predictions against.
    # -----------------------------------------------------------------------
    df = df.dropna(subset=['text', 'label']).copy()
    removed_missing = initial_size - len(df)
    if removed_missing > 0:
        print(f"   Removed {removed_missing} rows with missing text/label")
    
    # -----------------------------------------------------------------------
    # Step 2: Apply text cleaning
    # HOW: Pandas .apply() executes clean_text on every row of the 'text' series,
    # producing the standardized 'clean_text' column.
    # -----------------------------------------------------------------------
    print("   Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # -----------------------------------------------------------------------
    # Step 3: Remove empty texts after cleaning
    # WHY: Some tweets consist ONLY of URLs, mentions, or emojis
    # (e.g., "Check this out: https://t.co/xyz123 @friend").
    # After our cleaning steps, such tweets become empty strings ("").
    # Empty strings produce zero-vectors in TF-IDF and cannot help model training.
    # -----------------------------------------------------------------------
    before = len(df)
    df = df[df['clean_text'].str.len() > 0]
    removed_empty = before - len(df)
    if removed_empty > 0:
        print(f"   Removed {removed_empty} empty texts after cleaning")
    
    # -----------------------------------------------------------------------
    # Step 4: Remove duplicates based on cleaned text
    # WHY: In Twitter data, retweets create thousands of identical messages.
    # If duplicates exist:
    # 1. Data Leakage: The same tweet might fall into both train and test splits,
    #    giving falsely optimistic test performance.
    # 2. Bias: The model overweights repeated tweets over diverse disaster reports.
    # -----------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=['clean_text'])
    removed_dupes = before - len(df)
    print(f"   Removed {removed_dupes} duplicate texts")
    
    # -----------------------------------------------------------------------
    # Step 5: Encode labels (Categorical -> Binary Numeric)
    # WHAT: Label encoding maps:
    #   'informative'     --> 1 (Positive class: actionable disaster info)
    #   'not_informative' --> 0 (Negative class: chatter, opinions, personal talk)
    # WHY: Classification algorithms (Logistic Regression, Random Forest, SVM)
    # calculate loss functions and decision boundaries using numbers, not strings.
    # -----------------------------------------------------------------------
    print("   Encoding labels...")
    df['label_encoded'] = df['label'].map({
        'informative': 1,
        'not_informative': 0
    })
    
    # Safety check: Drop any unexpected values that failed mapping (became NaN)
    # and convert the target column to integer type (0 or 1).
    df = df.dropna(subset=['label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    
    print(f"✅ Preprocessing complete: {initial_size} → {len(df)} samples")
    
    return df


def save_preprocessed_data(df, output_path=None):
    """
    Save the preprocessed DataFrame to CSV.
    
    Saves clean_text alongside original text and the binary encoded label.
    This file is ready for feature extraction (TF-IDF vectorization) and model training.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        output_path (str, optional): File path. Defaults to data/processed/preprocessed_data.csv.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_data.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select columns to save: keep text, clean_text, label, and label_encoded
    cols = ['text', 'clean_text', 'label', 'label_encoded']
    if 'event' in df.columns:
        cols.append('event')
    
    df[cols].to_csv(output_path, index=False)
    print(f"💾 Saved preprocessed data: {output_path}")


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧹 Disaster Intelligence Engine — Preprocessing")
    print("=" * 60)
    
    # Input file produced by Stage 1 (src/data_loader.py)
    input_path = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv")
    
    if not os.path.exists(input_path):
        print(f"\n❌ Input file not found: {input_path}")
        print("Please run 'python src/data_loader.py' first.")
        sys.exit(1)
    
    # Load consolidated dataset
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    
    # Run preprocessing steps
    df_processed = preprocess_dataframe(df)
    
    # Persist clean dataset for features.py / train.py
    save_preprocessed_data(df_processed)
    
    # Inspect final class distribution (0 vs 1 balance)
    print(f"\n📊 Label distribution after preprocessing:")
    print(df_processed['label_encoded'].value_counts().to_string())
    print("\n✅ Done!")
