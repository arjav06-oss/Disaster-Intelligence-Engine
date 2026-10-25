"""
train.py - Main Training Pipeline
===================================
Runs the complete ML pipeline: load → preprocess → split → featurize → train → evaluate → save.

Educational Overview for AI/ML Students:
----------------------------------------
This file orchestrates the entire End-to-End Machine Learning lifecycle for NLP text classification:
1. Data Ingestion: Loads raw disaster tweets from CSV files or HuggingFace dataset.
2. Data Preprocessing: Cleans noisy social media text (URLs, mentions, special chars, lowercasing).
3. Train/Test Split: Splits data into 80% training and 20% testing sets using Stratified Sampling.
4. Feature Extraction: Converts raw text into numerical TF-IDF feature vectors.
5. Model Training: Trains 4 diverse classification algorithms (Logistic Regression, Naive Bayes, Linear SVC, Random Forest).
6. Model Evaluation: Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrices on the unseen test set.
7. Model Selection: Selects the top performer based on F1-Score (balances precision and recall).
8. Artifact Persistence: Serializes the winning model (.pkl), vectorizer (.pkl), and performance metrics (.json).
9. Visualization: Generates comparative bar charts and heatmaps for reporting and dashboards.

Usage:
    python src/train.py
    
    (Run from the project root directory)
"""

import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path

# Add project root to path so we can import from src
# This ensures Python can locate our sibling modules regardless of where the script is executed from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Machine Learning algorithms and utilities from scikit-learn:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Import our custom modules
from src.data_loader import load_dataset, save_processed_data
from src.preprocessing import preprocess_dataframe, save_preprocessed_data
from src.features import create_tfidf_features, save_vectorizer
from src.evaluate import evaluate_model, plot_model_comparison


def main():
    """
    Main training pipeline function.
    Executes all steps from data loading to model saving.
    
    Data Flow:
    -----------
    Raw Data (DataFrame) 
        ↓  preprocess_dataframe()
    Clean Text & Encoded Labels (DataFrame)
        ↓  train_test_split(stratify=y)
    Train/Test Text (Arrays: X_train_text, X_test_text)
        ↓  create_tfidf_features()
    Sparse TF-IDF Matrices (X_train_tfidf, X_test_tfidf)
        ↓  model.fit() & evaluate_model()
    Metrics & Best Model Object
        ↓  joblib.dump() & json.dump()
    Saved Artifacts (.pkl, .json, .png)
    """
    print("=" * 60)
    print("🚨 DISASTER INTELLIGENCE ENGINE — Training Pipeline")
    print("=" * 60)
    
    # ============================================================
    # STEP 1: Load Dataset
    # ============================================================
    # Ingests tweets from local CSVs (CrisisMMD / Disaster Tweets) or fallback.
    # The loader unifies column names into 'text' and 'label' ('Informative' vs 'Not Informative').
    print("\n" + "=" * 60)
    print("📂 STEP 1: Loading Dataset")
    print("=" * 60)
    
    df = load_dataset()
    
    if df is None or len(df) == 0:
        print("\n❌ No data loaded. Cannot proceed with training.")
        print("Please follow the dataset setup instructions above.")
        return
    
    # Save raw processed data for inspection and reproducibility
    save_processed_data(df)
    
    # ============================================================
    # STEP 2: Preprocess Data
    # ============================================================
    # Social media text contains noise: URLs, user mentions (@user), emojis, HTML tags,
    # and inconsistent casing. Preprocessing standardizes the text so our vocabulary
    # captures meaningful signal rather than noise.
    print("\n" + "=" * 60)
    print("🧹 STEP 2: Preprocessing Data")
    print("=" * 60)
    
    df_processed = preprocess_dataframe(df)
    save_preprocessed_data(df_processed)
    
    # Extract text and labels:
    # X_text: 1D array of cleaned strings (e.g., "urgent 3 buildings collapsed downtown")
    # y: 1D array of binary integer labels (1 for Informative disaster reports, 0 for Not Informative)
    # Convert to plain numpy arrays to avoid compatibility issues with
    # Arrow-backed pandas (newer pandas versions use PyArrow strings by default,
    # which can cause TypeError in sklearn's train_test_split indexing)
    X_text = df_processed['clean_text'].to_numpy(dtype=str, na_value='')
    y = df_processed['label_encoded'].to_numpy(dtype=int)
    
    print(f"\n📊 Final dataset size: {len(X_text)} samples")
    print(f"   Informative (1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
    print(f"   Not Informative (0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
    
    # Save dataset stats for the Streamlit app dashboard
    # This dictionary will be written to model_results.json so the frontend can display dataset metrics.
    dataset_stats = {
        'total_samples': int(len(X_text)),
        'informative_count': int(np.sum(y == 1)),
        'not_informative_count': int(np.sum(y == 0)),
    }
    
    # If event column exists, save event distribution (e.g., earthquake, flood, hurricane)
    if 'event' in df_processed.columns:
        event_counts = df_processed['event'].value_counts().to_dict()
        dataset_stats['event_distribution'] = {str(k): int(v) for k, v in event_counts.items()}
    
    # ============================================================
    # STEP 3: Train/Test Split
    # ============================================================
    # WHY STRATIFIED SPLIT?
    # In real-world disaster data, classes may be imbalanced (e.g. 70% non-informative, 30% informative).
    # A regular random split might accidentally assign too many informative samples to the test set,
    # or starve the training set. Stratified splitting guarantees that both training and testing sets
    # preserve the EXACT same proportion of informative (1) vs non-informative (0) samples as the full dataset.
    # We use an 80/20 split: 80% for learning parameters, 20% for unbiased final evaluation.
    print("\n" + "=" * 60)
    print("✂️ STEP 3: Stratified Train/Test Split (80/20)")
    print("=" * 60)
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,  # Fixed seed ensures identical reproducible splits across runs
        stratify=y        # Stratification maintains identical class distribution in train and test sets
    )
    
    print(f"   Training set: {len(X_train_text)} samples")
    print(f"   Test set:     {len(X_test_text)} samples")
    
    dataset_stats['train_size'] = int(len(X_train_text))
    dataset_stats['test_size'] = int(len(X_test_text))
    
    # ============================================================
    # STEP 4: Feature Extraction (TF-IDF)
    # ============================================================
    # Machine learning models cannot process raw strings; they require numerical feature vectors.
    # TF-IDF (Term Frequency - Inverse Document Frequency) balances:
    # 1. Term Frequency (TF): How frequently a word occurs in the specific document.
    # 2. Inverse Document Frequency (IDF): Penalizes common words appearing across all documents (e.g., 'the', 'is')
    #    and amplifies rare, domain-specific words (e.g., 'casualties', 'evacuation', 'shelter').
    # NOTE: The vectorizer is FIT only on X_train_text, and then transforms both train and test.
    # This prevents Data Leakage (information from test set leaking into training features).
    print("\n" + "=" * 60)
    print("📝 STEP 4: TF-IDF Feature Extraction")
    print("=" * 60)
    
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(
        X_train_text, X_test_text
    )
    
    print(f"   Feature matrix shape (train): {X_train_tfidf.shape}")
    print(f"   Feature matrix shape (test):  {X_test_tfidf.shape}")
    
    # ============================================================
    # STEP 5: Train Models
    # ============================================================
    # We compare 4 distinct machine learning algorithms to discover which best captures
    # the textual patterns of disaster communications:
    #
    # 1. Logistic Regression:
    #    - Concept: Linear model that applies the sigmoid function 1 / (1 + e^(-z)) to compute class probabilities.
    #    - Why chosen: Excellent linear baseline for text classification. Fast to train, robust against overfitting
    #      with L2 regularization (C=1.0), and highly interpretable with sparse high-dimensional TF-IDF vectors.
    #
    # 2. Naive Bayes (MultinomialNB):
    #    - Concept: Probabilistic classifier based on Bayes' Theorem P(C|X) = P(X|C)*P(C) / P(X) with the
    #      "naive" assumption of conditional independence between words given the class label.
    #    - Why chosen: The classic standard benchmark for NLP and spam/disaster filtering. Extremely fast, works
    #      exceptionally well with word frequency/count distributions, and requires very few hyperparameter adjustments (alpha=1.0 Laplace smoothing).
    #
    # 3. Linear Support Vector Classifier (LinearSVC):
    #    - Concept: Finds the optimal maximum-margin hyperplane that separates the two classes in high-dimensional feature space.
    #    - Why chosen: Renowned for outstanding accuracy in high-dimensional sparse text classification.
    #    - CalibratedClassifierCV wrap: LinearSVC maximizes the margin using hinge loss, which outputs raw distance scores
    #      (decision_function), NOT probabilities. It lacks a `predict_proba()` method. By wrapping it with
    #      CalibratedClassifierCV(cv=3), scikit-learn uses 3-fold cross-validation and Platt scaling (sigmoid calibration)
    #      to convert raw margin distances into well-calibrated class probabilities [0.0 to 1.0].
    #      This enables confidence scoring in both our inference pipeline and the Streamlit UI!
    #
    # 4. Random Forest:
    #    - Concept: Ensemble method that builds an array (forest) of 200 decision trees using bagging (bootstrap aggregation)
    #      and random feature subsets at each split.
    #    - Why chosen: Evaluates whether non-linear combinations of words improve classification, provides resilience against
    #      outliers, and avoids overfitting inherent to single decision trees.
    print("\n" + "=" * 60)
    print("🤖 STEP 5: Training 4 ML Models")
    print("=" * 60)
    
    models = {
        'Logistic_Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'Naive_Bayes': MultinomialNB(
            alpha=1.0
        ),
        'Linear_SVC': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42),
            cv=3  # 3-fold cross-validation for probability calibration
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }
    
    # ============================================================
    # STEP 6: Evaluate Each Model
    # ============================================================
    # Each model is fitted on the training set (X_train_tfidf, y_train),
    # then evaluated strictly on unseen test data (X_test_tfidf, y_test).
    # evaluate_model() computes Accuracy, Precision, Recall, F1, and saves confusion matrices.
    print("\n" + "=" * 60)
    print("📊 STEP 6: Evaluating All Models")
    print("=" * 60)
    
    all_results = {}
    best_model_name = None
    best_f1 = -1
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"🔄 Training: {name.replace('_', ' ')}...")
        
        # Train
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test_tfidf, y_test, name)
        all_results[name] = metrics
        
        # Track best model based on F1 Score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_model_obj = model
    
    # ============================================================
    # STEP 7: Model Comparison & Selection
    # ============================================================
    # Print formatted table to compare models across all 4 metrics.
    #
    # WHY F1 SCORE IS THE SELECTION METRIC:
    # In disaster management, standard accuracy can be dangerously misleading when classes are imbalanced.
    # 1. False Negatives (FN): Missing an urgent plea for help means victims don't get rescued (life safety risk).
    #    Recall measures this: Recall = TP / (TP + FN).
    # 2. False Positives (FP): Marking random chatter as a disaster wastes scarce emergency response resources.
    #    Precision measures this: Precision = TP / (TP + FP).
    # F1 Score is the Harmonic Mean of Precision and Recall: 2 * (P * R) / (P + R).
    # Unlike arithmetic mean, the harmonic mean heavily penalizes models that sacrifice one metric for the other.
    # Therefore, F1 score selects the model that offers the best operational balance for disaster response.
    print("\n" + "=" * 60)
    print("🏆 STEP 7: Model Comparison & Selection")
    print("=" * 60)
    
    # Print comparison table
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("─" * 65)
    for name, metrics in all_results.items():
        marker = " ⭐" if name == best_model_name else ""
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}{marker}")
    
    print(f"\n🏆 Best Model: {best_model_name} (F1 = {best_f1:.4f})")
    print(f"\n💡 WHY F1 SCORE?")
    print(f"   In disaster response, missing an informative report (low recall)")
    print(f"   is dangerous. But too many false alarms (low precision) waste")
    print(f"   resources. F1 balances both precision and recall.")
    
    # ============================================================
    # STEP 8: Save Everything
    # ============================================================
    # We persist all artifacts so they can be consumed without retraining:
    # 1. 'disaster_model.pkl': The trained scikit-learn model object with highest F1 score.
    # 2. 'tfidf_vectorizer.pkl': The fitted TfidfVectorizer object (vocabulary and IDF weights).
    # 3. 'model_results.json': Clean JSON file storing evaluation metrics and metadata.
    #
    # JSON STRUCTURE EXPLANATION:
    # {
    #   "best_model": "Linear_SVC",                  <-- Name of chosen model
    #   "dataset_stats": {                           <-- Counts and split sizes
    #       "total_samples": 10000,
    #       "informative_count": 4000,
    #       "not_informative_count": 6000,
    #       "train_size": 8000,
    #       "test_size": 2000,
    #       "event_distribution": { ... }
    #   },
    #   "models": {                                  <-- Nested metrics for each tested model
    #       "Logistic_Regression": {
    #           "accuracy": 0.8125,
    #           "precision": 0.7910,
    #           "recall": 0.7520,
    #           "f1": 0.7710,
    #           "confusion_matrix": [[TN, FP], [FN, TP]]
    #       }, ...
    #   }
    # }
    print("\n" + "=" * 60)
    print("💾 STEP 8: Saving Model, Vectorizer & Results")
    print("=" * 60)
    
    # Create directories
    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save best model
    model_path = os.path.join(models_dir, "disaster_model.pkl")
    joblib.dump(best_model_obj, model_path)
    print(f"   ✅ Saved best model ({best_model_name}) → {model_path}")
    
    # Save TF-IDF vectorizer
    save_vectorizer(vectorizer, models_dir)
    
    # Save model comparison results as JSON
    # (Remove non-serializable items like classification_report text)
    json_results = {
        'best_model': best_model_name,
        'dataset_stats': dataset_stats,
        'models': {}
    }
    
    for name, metrics in all_results.items():
        json_results['models'][name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'confusion_matrix': metrics.get('confusion_matrix', [])
        }
    
    results_path = os.path.join(models_dir, "model_results.json")
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"   ✅ Saved results → {results_path}")
    
    # ============================================================
    # STEP 9: Generate Comparison Charts
    # ============================================================
    # Creates a grouped bar chart comparing all 4 models across all 4 metrics
    # and saves it as 'reports/figures/model_comparison.png'.
    print("\n" + "=" * 60)
    print("📈 STEP 9: Generating Comparison Charts")
    print("=" * 60)
    
    chart_path = os.path.join(reports_dir, "model_comparison.png")
    plot_model_comparison(all_results, chart_path)
    
    # ============================================================
    # DONE!
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 Saved files:")
    print(f"   models/disaster_model.pkl")
    print(f"   models/tfidf_vectorizer.pkl")
    print(f"   models/model_results.json")
    print(f"   reports/figures/model_comparison.png")
    print(f"   reports/figures/confusion_matrix_*.png")
    print(f"\n🚀 To launch the web app, run:")
    print(f"   streamlit run app/app.py")


if __name__ == "__main__":
    main()
