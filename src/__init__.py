"""
Disaster Intelligence Engine - Source Package
==============================================
This package contains all the core ML pipeline modules for the
Disaster Intelligence Engine project.

Modules:
    data_loader.py   - Loads and combines CrisisMMD dataset TSV files
    preprocessing.py - Cleans text data (URLs, mentions, hashtags, etc.)
    features.py      - Extracts TF-IDF features from cleaned text
    train.py         - Main training pipeline (runs all steps)
    evaluate.py      - Evaluates models and generates visualizations
    predict.py       - Makes predictions using the trained model

The typical workflow is:
    1. data_loader loads raw TSV files from data/raw/
    2. preprocessing cleans the text and encodes labels
    3. features extracts TF-IDF numerical features
    4. train.py orchestrates the full pipeline and trains 4 ML models
    5. evaluate.py computes metrics and saves plots
    6. predict.py loads the saved model for inference in the Streamlit app
"""
