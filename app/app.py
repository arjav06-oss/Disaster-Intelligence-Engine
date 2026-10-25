"""
app.py - Disaster Intelligence Engine Streamlit Application
=============================================================
A polished web dashboard for disaster text classification.

Usage:
    streamlit run app/app.py
    
    (Run from the project root directory)

=============================================================================
STUDENT / ARCHITECTURAL OVERVIEW:
=============================================================================
What is this file?
- This file is the primary user interface (UI) for the Disaster Intelligence Engine.
- Built using Streamlit, an open-source Python framework designed for rapidly
  deploying data science and machine learning applications as interactive web apps.

How Streamlit Works (Execution Model):
- Streamlit follows a reactive, top-to-bottom script execution paradigm.
- Every time a user interacts with a widget (clicks a button, selects a radio
  option, or types into a text box), Streamlit reruns the ENTIRE script from
  line 1 to the end.
- To prevent heavy computations (like loading ML models or parsing JSON files)
  from running repeatedly on every user click, Streamlit provides caching
  decorators (@st.cache_resource and @st.cache_data).

IMPORTANT — Pre-Trained Inference vs. Retraining:
- This web application is strictly an INFERENCE and VISUALIZATION dashboard.
- It does NOT train or retrain machine learning models.
- All models are trained offline via `src/train.py`, which saves the best
  serialized model (`best_model.joblib`), vectorizer (`tfidf_vectorizer.joblib`),
  and performance metrics (`model_results.json`) into the `models/` directory.
- This app merely loads those pre-computed artifacts to classify incoming disaster
  reports in real-time and display pre-calculated evaluation benchmarks.

Application Structure:
1. Setup & Imports: System path configuration and module imports.
2. Page Config & CSS: Visual layout and custom emergency-themed styles.
3. Resource Loading: Cached loading of pre-trained model and evaluation JSON.
4. Session State: In-memory history tracking across web reruns.
5. Sidebar Navigation: Multi-page router directing user between 4 pages:
   - Page 1: Home / Analyzer (live text prediction with confidence scores)
   - Page 2: Dashboard (high-level dataset and model overview cards & charts)
   - Page 3: Model Performance (comparison table, confusion matrices, rationale)
   - Page 4: About (system architecture, CrisisMMD dataset info, limitations)
=============================================================================
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import matplotlib
# Set Matplotlib to 'Agg' backend (non-interactive, headless).
# This is crucial for web servers so Matplotlib doesn't attempt to open GUI desktop windows.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# PATH SETUP — Add project root so we can import from src/
# ============================================================
# Explanation for students:
# When executing `streamlit run app/app.py`, Python's working directory or module
# search path (sys.path) might only include the `app/` folder.
# To import modules from `src/` (e.g., `src.predict`, `src.preprocessing`), we
# dynamically determine the project root directory (one level up from `app/`)
# and insert it at index 0 of `sys.path`.
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import our custom project modules with graceful error handling.
# If dependencies or source files are missing, we capture the error rather than crashing
# immediately, allowing us to display an educational warning message to the user in the UI.
IMPORT_OK = True
try:
    from src.predict import predict, load_model
    from src.preprocessing import clean_text
except ImportError as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


# ============================================================
# PAGE CONFIG
# ============================================================
# `st.set_page_config` must be the first Streamlit command called on the page.
# - page_title: Sets the title displayed in the browser's tab.
# - page_icon: Sets the favicon (an emergency siren emoji).
# - layout="wide": Expands the interface to use the full browser width rather than a narrow central column.
# - initial_sidebar_state="expanded": Ensures the navigation sidebar is visible upon initial load.
st.set_page_config(
    page_title="Disaster Intelligence Engine",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — Professional emergency/disaster theme
# ============================================================
# How Custom CSS Works in Streamlit:
# Streamlit allows developers to inject raw HTML and CSS using `st.markdown(..., unsafe_allow_html=True)`.
# This allows us to customize default styling, giving our application a professional,
# emergency-response command center look:
# - Deep navy and red accents representing emergency management.
# - Styled metric cards with soft drop shadows and blue accent borders.
# - Prominent red primary action buttons for immediate visual hierarchy.
# - Color-coded prediction banners: soft green for 'Informative' and soft red for 'Not Informative'.
# - A sleek dark slate sidebar (#0f172a) providing high contrast against the light content area.
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fc; }
    
    /* Headers */
    h1 { color: #1e293b; }
    h2, h3 { color: #1e3a8a; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #1e3a8a;
    }
    
    /* Primary button */
    div.stButton > button[kind="primary"] {
        background-color: #dc2626;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #b91c1c;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    
    /* Example buttons */
    div.stButton > button {
        border-radius: 6px;
        font-size: 0.85rem;
    }
    
    /* Result boxes */
    .result-informative {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #166534;
        padding: 24px;
        border-radius: 12px;
        border-left: 8px solid #16a34a;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 16px 0;
    }
    .result-not-informative {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        padding: 24px;
        border-radius: 12px;
        border-left: 8px solid #dc2626;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 16px 0;
    }
    
    /* Info cards */
    .info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
# Why use @st.cache_resource?
# Loading a machine learning model and a TF-IDF vocabulary from disk involves
# deserializing binary files via joblib, which can take hundreds of milliseconds.
# `@st.cache_resource` ensures that `get_model_and_vectorizer()` executes ONLY ONCE
# when the app initializes. Its returned objects (model, vectorizer) remain cached
# in memory across all user sessions and reruns.
@st.cache_resource
def get_model_and_vectorizer():
    """
    Load pre-trained model and TF-IDF vectorizer once from disk, then cache in memory.
    
    Returns:
        tuple: (model, vectorizer) if files exist, else (None, None).
    """
    global MODEL_LOAD_ERROR
    try:
        model, vectorizer = load_model(str(PROJECT_ROOT / "models"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None
    except Exception as error:
        MODEL_LOAD_ERROR = str(error)
        return None, None


MODEL_LOAD_ERROR = None


# Why use @st.cache_data?
# `@st.cache_data` is used for serializable data structures (like dicts, dataframes, JSON).
# It caches the loaded JSON dictionary so that disk I/O is not repeated on every rerun.
@st.cache_data
def get_results():
    """
    Load pre-calculated evaluation results from models/model_results.json.
    
    Data Flow Explanation:
    During offline training (`src/train.py`), all evaluated models are tested on a
    held-out 20% test split. The evaluation metrics (accuracy, precision, recall, f1)
    and dataset metadata (sample counts, train/test split size) are exported to
    `models/model_results.json`. This function reads that JSON file and passes it
    directly into our Streamlit dashboard to power the metric cards and charts.
    
    Returns:
        dict: Parsed evaluation results dictionary, or None if file doesn't exist.
    """
    results_path = PROJECT_ROOT / "models" / "model_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def show_setup_warning():
    """
    Display setup instructions when pre-trained model files are missing.
    Educates the student or user on the prerequisite steps required to run the engine.
    """
    if MODEL_LOAD_ERROR:
        st.error(f"Model files were found but could not be loaded: {MODEL_LOAD_ERROR}")
        st.markdown(
            "Check that the deployed model files were created with compatible "
            "Python and scikit-learn versions."
        )
        return

    st.error("⚠️ **Model files not found!** The app needs its trained model artifacts.")
    st.markdown("""
    Add `models/disaster_model.pkl` and `models/tfidf_vectorizer.pkl` to the deployed
    repository. To create them, place your dataset TSV files in `data/raw/` and run
    the training script from the project root:
    ```bash
    python src/train.py
    ```
    Commit and push the generated model files, then redeploy the app.
    """)


# ============================================================
# LOAD RESOURCES
# ============================================================
# Execute the cached loader functions. Because of caching, these return
# almost instantaneously on all reruns after the first call.
model, vectorizer = get_model_and_vectorizer()
results = get_results()

# How Session State Maintains Prediction History:
# In web development, HTTP is stateless. Streamlit reruns the script on each user interaction,
# which would normally reset standard Python variables back to their initial values.
# `st.session_state` is a dictionary-like object that persists data across reruns for the
# current browser session.
# Here, we initialize 'prediction_history' as an empty list if it does not already exist.
# Every time the user tests a tweet/report, we prepend the result to this list,
# allowing the user to review all previous predictions made during their session.
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
# How Sidebar Navigation Works:
# Using the `with st.sidebar:` context manager, all nested Streamlit widgets are placed
# in the collapsible left navigation bar.
# We use `st.radio()` to present the available views. The selected option is stored
# in the variable `page`.
# Downstream `if/elif` blocks inspect `page` and render ONLY the corresponding page view.
# This creates a seamless multi-page application feel within a single script.
with st.sidebar:
    st.markdown("# 🚨 Disaster Intel")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home / Analyzer", "📊 Dashboard", "🔬 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #94a3b8;'>
        Disaster Intelligence Engine<br>
        v1.0 • College Project<br>
        ML + NLP + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: HOME / ANALYZER
# ============================================================
# What this page does:
# - Serves as the primary interactive inference tool for the engine.
# - Allows an emergency operator or student to input arbitrary text or choose from
#   realistic disaster report presets.
# - Cleans the text, transforms it using the pre-fitted TF-IDF vectorizer, and passes it
#   to the best trained classifier.
# - Displays the predicted category ('INFORMATIVE' vs 'NOT INFORMATIVE'), probability confidence,
#   and logs the classification to the session history.
if page == "🏠 Home / Analyzer":
    st.title("🚨 Disaster Intelligence Engine")
    st.markdown("##### AI-powered classification of disaster-related information")
    st.markdown("---")
    
    # Defensive checks: If model is not loaded or imports failed, stop execution cleanly.
    if model is None or vectorizer is None:
        show_setup_warning()
        st.stop()
    
    if not IMPORT_OK:
        st.error(f"Import error: {IMPORT_ERROR}")
        st.stop()
    
    # Layout: Divide screen into two columns.
    # col_input (3 parts): The main text area and analyze button.
    # col_examples (1 part): Quick-fill buttons with realistic test scenarios.
    col_input, col_examples = st.columns([3, 1])
    
    # Curated example texts representing diverse disaster communication styles.
    # These demonstrate edge cases:
    # - Informative: urgent rescue needs, infrastructure damage, emergency shelter announcements.
    # - Not Informative: casual emotional reactions, prayers, thoughts, and opinions without actionable details.
    EXAMPLES = {
        "🌊 Flood Rescue": "People are trapped after severe flooding near the river bridge. "
                           "Emergency rescue teams needed immediately. Water levels still rising.",
        "🏚️ Earthquake Damage": "Multiple buildings collapsed in downtown area after 6.8 magnitude "
                                "earthquake. At least 15 casualties reported. Roads blocked by debris.",
        "💬 Casual Tweet": "This storm is crazy omg 😭 hope everyone is okay",
        "🙏 Prayer Post": "Praying for all the victims of the hurricane. God bless everyone. "
                          "Stay strong 🙏❤️",
        "🏥 Relief Update": "Red Cross has set up 3 emergency shelters at Lincoln High School. "
                            "Clean water and medical supplies available. Contact 1-800-HELP.",
        "🌀 News Report": "Hurricane category 4 making landfall. Wind speeds exceeding 150 mph. "
                          "Mandatory evacuation order for coastal zones A through D.",
    }
    
    with col_examples:
        st.markdown("**📋 Quick Fill:**")
        selected_example = None
        # Create a button for each example. When clicked, Streamlit reruns and captures
        # the clicked text to populate the input box on the left.
        for name, text in EXAMPLES.items():
            if st.button(name, use_container_width=True):
                selected_example = text
    
    with col_input:
        # Pre-populate text area if an example button was clicked; otherwise keep empty.
        default_text = selected_example if selected_example else ""
        
        user_input = st.text_area(
            "📝 Enter Disaster Report Text:",
            value=default_text,
            height=160,
            placeholder="Type or paste a disaster-related report here...\n\n"
                        "Example: 'People are trapped after the flood near the bridge "
                        "and need immediate rescue.'"
        )
        
        # Primary call-to-action button to initiate model inference.
        analyze_btn = st.button("🔍 ANALYZE REPORT", type="primary", use_container_width=True)
    
    # --- PREDICTION RESULTS ---
    # How Predictions are Made and Displayed:
    # 1. Text is validated to ensure it's not empty or just whitespace.
    # 2. `predict()` in `src/predict.py` cleans the raw string (removes URLs, handles, special chars),
    #    computes TF-IDF feature weights, calls `model.predict_proba()`, and returns:
    #    - `label`: "Informative" or "Not Informative"
    #    - `confidence`: Maximum class probability float between 0.0 and 1.0 (e.g., 0.94 -> 94%).
    # 3. Dynamic HTML banners (.result-informative or .result-not-informative) render the outcome.
    # 4. A progress bar visualizes the model's confidence level.
    # 5. The result is logged into `st.session_state.prediction_history`.
    if analyze_btn and user_input.strip():
        with st.spinner("🔄 Analyzing report..."):
            try:
                label, confidence = predict(user_input, model, vectorizer)
                
                # Display result banner with corresponding color and educational explanation
                if label == "Informative":
                    st.markdown(
                        f'<div class="result-informative">'
                        f'✅ Prediction: INFORMATIVE</div>',
                        unsafe_allow_html=True
                    )
                    explanation = (
                        "The model classified this report as **informative** — it appears to "
                        "contain actionable disaster-related information such as damage reports, "
                        "casualty updates, rescue needs, or relief efforts."
                    )
                else:
                    st.markdown(
                        f'<div class="result-not-informative">'
                        f'❌ Prediction: NOT INFORMATIVE</div>',
                        unsafe_allow_html=True
                    )
                    explanation = (
                        "The model classified this report as **not informative** — it appears to "
                        "be conversational, opinion-based, or lacking specific actionable "
                        "disaster information."
                    )
                
                # Confidence bar: visual feedback of model certainty
                st.markdown(f"### 📊 Confidence: {confidence:.1%}")
                st.progress(min(confidence, 1.0))
                
                # Contextual explanation and disclaimer for emergency response ethics
                st.info(f"💡 {explanation}")
                st.caption(
                    "⚠️ *Note: The model predicts informativeness based on learned text "
                    "patterns. It does NOT verify factual accuracy.*"
                )
                
                # Save prediction to session history (insert at index 0 so newest appears at top)
                st.session_state.prediction_history.insert(0, {
                    "Report (preview)": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                    "Prediction": label,
                    "Confidence": f"{confidence:.1%}"
                })
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    
    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # --- PREDICTION HISTORY ---
    # Renders a table of all past classifications performed during the active session.
    # Users can inspect historical inferences or reset the history via a clear button.
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Session Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Clearing history empties the list and triggers a rerun to refresh the UI immediately
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


# ============================================================
# PAGE 2: DASHBOARD
# ============================================================
# What this page does:
# - Serves as an executive summary dashboard showing data flow from `model_results.json`.
# - Displays high-level Key Performance Indicators (KPIs) for the dataset:
#   total samples, informative count, not-informative count, and train/test split.
# - Displays best-model metrics: Accuracy, Precision, Recall, and F1 Score.
# - Renders visual analytics:
#   1. Bar chart comparing all evaluated models by F1 Score.
#   2. Pie chart showing target class distribution in the dataset.
elif page == "📊 Dashboard":
    st.title("📊 System Dashboard")
    st.markdown("---")
    
    if results is None:
        show_setup_warning()
        st.stop()
    
    # How Data Flows from model_results.json to the Dashboard:
    # The JSON structure contains:
    # {
    #   "best_model": "linear_svm",
    #   "dataset_stats": {"total_samples": 10000, "informative_count": 6000, ...},
    #   "models": {
    #       "logistic_regression": {"accuracy": 0.82, "f1": 0.81, ...},
    #       "linear_svm": {"accuracy": 0.85, "f1": 0.84, ...}
    #   }
    # }
    # We unpack these nested keys to drive the metrics and charts below.
    best_model_name = results.get("best_model", "Unknown")
    models_data = results.get("models", {})
    dataset_stats = results.get("dataset_stats", {})
    best_metrics = models_data.get(best_model_name, {})
    
    # --- Dataset Stats Row ---
    # `st.columns(4)` creates a 4-card responsive horizontal grid layout.
    st.markdown("### 📋 Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Samples", f"{dataset_stats.get('total_samples', 'N/A'):,}")
    d2.metric("Informative", f"{dataset_stats.get('informative_count', 'N/A'):,}")
    d3.metric("Not Informative", f"{dataset_stats.get('not_informative_count', 'N/A'):,}")
    d4.metric("Train / Test", f"{dataset_stats.get('train_size', '?')} / {dataset_stats.get('test_size', '?')}")
    
    st.markdown("---")
    
    # --- Best Model Stats Row ---
    # Highlights the metrics of whichever model achieved the highest F1 score during training.
    st.markdown(f"### 🏆 Best Model: {best_model_name.replace('_', ' ')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2%}")
    m2.metric("Precision", f"{best_metrics.get('precision', 0):.2%}")
    m3.metric("Recall", f"{best_metrics.get('recall', 0):.2%}")
    m4.metric("F1 Score", f"{best_metrics.get('f1', 0):.2%}")
    
    st.markdown("---")
    
    # --- Charts ---
    # Divides the lower section into two equal columns for visual comparison charts.
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Bar chart comparing F1 Scores across all trained algorithms.
        # F1 score is used because it balances Precision and Recall, making it the
        # primary metric for model selection in imbalanced classification.
        st.markdown("### 📊 Model Comparison (F1 Score)")
        f1_data = {name.replace('_', ' '): metrics.get('f1', 0) 
                    for name, metrics in models_data.items()}
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1e3a8a', '#dc2626', '#f59e0b', '#16a34a'][:len(f1_data)]
        bars = ax.bar(f1_data.keys(), f1_data.values(), color=colors)
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        # Add numerical labels on top of each bar for readability
        for bar, val in zip(bars, f1_data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with chart_col2:
        # Chart 2: Pie chart displaying ground truth class balance.
        # Visualizes whether the training data was balanced or skewed towards one category.
        st.markdown("### 📊 Class Distribution")
        info_count = dataset_stats.get('informative_count', 0)
        not_info_count = dataset_stats.get('not_informative_count', 0)
        
        if info_count > 0 or not_info_count > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = ['Informative', 'Not Informative']
            sizes = [info_count, not_info_count]
            colors = ['#16a34a', '#dc2626']
            explode = (0.05, 0.05)  # Slight separation between slices for visual clarity
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=explode, 
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            ax.set_title("Dataset Label Distribution")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Dataset statistics not available.")


# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
# What this page does:
# - Provides in-depth technical evaluation intended for machine learning students,
#   evaluators, and practitioners.
# - Displays a complete comparison table across all 4 algorithms with 4 key metrics:
#   Accuracy, Precision, Recall, and F1 Score (rounded to 4 decimal places).
# - Dynamically scans and renders pre-generated Confusion Matrix plots from `reports/figures/`.
# - Displays a comprehensive side-by-side metric comparison chart.
# - Includes educational deep dives:
#   1. Rationale for selecting the best model based on F1 Score rather than raw Accuracy.
#   2. Why Recall is the most critical metric in disaster and life-safety applications.
elif page == "🔬 Model Performance":
    st.title("🔬 Model Performance & Evaluation")
    st.markdown("---")
    
    if results is None:
        show_setup_warning()
        st.stop()
    
    models_data = results.get("models", {})
    best_model_name = results.get("best_model", "Unknown")
    
    # --- Comparison Table ---
    # Formats metrics into a clean, sortable tabular presentation.
    # A gold star (⭐) highlights the best-performing model.
    st.markdown("### 📋 All Models Comparison")
    
    table_data = []
    for name, metrics in models_data.items():
        is_best = "⭐ " if name == best_model_name else ""
        table_data.append({
            "Model": is_best + name.replace('_', ' '),
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Precision": f"{metrics.get('precision', 0):.4f}",
            "Recall": f"{metrics.get('recall', 0):.4f}",
            "F1 Score": f"{metrics.get('f1', 0):.4f}",
        })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Confusion Matrices ---
    # Confusion matrices provide insight into where models make mistakes:
    # - True Positives (TP): Informative correctly predicted as Informative.
    # - True Negatives (TN): Not Informative correctly predicted as Not Informative.
    # - False Positives (FP): Non-informative falsely flagged as Informative.
    # - False Negatives (FN): Informative report missed (flagged as Non-informative).
    # We display each saved confusion matrix image inside its own interactive tab.
    st.markdown("### 🔢 Confusion Matrices")
    
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    cm_files = sorted(figures_dir.glob("confusion_matrix_*.png")) if figures_dir.exists() else []
    
    if cm_files:
        # Show confusion matrices in dynamic tabs
        tab_names = [f.stem.replace("confusion_matrix_", "").replace("_", " ") for f in cm_files]
        tabs = st.tabs(tab_names)
        
        for tab, cm_file in zip(tabs, cm_files):
            with tab:
                st.image(str(cm_file), use_container_width=True)
    else:
        st.info("Confusion matrix images not found. Run training first.")
    
    # --- Model Comparison Chart ---
    # Displays the multi-metric grouped bar chart generated by `src/train.py`.
    comparison_chart = PROJECT_ROOT / "reports" / "figures" / "model_comparison.png"
    if comparison_chart.exists():
        st.markdown("---")
        st.markdown("### 📊 Visual Comparison")
        st.image(str(comparison_chart), use_container_width=True)
    
    st.markdown("---")
    
    # --- Explanations ---
    # Educational callout boxes explaining the machine learning rationale behind metric choices.
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Model Selection Rationale")
        st.success(f"**Selected: {best_model_name.replace('_', ' ')}**")
        st.markdown("""
        The best model was selected based on the **F1 Score**, which provides a 
        balanced measure between precision and recall.
        
        - **Accuracy alone** can be misleading with imbalanced classes.
        - **F1 Score** penalizes both false positives and false negatives equally.
        - This makes it ideal for disaster information filtering where both 
        types of errors have real-world consequences.
        """)
    
    with col2:
        st.markdown("### ⚠️ Why Recall Matters")
        st.warning("""
        **In disaster response, missing an informative report can cost lives.**
        
        - A **False Negative** (informative report classified as not informative) 
          means a cry for help might be ignored.
        - A **False Positive** (non-informative classified as informative) 
          wastes resources but is less dangerous.
        - Therefore, **high recall for the informative class** is especially 
          critical in real-world deployment.
        """)


# ============================================================
# PAGE 4: ABOUT
# ============================================================
# What this page does:
# - Serves as project documentation and viva/presentation reference for students.
# - Covers the core problem statement: filtering social media noise during emergencies.
# - Describes the CrisisMMD multimodal disaster dataset and events covered.
# - Illustrates the end-to-end Machine Learning pipeline (from raw TSV to web UI).
# - Details the 4 algorithms trained (Logistic Regression, Naive Bayes, Linear SVM, Random Forest)
#   and explains why each was chosen.
# - Explicitly outlines system limitations and ethical disclaimers (preventing automation bias).
# - Outlines future multi-modal research directions (OCR, satellite verification, emergency routing).
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Problem Statement
    
    During natural disasters, social media platforms are flooded with millions of posts. 
    Emergency responders need to quickly distinguish between:
    - **Informative** posts (damage reports, rescue needs, relief updates)
    - **Not informative** posts (opinions, prayers, casual comments)
    
    The **Disaster Intelligence Engine** automates this filtering using Machine Learning 
    and Natural Language Processing.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Dataset
    
    **CrisisMMD v2.0** (Crisis Multimodal Dataset)
    - **Source**: [CrisisNLP - QCRI](https://crisisnlp.qcri.org/crisismmd)
    - **Content**: Disaster-related social media posts with human annotations
    - **Task**: Binary classification — Informative vs Not Informative
    - **Events covered**: Hurricane Harvey, Hurricane Irma, Hurricane Maria, 
      California Wildfires, Iraq-Iran Earthquake, Mexico Earthquake, Sri Lanka Floods
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### ⚙️ ML Pipeline
    
    ```
    📂 Raw Dataset (CrisisMMD TSV files)
         │
         ▼
    🧹 Data Cleaning & Preprocessing
         │  • Remove URLs, mentions, noise
         │  • Lowercase, normalize whitespace
         │  • Remove duplicates
         ▼
    📝 TF-IDF Feature Extraction
         │  • Unigrams + Bigrams
         │  • max_features=10,000
         │  • Fit on training data ONLY
         ▼
    ✂️ Stratified Train/Test Split (80/20)
         │
         ▼
    🤖 Train 4 ML Models
         │  • Logistic Regression
         │  • Multinomial Naive Bayes
         │  • Linear SVM (CalibratedClassifierCV)
         │  • Random Forest
         ▼
    📊 Evaluate & Compare (F1 Score)
         │
         ▼
    💾 Save Best Model + Vectorizer
         │
         ▼
    🌐 Streamlit Web Application
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🧠 Algorithms Used
    
    | Algorithm | Description | Why Used |
    |-----------|-------------|----------|
    | **Logistic Regression** | Linear model for binary classification | Strong baseline, interpretable, fast |
    | **Multinomial Naive Bayes** | Probabilistic classifier based on Bayes' theorem | Excellent for text/word count data |
    | **Linear SVM** | Finds optimal hyperplane to separate classes | Effective in high-dimensional TF-IDF space |
    | **Random Forest** | Ensemble of decision trees | Robust, handles non-linear patterns |
    
    > **Note**: Linear SVM is wrapped in `CalibratedClassifierCV` to provide proper 
    > probability estimates for confidence scores.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛑 Limitations")
        st.error("""
        **Important Disclaimers:**
        
        - The system predicts **informativeness**, NOT factual truth
        - Based on historical social media data — may not generalize to new disaster types
        - Does NOT verify reports using satellites, GPS, or external sources
        - Should NOT be treated as an autonomous emergency response system
        - Predictions may contain errors or bias from the training data
        - Model accuracy depends on the quality and diversity of training data
        """)
    
    with col2:
        st.markdown("### 🚀 Future Improvements")
        st.info("""
        **Planned enhancements (not implemented):**
        
        - 🖼️ OCR from disaster images
        - 📸 Image classification for damage assessment
        - 🔗 Multimodal text + image analysis
        - 📍 GPS/location verification
        - 📡 Real-time social media monitoring
        - 🔄 Duplicate report detection
        - ✅ Cross-source verification
        - 🛰️ Satellite imagery analysis
        - 🚑 Emergency response prioritization
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 👨‍💻 Technical Stack
    
    `Python` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Seaborn` `Streamlit` `joblib`
    
    ---
    
    *Disaster Intelligence Engine — B.Tech CSE (AI & ML) Project*
    """)
