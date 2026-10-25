"""
data_loader.py - Dataset Loading Module
========================================
Loads CrisisMMD dataset TSV files from data/raw/ directory.
Combines multiple annotation files and extracts text + informativeness labels.

=============================================================================
🎓 STUDENT GUIDE - ROLE IN THE MACHINE LEARNING PIPELINE:
-----------------------------------------------------------------------------
This module is Stage 1 (Data Ingestion & Extraction) of the ML pipeline:
  [Raw TSV Files] ---> [data_loader.py] ---> [data/processed/processed_data.csv]
                              |
                              v
                     [preprocessing.py] ---> [features.py] ---> [train.py]

1. What is CrisisMMD?
   CrisisMMD is a benchmark multimodal crisis dataset containing thousands
   of tweets collected during major natural disasters (e.g., Hurricane Harvey,
   Mexico Earthquake, California Wildfires).

2. Why do we need this loader?
   - The dataset is split across multiple TSV files (one per disaster event).
   - Column names in these files vary (e.g., 'tweet_text' vs 'text', 'label' vs
     'informativeness_label').
   - This loader auto-detects column names, unifies the schema, combines all
     events into a single master dataset, and filters for informativeness.

3. Downstream Data Flow:
   Outputs 'processed_data.csv' which feeds directly into 'src/preprocessing.py'
   for text cleaning, noise removal, and label encoding.
=============================================================================

Usage:
    python src/data_loader.py
"""

import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration:
# Dynamically locate the project root folder (one directory above 'src/').
# Using dynamic pathing ensures this code runs correctly on any machine
# regardless of where the project directory is cloned or executed from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_tsv_files(raw_dir):
    """
    Recursively find all TSV files in the raw data directory.
    
    Why this is needed:
    CrisisMMD files are often organized into subfolders by event or task
    (e.g., data/raw/hurricane_harvey/text_annotations.tsv). Using recursive
    glob search ('**/*.tsv') guarantees we discover all TSV files no matter
    how deeply nested they are within the raw directory.
    
    Args:
        raw_dir (str): Path to the raw data directory.
    
    Returns:
        list: List of paths to TSV files found.
    """
    # glob with recursive=True matches *.tsv in raw_dir and any subdirectories
    tsv_files = glob.glob(os.path.join(raw_dir, "**", "*.tsv"), recursive=True)
    
    # If no files found, provide clear, step-by-step instructions to the user
    # to guide them on downloading and placing the dataset.
    if not tsv_files:
        print(f"\n❌ No TSV files found in: {raw_dir}")
        print("\n📋 DATASET SETUP INSTRUCTIONS:")
        print("=" * 60)
        print("1. Visit: https://crisisnlp.qcri.org/crisismmd")
        print("2. Download the CrisisMMD v2.0 dataset")
        print("   (You may need to fill out a request form)")
        print("3. Extract the downloaded archive")
        print("4. Copy ALL .tsv annotation files into:")
        print(f"   {raw_dir}")
        print("5. Run this script again")
        print("\nAlternative sources:")
        print("  - Kaggle: Search 'CrisisMMD'")
        print("  - HuggingFace: Search 'CrisisMMD'")
        print("=" * 60)
        return []
    
    print(f"✅ Found {len(tsv_files)} TSV file(s):")
    for f in tsv_files:
        print(f"   - {os.path.basename(f)}")
    
    return tsv_files


def detect_columns(df):
    """
    Auto-detect the text column and label column from the DataFrame.
    Handles variations in CrisisMMD column naming conventions.
    
    Why auto-detection is necessary:
    In open-source datasets like CrisisMMD, different annotations were created
    across multiple tasks (e.g., image informativeness vs. tweet text informativeness).
    As a result, column names often differ across files:
      - Text: 'tweet_text', 'text', 'message', 'tweet_text ' (trailing spaces)
      - Label: 'label', 'informativeness_label', 'text_info', etc.
    Hardcoding a single column name would crash when reading different files.
    
    Args:
        df (pd.DataFrame): Raw DataFrame loaded from TSV.
    
    Returns:
        tuple: (text_column_name, label_column_name, event_column_name or None)
    """
    # Create a lowercase, stripped mapping of column names:
    # Key = normalized column name (e.g., 'tweet_text')
    # Value = original column name in the DataFrame (e.g., 'tweet_text ')
    # This prevents bugs caused by accidental casing or leading/trailing whitespace.
    columns_lower = {col.lower().strip(): col for col in df.columns}
    
    # --- Detect text column ---
    # Ordered by likelihood: search through common names used in crisis NLP datasets
    text_candidates = [
        "tweet_text", "text", "tweet", "message", "content",
        "tweet_text ", "text_content"
    ]
    text_col = None
    for candidate in text_candidates:
        if candidate in columns_lower:
            text_col = columns_lower[candidate]
            break
    
    # --- Detect label column ---
    # The label tells whether the tweet is informative or not
    label_candidates = [
        "label", "informativeness_label", "informativeness",
        "label_text", "text_info", "text_info_label",
        "label_image", "info_label", "class", "category"
    ]
    label_col = None
    for candidate in label_candidates:
        if candidate in columns_lower:
            label_col = columns_lower[candidate]
            break
    
    # --- Detect event/disaster type column ---
    # Optional metadata indicating which disaster the tweet was from
    event_candidates = [
        "event_name", "event", "disaster_type", "event_type",
        "crisis_type", "disaster"
    ]
    event_col = None
    for candidate in event_candidates:
        if candidate in columns_lower:
            event_col = columns_lower[candidate]
            break
    
    return text_col, label_col, event_col


def load_single_tsv(filepath):
    """
    Load a single TSV file into a DataFrame.
    
    Why TSV instead of CSV?
    Social media posts contain many punctuation marks, including commas.
    Tab-Separated Values (TSV) use tabs ('\t') as separators, which prevents
    accidental column splitting when tweet text contains commas.
    
    Args:
        filepath (str): Path to the TSV file.
    
    Returns:
        pd.DataFrame or None: Loaded DataFrame, or None if loading fails.
    """
    try:
        # Try tab-separated first (standard format for CrisisMMD files)
        # on_bad_lines='skip' ignores corrupted rows without crashing the script
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')
        
        # Fallback: If separator was comma instead of tab, all data ends up in 1 column
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8', on_bad_lines='skip')
        
        print(f"   Loaded {os.path.basename(filepath)}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        # Gracefully handle file I/O or parsing errors so one bad file doesn't halt the pipeline
        print(f"   ⚠️ Error loading {os.path.basename(filepath)}: {e}")
        return None


def load_dataset(raw_dir=None):
    """
    Main function to load the CrisisMMD dataset.
    Finds all TSV files, loads them, detects columns, and combines them.
    
    Why combine multiple disaster files?
    If we train an ML model only on 'hurricane_harvey' data, it will overfit
    to hurricane-specific terms (e.g., 'surge', 'rainfall', 'wind').
    By combining multiple disaster events (earthquakes, floods, wildfires, hurricanes),
    the model learns generic disaster intelligence vocabulary (e.g., 'shelter',
    'injured', 'casualties', 'evacuation', 'urgent help') and generalizes better
    to future unseen disasters.
    
    Args:
        raw_dir (str, optional): Path to raw data directory. 
                                  Defaults to data/raw/ in project root.
    
    Returns:
        pd.DataFrame: Combined dataset with columns: 'text', 'label', 'event' (optional).
                      Returns empty DataFrame if no data found.
    """
    if raw_dir is None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    
    # Create directory if it doesn't exist yet
    os.makedirs(raw_dir, exist_ok=True)
    
    # Find all TSV files across directory and subdirectories
    tsv_files = find_tsv_files(raw_dir)
    if not tsv_files:
        return pd.DataFrame()
    
    # Load and combine all files
    print("\n📂 Loading TSV files...")
    all_dataframes = []
    
    for filepath in tsv_files:
        df = load_single_tsv(filepath)
        if df is not None and len(df) > 0:
            # Auto-detect the varied column names for this specific file
            text_col, label_col, event_col = detect_columns(df)
            
            # Validation: We strictly need both text and label for supervised learning
            if text_col is None:
                print(f"   ⚠️ Could not find text column in {os.path.basename(filepath)}")
                print(f"      Available columns: {list(df.columns)}")
                continue
            
            if label_col is None:
                print(f"   ⚠️ Could not find label column in {os.path.basename(filepath)}")
                print(f"      Available columns: {list(df.columns)}")
                continue
            
            # Standardize column names into a unified schema: 'text', 'label', 'event'
            # This ensures all DataFrames have identical structure before concatenation.
            standardized = pd.DataFrame()
            standardized['text'] = df[text_col]
            standardized['label'] = df[label_col]
            
            if event_col is not None:
                standardized['event'] = df[event_col]
            
            # If the file didn't have an explicit 'event' column, infer it from the filename
            # e.g., 'hurricane_harvey_final_data.tsv' -> 'hurricane_harvey'
            if event_col is None:
                filename = os.path.basename(filepath).lower()
                # Common CrisisMMD event names in filenames
                events = [
                    'hurricane_harvey', 'hurricane_irma', 'hurricane_maria',
                    'california_wildfires', 'iraq_iran_earthquake',
                    'mexico_earthquake', 'srilanka_floods'
                ]
                for event in events:
                    if event.replace('_', '') in filename.replace('_', '').replace('-', ''):
                        standardized['event'] = event
                        break
            
            all_dataframes.append(standardized)
    
    if not all_dataframes:
        print("\n❌ No valid data could be extracted from the TSV files.")
        return pd.DataFrame()
    
    # Combine all DataFrames row-wise into one single master DataFrame
    # ignore_index=True resets the index so it is contiguous [0, 1, ..., N-1]
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total rows")
    
    # --- Filter to informativeness labels only ---
    # Why filter labels?
    # 1. The original CrisisMMD dataset contains humanitarian categories
    #    (e.g., 'infrastructure_damage', 'rescue_volunteering') as well as
    #    informativeness ('informative', 'not_informative') or invalid entries.
    # 2. For our binary classification task ("Is this post useful for emergency response?"),
    #    we want clean ground-truth binary targets: 'informative' vs 'not_informative'.
    
    # Normalize labels: convert to string, remove whitespace, and convert to lowercase
    combined_df['label'] = combined_df['label'].astype(str).str.strip().str.lower()
    
    # Keep only informative / not_informative labels
    valid_labels = ['informative', 'not_informative']
    filtered_df = combined_df[combined_df['label'].isin(valid_labels)].copy()
    
    removed = len(combined_df) - len(filtered_df)
    if removed > 0:
        print(f"   Filtered out {removed} rows with other labels")
    
    print(f"   Final dataset: {len(filtered_df)} rows")
    
    # --- Print label distribution ---
    # In ML, checking class distribution (balance vs. imbalance) is critical.
    # If one class represents 90% of data, standard accuracy can be misleading.
    print("\n📈 Label Distribution:")
    label_counts = filtered_df['label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(filtered_df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    return filtered_df


def save_processed_data(df, output_path=None):
    """
    Save the processed DataFrame to a CSV file.
    
    Data Handoff in Pipeline:
    Saves the cleaned and unified dataset to disk (data/processed/processed_data.csv).
    This decouples data extraction from preprocessing, allowing next steps
    (e.g., preprocessing.py) to load the consolidated dataset quickly without
    re-reading and parsing all raw TSVs each time.
    
    Args:
        df (pd.DataFrame): Processed dataset.
        output_path (str, optional): Path to save. Defaults to data/processed/processed_data.csv.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv")
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save without the pandas index column (index=False keeps file clean)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Processed data saved to: {output_path}")


# ============================================================
# Main execution - Run this script to load and process the dataset
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚨 Disaster Intelligence Engine - Data Loader")
    print("=" * 60)
    
    # Execute loading, column normalization, and filtering
    df = load_dataset()
    
    if len(df) == 0:
        print("\n⚠️ No data loaded. Please follow the setup instructions above.")
    else:
        # Print summary statistics for inspection
        print("\n" + "=" * 60)
        print("📋 DATASET SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        # Check distribution across disaster events if available
        if 'event' in df.columns:
            print(f"\n🌪️ Disaster Events:")
            event_counts = df['event'].value_counts()
            for event, count in event_counts.items():
                print(f"   {event}: {count}")
        
        # Persist consolidated dataset to disk for preprocessing.py
        save_processed_data(df)
        print("\n✅ Data loading complete!")
