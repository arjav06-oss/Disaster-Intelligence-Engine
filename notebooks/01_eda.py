"""
01_eda.py - Exploratory Data Analysis
========================================
Comprehensive EDA for the CrisisMMD disaster text dataset.
Generates visualizations and saves them to reports/figures/.

Can be run as a script or converted to Jupyter notebook.

Usage:
    python notebooks/01_eda.py

=============================================================================
STUDENT / CONCEPTUAL GUIDE TO EXPLORATORY DATA ANALYSIS (EDA):
=============================================================================
What is EDA and why is it important in Machine Learning?
- Exploratory Data Analysis (EDA) is the foundational first step of any data
  science and machine learning project.
- Before writing any model code (like SVM or Naive Bayes), a machine learning
  engineer must deeply understand the underlying dataset.
- "Garbage In, Garbage Out" (GIGO) is the golden rule of AI: a model trained
  on dirty, imbalanced, or misunderstood data will inevitably perform poorly
  in production, regardless of algorithm sophistication.

What this script accomplishes:
1. Data Integrity Checks:
   - Validates dataset shape, column types, missing values, and duplicate records.
   - Social media text often contains retweets and duplicate spam that can artificially
     inflate test performance (data leakage) if not audited.
2. Class Balance Analysis:
   - Evaluates whether positive ('informative') and negative ('not_informative')
     samples are evenly represented, preventing majority-class prediction bias.
3. Text Length & Word Count Distributions:
   - Compares the length profiles of informative vs. non-informative reports
     using histograms and boxplots to detect structural differences.
4. Vocabulary & Keyword Analysis:
   - Extracts word frequencies after stop-word removal.
   - Highlights the distinct semantic vocabulary separation between actionable
     crisis reports and casual/emotional social posts.
5. Automated Visual Asset Generation:
   - Exports publication-ready visualization charts to `reports/figures/` for use
     in technical reports and the Streamlit dashboard.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
# Headless rendering backend: generates plot images without requiring a GUI display server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ============================================================
# PATH SETUP — Enable imports from project src/ directory
# ============================================================
# Determines absolute path of the current script and inserts the root project
# directory into sys.path so modules like `src.data_loader` can be imported cleanly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, save_processed_data
from src.preprocessing import clean_text

# Configure global visualization aesthetic
# 'whitegrid' provides clean horizontal/vertical reference gridlines for easier comparison
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Output directory where generated plots will be saved
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(fig, filename):
    """
    Save a Matplotlib figure to the reports/figures/ directory with consistent DPI.
    
    Why this helper exists:
    - Encapsulates file path resolution and directory management.
    - Saves at 150 DPI (dots per inch) for crisp text and graphics.
    - Uses bbox_inches='tight' so labels and legends are never cropped off.
    - Calls plt.close(fig) to free memory immediately after saving.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to persist.
        filename (str): Target filename (e.g., 'eda_class_distribution.png').
    """
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 Saved: {filename}")


def main():
    """
    Main execution workflow for Exploratory Data Analysis.
    Sequentially inspects dataset dimensions, null values, duplicates, class distributions,
    text length properties, and class-specific n-gram vocabulary.
    """
    print("=" * 60)
    print("📊 DISASTER INTELLIGENCE ENGINE — Exploratory Data Analysis")
    print("=" * 60)
    
    # ============================================================
    # 1. Load Dataset
    # ============================================================
    # The data loader aggregates raw CrisisMMD TSV annotations from data/raw/
    # into a unified Pandas DataFrame with columns: ['text', 'label', 'event'].
    print("\n" + "=" * 60)
    print("1️⃣ LOADING DATASET")
    print("=" * 60)
    
    df = load_dataset()
    
    # Defensive programming: ensure files exist before continuing analysis
    if df is None or len(df) == 0:
        print("\n❌ No data available. Please download the CrisisMMD dataset first.")
        print("See README.md for instructions.")
        return
    
    # ============================================================
    # 2. Basic Dataset Info
    # ============================================================
    # Why inspect shape and data types?
    # - Verifies sample volume (sample count N dictates whether deep learning or classical ML is suitable).
    # - Ensures target and text columns have expected object/string data types.
    # - `df.head()` gives a concrete qualitative glimpse of real data rows.
    print("\n" + "=" * 60)
    print("2️⃣ DATASET SHAPE & INFO")
    print("=" * 60)
    
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn Names: {list(df.columns)}")
    print(f"\nData Types:")
    print(df.dtypes.to_string())
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    
    # ============================================================
    # 3. Missing Values
    # ============================================================
    # Why missing value analysis is critical in NLP:
    # - Missing values (NaN / None) in text columns will crash tokenizers and
    #   vectorizers (e.g. TfidfVectorizer.fit_transform expects string inputs).
    # - If labels are missing, supervised learning cannot use those instances.
    print("\n" + "=" * 60)
    print("3️⃣ MISSING VALUES")
    print("=" * 60)
    
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100)
    
    missing_info = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct.round(2)
    })
    print(missing_info.to_string())
    
    if missing.sum() == 0:
        print("\n✅ No missing values found!")
    
    # ============================================================
    # 4. Duplicate Records
    # ============================================================
    # Why duplicate analysis matters:
    # - In social media datasets, identical tweets arise frequently due to retweets,
    #   syndicated bot broadcasts, or news syndication.
    # - If duplicate texts appear in both the training and test splits, the model
    #   will achieve artificially high test scores (data leakage), hiding its true
    #   inability to generalize to unseen data.
    print("\n" + "=" * 60)
    print("4️⃣ DUPLICATE RECORDS")
    print("=" * 60)
    
    duplicates = df.duplicated(subset=['text']).sum()
    print(f"Duplicate texts: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # ============================================================
    # 5. Target Class Distribution
    # ============================================================
    # Why Class Distribution Matters:
    # - Class imbalance is one of the most critical challenges in classification.
    # - If 90% of samples are 'not_informative' and 10% are 'informative', a naive
    #   classifier that always guesses 'not_informative' achieves 90% accuracy
    #   while being completely useless for disaster response!
    # - Checking the balance informs:
    #   1. Metric Selection: When classes are imbalanced, Accuracy is misleading;
    #      F1-Score, Precision, and Recall must be prioritized.
    #   2. Sampling Strategy: Informs whether stratified train-test splitting is mandatory.
    #
    # What the Visualizations Show:
    # - Subplot 1 (Bar Chart): Shows raw sample volume for each category with count annotations.
    # - Subplot 2 (Pie Chart): Shows proportion percentages with separated slices for visual clarity.
    print("\n" + "=" * 60)
    print("5️⃣ TARGET CLASS DISTRIBUTION")
    print("=" * 60)
    
    label_counts = df['label'].value_counts()
    print(label_counts.to_string())
    
    # Plot class distribution (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: visualizes absolute frequency
    colors = ['#16a34a', '#dc2626']
    label_counts.plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    for i, (label, count) in enumerate(label_counts.items()):
        axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    # Pie chart: visualizes relative proportions
    axes[1].pie(
        label_counts.values,
        labels=label_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Target Variable Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'eda_class_distribution.png')
    
    # ============================================================
    # 6. Disaster Event Distribution (if available)
    # ============================================================
    # What Disaster Event Analysis Shows:
    # - The CrisisMMD dataset covers multiple natural disaster events across the globe
    #   (e.g., Hurricane Irma, California Wildfires, Mexico Earthquake).
    # - Inspecting event distribution reveals whether certain disaster types dominate the dataset.
    # - The stacked bar chart (Event vs. Label) explores domain shift: do earthquakes generate
    #   a higher proportion of informative tweets than hurricanes? Understanding this helps
    #   evaluate if the trained model will generalize across different disaster types.
    if 'event' in df.columns:
        print("\n" + "=" * 60)
        print("6️⃣ DISASTER EVENT DISTRIBUTION")
        print("=" * 60)
        
        event_counts = df['event'].value_counts()
        print(event_counts.to_string())
        
        # Horizontal bar chart of report counts per disaster event
        fig, ax = plt.subplots(figsize=(12, 6))
        event_counts.plot(kind='barh', ax=ax, color=sns.color_palette('viridis', len(event_counts)))
        ax.set_title('Distribution of Disaster Events', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Reports')
        ax.set_ylabel('Disaster Event')
        for i, count in enumerate(event_counts.values):
            ax.text(count + 10, i, str(count), va='center', fontweight='bold')
        plt.tight_layout()
        save_fig(fig, 'eda_event_distribution.png')
        
        # Event vs Label cross-tabulation: explores label proportion per disaster
        cross_tab = pd.crosstab(df['event'], df['label'])
        print(f"\nEvent x Label Cross-tabulation:")
        print(cross_tab.to_string())
        
        # Stacked bar chart showing informative vs not_informative breakdown per event
        fig, ax = plt.subplots(figsize=(12, 6))
        cross_tab.plot(kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_title('Informativeness by Disaster Event', fontsize=14, fontweight='bold')
        ax.set_xlabel('Disaster Event')
        ax.set_ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        plt.legend(title='Label')
        plt.tight_layout()
        save_fig(fig, 'eda_event_vs_label.png')
    
    # ============================================================
    # 7. Text Length Distribution
    # ============================================================
    # What Text Length Analysis Reveals:
    # - Length in characters (`text_length`) and word count (`word_count`):
    #   1. Social Media Constraints: Reveals platform-specific boundaries (e.g. Twitter's
    #      historical 140 or 280 character limits).
    #   2. Feature Informativeness: Informative tweets often contain specific location names,
    #      directions, contact numbers, and descriptive casualty reports, making them
    #      consistently longer on average than casual reactions.
    #   3. Outlier Detection: Identifies ultra-short noise (e.g., "omg", "sad") or abnormally
    #      long concatenated strings that could skew TF-IDF statistics.
    #
    # Visualizations:
    # - Overlaid Histograms: Shows shape, spread, and peak densities of character & word counts.
    # - Boxplots: Shows medians, interquartile ranges (IQR: 25th to 75th percentile), and outliers.
    print("\n" + "=" * 60)
    print("7️⃣ TEXT LENGTH DISTRIBUTION")
    print("=" * 60)
    
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\nText Length Statistics:")
    print(df['text_length'].describe().to_string())
    
    print(f"\nWord Count Statistics:")
    print(df['word_count'].describe().to_string())
    
    # Overlaid histograms comparing distributions across classes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, color in zip(['informative', 'not_informative'], colors):
        subset = df[df['label'] == label]
        axes[0].hist(subset['text_length'], bins=50, alpha=0.6, label=label, color=color)
        axes[1].hist(subset['word_count'], bins=40, alpha=0.6, label=label, color=color)
    
    axes[0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Character Count')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    save_fig(fig, 'eda_text_length_distribution.png')
    
    # Box plots comparing medians and quartiles by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=df, x='label', y='text_length', ax=axes[0], palette=colors)
    axes[0].set_title('Text Length by Class', fontsize=14, fontweight='bold')
    
    sns.boxplot(data=df, x='label', y='word_count', ax=axes[1], palette=colors)
    axes[1].set_title('Word Count by Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'eda_text_length_boxplot.png')
    
    # ============================================================
    # 8. Word Frequency Analysis
    # ============================================================
    # What Word Frequency Analysis Shows:
    # - Identifies the most prominent terms in the disaster corpus.
    # - Text Cleaning Step: Applies `clean_text` to strip URLs, user handles (@mentions),
    #   and punctuation so frequency counting captures meaningful root tokens.
    # - Stop Words Removal:
    #   Common English words ('the', 'is', 'at', 'and') appear with high frequency
    #   in virtually every text document, but provide zero discriminative signal for
    #   distinguishing emergency information. We filter them out to uncover domain vocabulary.
    print("\n" + "=" * 60)
    print("8️⃣ WORD FREQUENCY ANALYSIS")
    print("=" * 60)
    
    # Clean texts using our standard preprocessing pipeline
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Flatten all tokens across the corpus and compute frequency counts
    all_words = ' '.join(df['clean_text'].values).split()
    word_freq = Counter(all_words)
    
    # Domain-specific stop words set (including social media noise like 'rt' for retweet)
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
        'to', 'for', 'of', 'and', 'or', 'but', 'not', 'it', 'this',
        'that', 'with', 'from', 'by', 'as', 'be', 'has', 'have', 'had',
        'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
        'may', 'might', 'shall', 'if', 'so', 'no', 'all', 'my', 'we',
        'they', 'you', 'he', 'she', 'i', 'me', 'us', 'them', 'our',
        'your', 'his', 'her', 'its', 'up', 'out', 'just', 'rt'
    }
    
    # Exclude stop words and single/two-letter noise
    filtered_freq = {word: count for word, count in word_freq.items() 
                     if word not in stop_words and len(word) > 2}
    
    top_20 = Counter(filtered_freq).most_common(20)
    print(f"\nTop 20 words (excluding stop words):")
    for word, count in top_20:
        print(f"   {word}: {count}")
    
    # Horizontal bar chart: Top 20 terms across the entire corpus
    fig, ax = plt.subplots(figsize=(12, 6))
    words, counts = zip(*top_20)
    ax.barh(range(len(words)), counts, color='#1e3a8a')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # Puts the highest frequency word at the very top
    ax.set_title('Top 20 Most Frequent Words', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    save_fig(fig, 'eda_word_frequency.png')
    
    # ============================================================
    # 9. Most Common Words by Class
    # ============================================================
    # How Informative vs. Not-Informative Terms Differ:
    # - This is the central linguistic hypothesis of our NLP classifier!
    # - Informative Terms:
    #   Actionable, concrete nouns and verbs describing operational reality:
    #   e.g., 'damage', 'relief', 'shelter', 'victims', 'rescue', 'emergency',
    #   'water', 'hospital', 'collapsed', 'redcross'.
    # - Not-Informative Terms:
    #   Conversational, emotional, personal, or religious sentiments:
    #   e.g., 'prayers', 'god', 'bless', 'stay', 'safe', 'crazy', 'hope', 'sad', 'scary'.
    # - This sharp semantic divergence proves that linear classifiers (like Logistic
    #   Regression and Linear SVM) with TF-IDF features will be highly effective,
    #   because the presence of specific keywords strongly separates the two classes.
    print("\n" + "=" * 60)
    print("9️⃣ MOST COMMON WORDS BY CLASS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (label, title, color) in enumerate([
        ('informative', 'Most Common INFORMATIVE Words', '#16a34a'),
        ('not_informative', 'Most Common NOT INFORMATIVE Words', '#dc2626')
    ]):
        subset = df[df['label'] == label]
        class_words = ' '.join(subset['clean_text'].values).split()
        class_freq = {word: count for word, count in Counter(class_words).items() 
                      if word not in stop_words and len(word) > 2}
        top_15 = Counter(class_freq).most_common(15)
        
        print(f"\n   Top 15 {label} words:")
        for word, count in top_15:
            print(f"      {word}: {count}")
        
        if top_15:
            words, counts = zip(*top_15)
            axes[idx].barh(range(len(words)), counts, color=color)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_title(title, fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Frequency')
    
    plt.tight_layout()
    save_fig(fig, 'eda_words_by_class.png')
    
    # ============================================================
    # Summary
    # ============================================================
    # Synthesizes all empirical findings into clear takeaway bullet points.
    # Evaluates the class imbalance ratio to inform training configuration.
    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {FIGURES_DIR}")
    
    # List all generated figure artifacts
    eda_files = [f for f in os.listdir(FIGURES_DIR) if f.startswith('eda_')]
    for f in sorted(eda_files):
        print(f"   📊 {f}")
    
    print(f"\n📋 Key Findings:")
    print(f"   • Dataset size: {len(df)} samples")
    print(f"   • Classes: {dict(label_counts)}")
    
    # Imbalance ratio rule of thumb: ratio > 1.5 indicates moderate to severe imbalance
    imbalance = label_counts.max() / label_counts.min()
    if imbalance > 1.5:
        print(f"   • ⚠️ Class imbalance ratio: {imbalance:.2f}x")
    else:
        print(f"   • ✅ Classes are relatively balanced ({imbalance:.2f}x)")
    
    print(f"   • Average text length: {df['text_length'].mean():.0f} chars / {df['word_count'].mean():.0f} words")
    print(f"   • Duplicates: {duplicates}")


if __name__ == "__main__":
    main()
