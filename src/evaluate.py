"""
evaluate.py - Model Evaluation Module
======================================
Evaluates ML models and generates performance visualizations.
Produces confusion matrices, classification reports, and comparison charts.

Educational Overview for AI/ML Students:
----------------------------------------
Evaluating machine learning models is just as crucial as training them. In disaster response,
a high overall accuracy can be dangerously misleading if the model achieves it by simply
predicting "Not Informative" for everything.

This module computes standard classification metrics, renders confusion matrix heatmaps,
and generates side-by-side grouped bar charts to justify model selection to stakeholders.

Key Concepts:
- Accuracy: Overall percentage of correct predictions.
- Precision: Reliability of positive alarms (minimizing false alarms).
- Recall: Ability to detect actual emergencies (minimizing missed cries for help).
- F1 Score: Balanced harmonic mean of Precision and Recall.
- Confusion Matrix: 2x2 grid breaking down True/False Positives and Negatives.

Usage:
    Called by train.py during the training pipeline.
"""

import os
import numpy as np
import matplotlib
# Matplotlib 'Agg' backend allows generating and saving plots in non-GUI / headless environments
# (e.g. cloud servers, automated scripts) without throwing GUI display errors.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Project root directory (parent of src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on test data.
    
    Computes accuracy, precision, recall, F1-score, confusion matrix,
    and classification report. Saves confusion matrix plot and report.
    
    Metric Definitions for Students:
    --------------------------------
    1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
       - The overall fraction of correct predictions across all samples.
       - Caution: Misleading when classes are imbalanced!
    
    2. Precision = TP / (TP + FP)
       - "When the model says a tweet is a disaster, how often is it right?"
       - High precision means very few FALSE ALARMS (low FP).
       - Critical because false alarms waste emergency rescue teams and resources.
    
    3. Recall (Sensitivity / True Positive Rate) = TP / (TP + FN)
       - "Out of all actual disaster tweets, how many did the model catch?"
       - High recall means very few MISSED EMERGENCIES (low FN).
       - Critical in disaster response because a missed report could mean lives lost.
    
    4. F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
       - Harmonic mean of Precision and Recall.
       - Unlike the arithmetic mean, if either precision or recall is close to 0,
         the F1 score plummets. This forces the model to balance both metrics.
    
    Args:
        model: Trained sklearn model.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name (str): Name of the model for labeling.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    # Make predictions on unseen test features:
    # y_pred will contain predicted binary classes (0 or 1) for each test sample.
    y_pred = model.predict(X_test)
    
    # Calculate metrics:
    # WHY zero_division=0?
    # If a model predicts 0 positive instances (TP = 0, FP = 0), precision has denominator 0: (0 / 0).
    # In math this is undefined; in Python it triggers ZeroDivisionError or noisy scikit-learn warnings.
    # Setting zero_division=0 instructs scikit-learn to safely return 0.0 instead of crashing.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix and classification report:
    # A Confusion Matrix is a 2x2 table that cross-tabulates True Labels vs Predicted Labels:
    #                 Predicted 0 (Not Informative)    Predicted 1 (Informative)
    # True 0 (Not):         True Negative (TN)            False Positive (FP) - False Alarm
    # True 1 (Info):        False Negative (FN) - Missed  True Positive (TP) - Correct Hit
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Not Informative', 'Informative']
    )
    
    # Save confusion matrix plot
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    cm_path = os.path.join(figures_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm, model_name, cm_path)
    
    # Save classification report to text file
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    report_path = os.path.join(reports_dir, f"classification_report_{model_name}.txt")
    save_classification_report(report, model_name, report_path)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n{report}")
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def plot_confusion_matrix(cm, model_name, save_path):
    """
    Generate and save a confusion matrix heatmap.
    
    Visualization Process:
    ----------------------
    1. Seaborn's heatmap renders the 2D confusion matrix array.
    2. 'annot=True' prints the exact integer sample count in each cell.
    3. 'fmt="d"' formats the values as raw decimal integers (e.g. 520 instead of 5.2e+02).
    4. 'cmap="Blues"' applies an intuitive monochromatic blue shading where darker cells
       indicate higher frequencies (ideally along the main diagonal: TN and TP).
    5. Clean axis labels ('True Label' on Y-axis, 'Predicted Label' on X-axis) ensure
       anyone inspecting the plot can immediately identify false alarms vs missed emergencies.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        model_name (str): Name for the plot title.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Not Informative', 'Informative'],
        yticklabels=['Not Informative', 'Informative'],
        annot_kws={'size': 14}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()  # Automatically adjusts subplot parameters to give specified padding
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # bbox_inches='tight' clips whitespace
    plt.close()  # Closes current figure window to free memory
    print(f"  💾 Saved confusion matrix: {os.path.basename(save_path)}")


def save_classification_report(report, model_name, save_path):
    """
    Save classification report as a text file.
    
    The classification report breaks down precision, recall, and F1-score
    for both individual classes ('Not Informative' and 'Informative'),
    plus Macro Average (unweighted mean) and Weighted Average (mean weighted by class frequency).
    
    Args:
        report (str): Classification report string.
        model_name (str): Name of the model.
        save_path (str): File path to save.
    """
    with open(save_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)


def plot_model_comparison(results_dict, save_path):
    """
    Create a grouped bar chart comparing all models across metrics.
    
    What the Comparison Chart Shows:
    ---------------------------------
    This chart visually contrasts the performance of all 4 candidate models:
    - Logistic Regression
    - Naive Bayes
    - Linear SVC
    - Random Forest
    
    For each model, 4 vertical bars are displayed side-by-side representing:
    - Accuracy (Dark Blue): Overall correctness
    - Precision (Red): False alarm prevention
    - Recall (Amber): Disaster detection sensitivity
    - F1 Score (Green): Balanced evaluation criterion
    
    Numeric annotations on top of each bar allow rapid side-by-side comparison,
    making it visually apparent why the top model was selected.
    
    Args:
        results_dict (dict): Dictionary of {model_name: {metric: value}}.
        save_path (str): File path to save the chart.
    """
    model_names = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Calculate bar positions along the x-axis:
    # np.arange(len(model_names)) creates base tick indices [0, 1, 2, 3] for each model.
    # width = 0.18 allocates bar width so 4 bars fit neatly side-by-side around each center tick.
    x = np.arange(len(model_names))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Accessible color palette:
    # Blue = Accuracy, Red = Precision, Amber = Recall, Green = F1
    colors = ['#1e3a8a', '#dc2626', '#f59e0b', '#16a34a']
    
    # Plot each metric's bar group with calculated horizontal offsets:
    # Offsets: (0 - 1.5)*0.18 = -0.27, (1 - 1.5)*0.18 = -0.09, +0.09, +0.27
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=colors[i])
        
        # Add numerical value labels directly on top of each bar for easy reading
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Clean up model names for display (e.g., 'Linear_SVC' -> 'Linear SVC')
    display_names = [name.replace('_', ' ') for name in model_names]
    ax.set_xticklabels(display_names, rotation=20, ha='right', fontsize=10)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)  # Range 0.0 to 1.1 ensures top value labels don't get clipped
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved model comparison chart: {os.path.basename(save_path)}")
