# Disaster Intelligence Engine

An educational machine-learning project that classifies disaster-related social media text as **informative** or **not informative**. It includes a training pipeline, model evaluation reports, and an interactive Streamlit app.

> **Important:** This project estimates whether a post contains potentially useful information. It does not verify whether a post is true, locate an incident, or provide emergency advice. Do not use it to make or automate real-world emergency-response decisions.

## What it does

- Cleans text and represents it with TF-IDF features.
- Compares Logistic Regression, Multinomial Naive Bayes, calibrated Linear SVC, and Random Forest classifiers.
- Selects the model with the best F1 score on a held-out test split.
- Reports accuracy, precision, recall, F1, and confusion matrices.
- Provides a Streamlit interface with an analyzer, dashboard, model-performance view, and project information.

## Requirements

- Python 3.11 or newer (required by the current scikit-learn and pandas releases)
- Windows, macOS, or Linux
- Internet access to install the Python packages

## Quick start

Open a terminal in the repository folder—the folder containing `requirements.txt`, `app`, and `src`.

### Windows PowerShell

These commands use the virtual environment's Python directly. Activation is optional, so this method also works if PowerShell blocks activation scripts.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

If `python` is not recognized when creating the environment, try `py -3 -m venv .venv`. If neither command works, install Python and reopen the terminal.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Streamlit prints a local URL, normally <http://localhost:8501>. Open it in your browser and leave the terminal running while you use the app. Stop the server with **Ctrl+C**.

## Model setup

The app needs trained files in `models/`. A fresh clone may not contain these generated model files.

### Try the pipeline without the real dataset

Run the synthetic end-to-end test. It creates sample data, trains the pipeline, checks predictions, and writes the model files needed by the app.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe tests\test_pipeline.py
```

**macOS or Linux, with the environment activated:**

```bash
python tests/test_pipeline.py
```

The generated model is for testing the software only; its metrics and predictions are not representative of real disaster reports. The test leaves `data/raw/test_synthetic_data.tsv` in place. Remove that file before training on real data so the synthetic examples are not included in your training dataset.

### Train with CrisisMMD data

1. Obtain the CrisisMMD dataset from the [official CrisisNLP page](https://crisisnlp.qcri.org/crisismmd) and follow its access and usage terms.
2. Extract the annotation TSV files into `data/raw/`. Subfolders are supported.
3. Ensure the files include a recognizable text column (for example, `tweet_text` or `text`) and a label column (for example, `label` or `informativeness_label`). This project uses rows labeled `informative` or `not_informative`; other labels are discarded.
4. From the repository root, run the training script.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe src\train.py
```

**macOS or Linux, with the environment activated:**

```bash
python src/train.py
```

Training combines the TSV files, preprocesses the text, makes a stratified 80/20 train/test split, compares four classifiers, and saves the best model and evaluation outputs. When training finishes, start the app using the command in [Quick start](#quick-start).

The dashboard reads its counts, event breakdown, and model scores from `models/model_results.json`. Each new training run records its completion time and the TSV filenames used; the dashboard warns when viewing older results without this provenance or when synthetic test data was used. Use **Refresh dashboard data** after updating the results file.

## Generated outputs

- `models/disaster_model.pkl` — selected trained classifier
- `models/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `models/model_results.json` — dataset statistics and evaluation metrics
- `data/processed/` — combined and preprocessed datasets
- `reports/figures/` — model comparison and confusion-matrix charts
- `reports/` — text classification reports

Raw TSV/CSV files placed directly in `data/raw/`, processed CSV files, and other generated model `.pkl` files are excluded from Git. The two model files listed above are exceptions: they are required at runtime and should be committed with the app so a deployment host can load them. Keep backups of anything you need; generated artifacts can be recreated by running the training pipeline.

## Deploy to Streamlit Community Cloud

The deployment host checks out the files in your GitHub repository; it cannot access model files that exist only on your computer. Before deploying, make sure these files are included in your commit:

- `app/app.py`
- `requirements.txt`
- `models/disaster_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/model_results.json` (for the dashboard and model-performance pages)

From the repository root, add and push the app and model artifacts:

```bash
git add .gitignore README.md app/app.py requirements.txt models/disaster_model.pkl models/tfidf_vectorizer.pkl models/model_results.json
git commit -m "Prepare app for Streamlit deployment"
git push
```

In Streamlit Community Cloud, create or update the app using the GitHub repository and branch that contain those files. Set **Main file path** to `app/app.py`. After pushing model changes to an existing deployment, allow it to redeploy or use the app's **Reboot** action.

Train and deploy with the same Python package versions when possible. Scikit-learn models saved with `joblib` may not load reliably across different scikit-learn versions. The included `requirements.txt` currently does not pin package versions.

## Project layout

```text
app/app.py                  Streamlit application
data/raw/                   Input TSV files
data/processed/             Generated datasets
models/                     Generated model and metrics
notebooks/01_eda.py         Exploratory data analysis
reports/                    Evaluation reports and figures
src/data_loader.py          Dataset loading and label filtering
src/preprocessing.py        Text cleaning and label encoding
src/features.py             TF-IDF feature extraction
src/train.py                Training and model selection
src/evaluate.py             Metrics and visualizations
src/predict.py              Model loading and inference
tests/test_pipeline.py      Synthetic end-to-end test
requirements.txt            Python dependencies
```

## Troubleshooting

- **`pip` is not recognized:** Use `python -m pip ...` or the environment-specific Python command shown above; avoid calling `pip` directly.
- **`python` is not recognized:** On Windows, try `py -3`. Otherwise install Python, ensure it is available to your terminal, and reopen the terminal.
- **PowerShell says script execution is disabled:** Activation is not required. Run commands using `.\.venv\Scripts\python.exe` as shown above.
- **The deployed app says the model is missing:** Confirm `models/disaster_model.pkl` and `models/tfidf_vectorizer.pkl` are committed and pushed to the deployed branch; `.gitignore` allows these two required files.
- **Training reports no TSV files:** Check that the extracted annotation files are `.tsv` files under `data/raw/` (including its subfolders).
- **Training reports no usable labels:** Check that a recognized label column contains `informative` and/or `not_informative` values. Other label categories are not used for this binary task.

## Limitations

- The classifier predicts the dataset's informativeness label, not truth, urgency, severity, or authenticity.
- Results depend on the training data and may not generalize to different events, platforms, languages, or populations.
- A confidence score is a model estimate, not a guarantee that a prediction is correct.
- Human review and independent verification are essential before acting on any report.

## Data and licensing

CrisisMMD is maintained by QCRI. Refer to the [official dataset page](https://crisisnlp.qcri.org/crisismmd) for dataset access, citation, and usage terms.

This repository does not currently include a `LICENSE` file. Do not assume that the project code is licensed for reuse until its owner adds one.
