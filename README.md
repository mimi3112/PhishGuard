## Submitted by 
   ```
   Soumya Dubey - A5-B1-13
   Harshita Chaudhari - A5-B1-16
   Meenakshi Motwani - A5-B2-19
   ```

# PhishGuard AI — Dual-Pipeline Phishing Risk Analyzer

PhishGuard AI is an educational cybersecurity machine learning project designed to detect phishing attacks across website URLs and email/SMS message contents. It pairs **Lexical Structural Feature Engineering** with **TF-IDF Natural Language Processing** across 8 state-of-the-art classification architectures.

---

## Architecture & Workflow

```
[ Input: Website URL and/or Message Text ]
           │
           ├──► [ URL Lexical Feature Extractor ] ──► [ Random Forest Classifier ] ──┐
           │     (10 structural & lexical features)                                  │
           │                                                                         ├──► [ Ensemble Risk Engine ] ──► Verdict & XAI Reasons
           └──► [ NLP TF-IDF Vectorizer (1-2 N-grams) ] ──► [ Linear SVC Pipeline ] ──┘
                 (Linguistic & urgency analysis)
```

### Models Evaluated & Compared (8 Models):
1. **URL Classifiers (4 State-of-the-art Models):**
   - Logistic Regression (with StandardScaler)
   - Decision Tree Classifier
   - Random Forest Classifier (Selected Production Pipeline)
   - Gradient Boosting Classifier
2. **Text / Message NLP Classifiers (4 State-of-the-art Models):**
   - Multinomial Naive Bayes
   - Logistic Regression (TF-IDF)
   - Linear Support Vector Machine (LinearSVC with Calibrated Probabilities - Selected Production Pipeline)
   - Random Forest Classifier (TF-IDF)

---

## Project Structure

```
PhishGuard/
├── app.py                      # Flask Application with /predict and /api/benchmark
├── model.py                    # Feature extraction, risk calculation & explainability
├── train.py                    # Automated training, evaluation & high-res chart generation
├── test_system.py              # Automated unit and integration test suite
├── requirements.txt            # Python dependencies
├── dataset/
│   ├── phishing_urls.csv       # 5,000 URLs with extracted lexical features & ground truth
│   └── phishing_emails.csv     # 4,000 emails/messages with subjects & ground truth
├── models/
│   ├── url_model.pkl           # Trained Random Forest pipeline
│   ├── text_model.pkl          # Trained Linear SVM NLP pipeline
│   └── metadata.json           # Model benchmarks, features, and parameters
├── results/
│   ├── url_model_comparison.png       # 300 DPI comparative bar chart for URL models
│   ├── text_model_comparison.png      # 300 DPI comparative bar chart for NLP models
│   ├── roc_curves.png                 # Receiver Operating Characteristic (ROC) curves
│   ├── confusion_matrices.png         # Confusion matrix analysis for best models
│   ├── feature_importance.png         # Feature importance ranking for URL features
│   ├── url_model_comparison.csv       # Benchmark tabular metrics for URL models
│   ├── text_model_comparison.csv      # Benchmark tabular metrics for Text models
│   └── report_metrics.txt             # Summary evaluation text report
├── templates/
│   └── index.html              # Modern, responsive UI with Scanner & Benchmark tabs
├── static/
│   ├── style.css               # Security-themed responsive styling
│   └── script.js               # Interactive presets, dynamic charts & async scanner
└── notebooks/
    └── PhishGuard_AI_Colab.ipynb # Interactive Jupyter/Colab notebook
```

---

##  How to Run in Visual Studio Code

### Step 1: Open the Project
Open VS Code, go to **File > Open Folder...**, and select the `PhishGuard` root folder.

### Step 2: Open the Integrated Terminal
Press `Ctrl + ` `~` (backtick) or go to **Terminal > New Terminal**.

### Step 3: Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
py app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 5: Run Automated Tests
```bash
py test_system.py
```

### Step 6: Retrain Models & Regenerate Figures (Optional)
```bash
py train.py
```

---

## Evaluation & Performance Summary

| Model Architecture | Task | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | URL Phishing | **95.60%** | **95.60%** | **95.79%** | **95.70%** | **0.958** |
| Gradient Boosting | URL Phishing | 95.30% | 95.03% | 95.79% | 95.41% | 0.960 |
| Decision Tree | URL Phishing | 95.30% | 95.03% | 95.79% | 95.41% | 0.951 |
| Logistic Regression | URL Phishing | 94.68% | 93.74% | 95.99% | 94.85% | 0.958 |
| **Naive Bayes (Selected)** | Message NLP | **94.87%** | **95.27%** | **96.96%** | **96.11%** | **0.933** |
| Linear SVM | Message NLP | 94.87% | 95.27% | 96.96% | 96.11% | 0.939 |
| Logistic Regression | Message NLP | 94.87% | 95.27% | 96.96% | 96.11% | 0.940 |
| Random Forest | Message NLP | 94.87% | 95.27% | 96.96% | 96.11% | 0.937 |

*All figures are exported at 300 DPI in [`results/`](results/) and can be viewed directly within the web app's **"Model Benchmarks & Metrics"** tab.*
