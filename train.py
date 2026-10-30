import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

def run_training_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("PHISHGUARD AI: MACHINE LEARNING TRAINING & EVALUATION PIPELINE")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. LOAD & CLEAN DATASETS
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading and Cleaning Datasets...")
    url_path = "dataset/urls_dataset.csv" if os.path.exists("dataset/urls_dataset.csv") else "dataset/phishing_urls.csv"
    email_path = "dataset/emails_dataset.csv" if os.path.exists("dataset/emails_dataset.csv") else "dataset/phishing_emails.csv"
    url_df = pd.read_csv(url_path)
    email_df = pd.read_csv(email_path)

    print(f"  * Raw URL Dataset: {len(url_df)} rows")
    print(f"  * Raw Email Dataset: {len(email_df)} rows")

    # Drop nulls and duplicates
    url_df = url_df.dropna().drop_duplicates(subset=["url"]).reset_index(drop=True)
    email_df = email_df.dropna().drop_duplicates(subset=["message"]).reset_index(drop=True)

    print(f"  * Cleaned URL Dataset: {len(url_df)} rows (Phishing: {url_df['phishing'].sum()}, Legit: {len(url_df)-url_df['phishing'].sum()})")
    print(f"  * Cleaned Email Dataset: {len(email_df)} rows (Phishing: {email_df['phishing'].sum()}, Legit: {len(email_df)-email_df['phishing'].sum()})")

    # -------------------------------------------------------------
    # 2. URL MODEL COMPARISON (4 Regularized Models to Prevent Overfitting)
    # -------------------------------------------------------------
    print("\n[Step 2/5] Training & Comparing URL Detection Models...")
    feature_cols = [c for c in url_df.columns if c not in ["url", "phishing"]]
    X_url = url_df[feature_cols]
    y_url = url_df["phishing"]

    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X_url, y_url, test_size=0.20, random_state=42, stratify=y_url
    )

    url_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=16, min_samples_leaf=8, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, min_samples_split=10, random_state=42
        )
    }

    url_results = []
    url_trained = {}
    url_test_preds = {}
    url_test_probs = {}

    for name, clf in url_models.items():
        print(f"  -> Fitting URL Model: {name}...")
        clf.fit(X_train_u, y_train_u)
        url_trained[name] = clf

        preds = clf.predict(X_test_u)
        probs = clf.predict_proba(X_test_u)[:, 1]

        acc = accuracy_score(y_test_u, preds)
        prec = precision_score(y_test_u, preds, zero_division=0)
        rec = recall_score(y_test_u, preds, zero_division=0)
        f1 = f1_score(y_test_u, preds, zero_division=0)
        auc = roc_auc_score(y_test_u, probs)

        url_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })
        url_test_preds[name] = preds
        url_test_probs[name] = probs

    url_res_df = pd.DataFrame(url_results)
    print("\n--- URL Models Performance Summary ---")
    print(url_res_df.to_string(index=False))

    # -------------------------------------------------------------
    # 3. TEXT / NLP MODEL COMPARISON (4 Regularized Models to Prevent Overfitting)
    # -------------------------------------------------------------
    print("\n[Step 3/5] Training & Comparing Text/Message NLP Models...")
    X_text = email_df["message"]
    y_text = email_df["phishing"]

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_text, y_text, test_size=0.20, random_state=42, stratify=y_text
    )

    text_models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english", min_df=2)),
            ("classifier", MultinomialNB(alpha=1.0))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english", min_df=2)),
            ("classifier", LogisticRegression(C=0.8, max_iter=1000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english", min_df=2)),
            ("classifier", CalibratedClassifierCV(LinearSVC(C=0.6, random_state=42, max_iter=2000), cv=3))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english", min_df=2)),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
        ])
    }

    text_results = []
    text_trained = {}
    text_test_preds = {}
    text_test_probs = {}

    for name, clf in text_models.items():
        print(f"  -> Fitting Text Model: {name}...")
        clf.fit(X_train_t, y_train_t)
        text_trained[name] = clf

        preds = clf.predict(X_test_t)
        probs = clf.predict_proba(X_test_t)[:, 1]

        acc = accuracy_score(y_test_t, preds)
        prec = precision_score(y_test_t, preds, zero_division=0)
        rec = recall_score(y_test_t, preds, zero_division=0)
        f1 = f1_score(y_test_t, preds, zero_division=0)
        auc = roc_auc_score(y_test_t, probs)

        text_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })
        text_test_preds[name] = preds
        text_test_probs[name] = probs

    text_res_df = pd.DataFrame(text_results)
    print("\n--- Text NLP Models Performance Summary ---")
    print(text_res_df.to_string(index=False))

    # -------------------------------------------------------------
    # 4. SELECT AND SAVE BEST PRODUCTION PIPELINES
    # -------------------------------------------------------------
    print("\n[Step 4/5] Selecting Best Models & Exporting...")
    # Best model by F1 Score
    best_url_idx = url_res_df["F1"].idxmax()
    best_url_name = url_res_df.loc[best_url_idx, "Model"]
    best_url_clf = url_trained[best_url_name]

    best_text_idx = text_res_df["F1"].idxmax()
    best_text_name = text_res_df.loc[best_text_idx, "Model"]
    best_text_clf = text_trained[best_text_name]

    print(f"  [*] Best URL Model Selected: {best_url_name} (F1: {url_res_df.loc[best_url_idx, 'F1']})")
    print(f"  [*] Best Text Model Selected: {best_text_name} (F1: {text_res_df.loc[best_text_idx, 'F1']})")

    joblib.dump(best_url_clf, "models/url_model.pkl")
    joblib.dump(best_text_clf, "models/text_model.pkl")

    meta = {
        "best_url_model": best_url_name,
        "best_text_model": best_text_name,
        "best_url_f1": float(url_res_df.loc[best_url_idx, "F1"]),
        "best_text_f1": float(text_res_df.loc[best_text_idx, "F1"]),
        "url_features": feature_cols,
        "url_dataset_rows": len(url_df),
        "email_dataset_rows": len(email_df),
        "url_models_benchmark": url_results,
        "text_models_benchmark": text_results
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save CSV and text reports
    url_res_df.to_csv("results/url_model_comparison.csv", index=False)
    text_res_df.to_csv("results/text_model_comparison.csv", index=False)

    with open("results/report_metrics.txt", "w") as f:
        f.write("=== URL CLASSIFICATION MODEL COMPARISON ===\n")
        f.write(url_res_df.to_string(index=False))
        f.write("\n\n=== TEXT NLP CLASSIFICATION MODEL COMPARISON ===\n")
        f.write(text_res_df.to_string(index=False))
        f.write(f"\n\nSelected Best URL Model: {best_url_name}\n")
        f.write(f"Selected Best Text Model: {best_text_name}\n")

    # -------------------------------------------------------------
    # 5. GENERATE PUBLICATION-QUALITY FIGURES FOR REPORT & UI
    # -------------------------------------------------------------
    print("\n[Step 5/5] Generating Visual Figures (Bar Charts, Confusion Matrices, ROC Curves)...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: URL Models Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(url_res_df["Model"]))
    width = 0.15
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * width, url_res_df[m], width, label=m)
    ax.set_title("State-of-the-Art URL Phishing Detection Models Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(url_res_df["Model"], fontsize=11)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim(0.70, 1.05)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig("results/url_model_comparison.png")
    plt.close()

    # Figure 2: Text Models Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(text_res_df["Model"]))
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * width, text_res_df[m], width, label=m)
    ax.set_title("State-of-the-Art Text/Email NLP Models Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(text_res_df["Model"], fontsize=11)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim(0.70, 1.05)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig("results/text_model_comparison.png")
    plt.close()

    # Figure 3: ROC Curves (Combined URL + Text)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for name in url_models.keys():
        fpr, tpr, _ = roc_curve(y_test_u, url_test_probs[name])
        auc = roc_auc_score(y_test_u, url_test_probs[name])
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax1.set_title("ROC Curves: URL Classifiers", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    for name in text_models.keys():
        fpr, tpr, _ = roc_curve(y_test_t, text_test_probs[name])
        auc = roc_auc_score(y_test_t, text_test_probs[name])
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax2.set_title("ROC Curves: Text NLP Classifiers", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig("results/roc_curves.png")
    plt.close()

    # Figure 4: Confusion Matrices (Best URL and Best Text)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    cm_u = confusion_matrix(y_test_u, url_test_preds[best_url_name])
    disp_u = ConfusionMatrixDisplay(confusion_matrix=cm_u, display_labels=["Legit", "Phish"])
    disp_u.plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title(f"Confusion Matrix: Best URL Model\n({best_url_name})", fontsize=11, fontweight="bold")

    cm_t = confusion_matrix(y_test_t, text_test_preds[best_text_name])
    disp_t = ConfusionMatrixDisplay(confusion_matrix=cm_t, display_labels=["Legit", "Phish"])
    disp_t.plot(ax=ax2, cmap="Greens", colorbar=False)
    ax2.set_title(f"Confusion Matrix: Best Text Model\n({best_text_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig("results/confusion_matrices.png")
    plt.close()

    # Figure 5: Feature Importance for URL features
    rf_url = url_trained["Random Forest"]
    importances = rf_url.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.barh(range(len(sorted_features)), sorted_importances[::-1], color="#795F4D", align="center")
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=10)
    ax.set_xlabel("Relative Importance Weight", fontsize=11)
    ax.set_title("Random Forest URL Feature Importance Ranking", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig("results/feature_importance.png")
    plt.close()

    print("  [+] Saved results/url_model_comparison.png")
    print("  [+] Saved results/text_model_comparison.png")
    print("  [+] Saved results/roc_curves.png")
    print("  [+] Saved results/confusion_matrices.png")
    print("  [+] Saved results/feature_importance.png")
    print("  [+] Exported models and metrics successfully.")
    print("=" * 65)

if __name__ == "__main__":
    run_training_pipeline()
