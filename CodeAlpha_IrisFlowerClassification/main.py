"""
===============================================================================
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
===============================================================================
Author: CodeAlpha Intern
Dataset Source: Kaggle (saurabh00007/iriscsv)
URL: https://www.kaggle.com/datasets/saurabh00007/iriscsv

Description:
    An end-to-end Machine Learning pipeline using the verified CodeAlpha Kaggle
    Iris dataset (Iris.csv) to classify iris flowers into three distinct species:
    - Iris-setosa
    - Iris-versicolor
    - Iris-virginica
    
Pipeline Steps:
    1. Environment & Plotting Setup
    2. Data Loading & Comprehensive Kaggle Dataset Inspection (9-step check)
    3. 'Id' Column Analysis & Preprocessing
    4. Exploratory Data Analysis (EDA) & Publication-Quality Visualizations
    5. Stratified Train-Test Split (80/20)
    6. Primary Model Training (K-Nearest Neighbors - KNN)
    7. Benchmark Model Training (Decision Tree Classifier)
    8. Comprehensive Model Evaluation & Comparison
    9. Real-World Inference on Unseen Sample Flower Measurements
===============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure clean UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup_environment():
    """Ensure output directories exist and configure aesthetic plotting parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Configure clean, publication-ready seaborn styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    return script_dir, images_dir


def load_and_inspect_data(script_dir):
    """
    Load the Kaggle Iris.csv dataset and perform complete 9-step inspection.
    Primary Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
    """
    print("=" * 80)
    print("STEP 1 & 2: LOADING & INSPECTING KAGGLE IRIS DATASET")
    print("=" * 80)
    
    csv_path = os.path.join(script_dir, "Iris.csv")
    if not os.path.exists(csv_path):
        print(">> Iris.csv not found locally. Attempting automatic download via kagglehub...")
        try:
            import kagglehub
            download_dir = kagglehub.dataset_download("saurabh00007/iriscsv")
            downloaded_csv = os.path.join(download_dir, "Iris.csv")
            import shutil
            shutil.copy2(downloaded_csv, csv_path)
            print(f">> Downloaded and cached: {csv_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load or download 'Iris.csv': {e}\n"
                "Please download from https://www.kaggle.com/datasets/saurabh00007/iriscsv"
            )
            
    df_raw = pd.read_csv(csv_path)
    print(f"Dataset Path   : {os.path.abspath(csv_path)}")
    print(f"Dataset Source : https://www.kaggle.com/datasets/saurabh00007/iriscsv\n")
    
    # [1] First 5 Rows
    print("--- [1] First 5 Rows of Raw CSV ---")
    print(df_raw.head().to_string(index=False))
    
    # [2] Column Names
    print(f"\n--- [2] Raw Column Names ({len(df_raw.columns)} columns) ---")
    print(list(df_raw.columns))
    
    # [3] Shape
    print(f"\n--- [3] Dataset Shape ---")
    print(f"Rows (Samples): {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")
    
    # [4] Data Types
    print("\n--- [4] Data Types ---")
    print(df_raw.dtypes)
    
    # [5] Missing Values
    print("\n--- [5] Missing Values ---")
    missing = df_raw.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print(">> Data Quality: 100% complete (0 missing values across all columns).")
        
    # [6] Duplicate Check
    print("\n--- [6] Duplicate Rows Check ---")
    print(f"Exact duplicates (with Id): {df_raw.duplicated().sum()}")
    if "Id" in df_raw.columns:
        print(f"Measurement-only duplicates (excluding Id): {df_raw.drop(columns=['Id']).duplicated().sum()}")
        
    # [7] Target Identification
    print("\n--- [7] Target Column (Species) ---")
    print(df_raw["Species"].value_counts())
    
    # [8] Feature Identification
    raw_features = [c for c in df_raw.columns if c not in ["Id", "Species"]]
    print(f"\n--- [8] Flower Measurement Features ---")
    print(raw_features)
    
    # =========================================================================
    # [9] UNNECESSARY COLUMN REMOVAL: 'Id'
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: UNNECESSARY COLUMN HANDLING ('Id')")
    print("=" * 80)
    print("• Evaluation: The 'Id' column is a sequential index (1 to 150) without biological meaning.")
    print("• Risk: Leaving 'Id' in the feature set introduces severe data leakage / artificial correlation.")
    print("• Decision: Dropping 'Id' to ensure models learn purely from physical flower morphology.\n")
    
    df_clean = df_raw.drop(columns=["Id"]) if "Id" in df_raw.columns else df_raw.copy()
    print(f"Clean Dataset Shape: {df_clean.shape} (4 morphological features + 1 target variable)")
    return df_clean, raw_features


def explore_data(df, features):
    """Print statistical summaries of clean flower measurements."""
    print("\n" + "=" * 80)
    print("STEP 4: SUMMARY STATISTICS OF MORPHOLOGICAL FEATURES")
    print("=" * 80)
    summary_stats = df[features].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(summary_stats.round(3))
    print()


def generate_visualizations(df, features, images_dir):
    """Generate and save clean, high-resolution EDA visual plots."""
    print("=" * 80)
    print("STEP 5: GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 80)
    
    palette = {
        "Iris-setosa": "#1f77b4",
        "Iris-versicolor": "#ff7f0e",
        "Iris-virginica": "#2ca02c"
    }
    
    # 1. Species Distribution Count Plot
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(data=df, x="Species", palette=palette, hue="Species", legend=False)
    plt.title("Species Distribution in Kaggle Iris Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Iris Species Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Sample Count", fontsize=11, fontweight="semibold")
    plt.ylim(0, 60)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                    ha='center', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(images_dir, "01_species_distribution.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Sepal Length vs Sepal Width Scatter Plot
    plt.figure(figsize=(8, 5.5))
    sns.scatterplot(
        data=df,
        x="SepalLengthCm",
        y="SepalWidthCm",
        hue="Species",
        style="Species",
        palette=palette,
        s=80,
        alpha=0.9
    )
    plt.title("Sepal Length vs. Sepal Width by Iris Species", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sepal Length (cm)", fontsize=11, fontweight="semibold")
    plt.ylabel("Sepal Width (cm)", fontsize=11, fontweight="semibold")
    plt.legend(title="Species", title_fontsize='10', loc="upper right")
    plt.tight_layout()
    p2 = os.path.join(images_dir, "02_sepal_length_vs_width.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Petal Length vs Petal Width Scatter Plot
    plt.figure(figsize=(8, 5.5))
    sns.scatterplot(
        data=df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        hue="Species",
        style="Species",
        palette=palette,
        s=80,
        alpha=0.9
    )
    plt.title("Petal Length vs. Petal Width by Iris Species", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Petal Length (cm)", fontsize=11, fontweight="semibold")
    plt.ylabel("Petal Width (cm)", fontsize=11, fontweight="semibold")
    plt.legend(title="Species", title_fontsize='10', loc="upper left")
    plt.tight_layout()
    p3 = os.path.join(images_dir, "03_petal_length_vs_width.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Box Plots Across All 4 Morphological Features
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, col in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x="Species", y=col, palette=palette, hue="Species", legend=False, ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("cm", fontsize=10)
    plt.suptitle("Morphological Feature Distributions Across Species", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    p4 = os.path.join(images_dir, "04_feature_distributions.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}\n")


def train_and_evaluate_models(df, features, images_dir):
    """Train KNN and Decision Tree classifiers and evaluate their performance."""
    print("=" * 80)
    print("STEP 6: DATA SPLITTING (STRATIFIED 80/20)")
    print("=" * 80)
    
    X = df[features]
    y = df["Species"]
    
    # 80% Train, 20% Test with stratification to preserve exact class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Total dataset:  {X.shape[0]} samples")
    print(f"Training set:   {X_train.shape[0]} samples (80%)")
    print(f"Testing set:    {X_test.shape[0]} samples (20%)")
    print(f"Feature set:    {features}\n")

    # -------------------------------------------------------------------------
    # MODEL 1: K-Nearest Neighbors (KNN) - Primary Model
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: TRAINING PRIMARY CLASSIFIER (K-NEAREST NEIGHBORS - KNN)")
    print("=" * 80)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    classes = sorted(y.unique())
    
    print(f"KNN Test Accuracy: {acc_knn * 100:.2f}%\n")
    print("--- Classification Report (KNN) ---")
    print(classification_report(y_test, y_pred_knn, target_names=classes))
    
    print("--- Confusion Matrix (KNN) ---")
    print(pd.DataFrame(cm_knn, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes]))
    
    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm_knn,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("KNN Confusion Matrix (Test Data)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Species", fontsize=11, fontweight="semibold")
    plt.ylabel("Actual Species", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    p5 = os.path.join(images_dir, "05_knn_confusion_matrix.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"\nSaved: {p5}")

    # -------------------------------------------------------------------------
    # MODEL 2: Decision Tree Classifier - Comparison Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 8: BENCHMARK MODEL TRAINING & COMPARISON (DECISION TREE)")
    print("=" * 80)
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    
    print(f"Decision Tree Test Accuracy: {acc_dt * 100:.2f}%")
    print(f"K-Nearest Neighbors Accuracy: {acc_knn * 100:.2f}%\n")
    
    # Comparison Bar Chart
    plt.figure(figsize=(6, 4))
    models = ["K-Nearest Neighbors", "Decision Tree"]
    accuracies = [acc_knn * 100, acc_dt * 100]
    bars = plt.bar(models, accuracies, color=["#1f77b4", "#2ca02c"], width=0.45)
    plt.title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="semibold")
    plt.ylim(80, 105)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}%",
                     (bar.get_x() + bar.get_width() / 2., height + 1.0),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    p6 = os.path.join(images_dir, "06_model_accuracy_comparison.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved: {p6}\n")

    return knn, classes


def test_sample_predictions(knn, features, classes):
    """Demonstrate inference on unseen flower measurements."""
    print("=" * 80)
    print("STEP 9: REAL-WORLD INFERENCE ON NEW FLOWER MEASUREMENTS")
    print("=" * 80)
    
    samples = [
        {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2, "expected": "Iris-setosa"},
        {"SepalLengthCm": 6.1, "SepalWidthCm": 2.9, "PetalLengthCm": 4.5, "PetalWidthCm": 1.4, "expected": "Iris-versicolor"},
        {"SepalLengthCm": 7.3, "SepalWidthCm": 3.0, "PetalLengthCm": 6.3, "PetalWidthCm": 2.0, "expected": "Iris-virginica"},
    ]
    
    sample_df = pd.DataFrame(samples)[features]
    predictions = knn.predict(sample_df)
    probabilities = knn.predict_proba(sample_df)
    
    for idx, row in sample_df.iterrows():
        pred_sp = predictions[idx]
        probs = probabilities[idx]
        expected = samples[idx]["expected"]
        
        print(f"[Sample Flower #{idx + 1}]")
        print(f"  Measurements : Sepal=[{row['SepalLengthCm']} x {row['SepalWidthCm']}] cm, "
              f"Petal=[{row['PetalLengthCm']} x {row['PetalWidthCm']}] cm")
        print(f"  Expected     : {expected}")
        print(f"  Predicted    : {pred_sp}")
        prob_str = ", ".join([f"{cls}={prob*100:.0f}%" for cls, prob in zip(knn.classes_, probs)])
        print(f"  Probabilities: {prob_str}")
        print(f"  Verification : {'CORRECT' if pred_sp == expected else 'MISMATCH'}\n")

    print("=" * 80)
    print("✔ FULL MACHINE LEARNING PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    script_dir, images_dir = setup_environment()
    clean_df, feature_names = load_and_inspect_data(script_dir)
    explore_data(clean_df, feature_names)
    generate_visualizations(clean_df, feature_names, images_dir)
    trained_knn, target_classes = train_and_evaluate_models(clean_df, feature_names, images_dir)
    test_sample_predictions(trained_knn, feature_names, target_classes)
