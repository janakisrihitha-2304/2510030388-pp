import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


print("=" * 60)
print("LAB 10 - XGBOOST AND LIGHTGBM")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "raw_placement_data.csv"
)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))


# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

TARGET = "placement_status"

# Remove salary to avoid target leakage
if "salary_package_lpa" in df.columns:
    df = df.drop(columns=["salary_package_lpa"])

X = df.drop(columns=[TARGET])
y = df[TARGET]


# Encode categorical features
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column].astype(str))


# Encode target
if y.dtype == "object":
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))


print("Number of features:", X.shape[1])


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {

    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    ),

    "lightgbm": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
}


# ------------------------------------------------------------
# TRAIN AND EVALUATE
# ------------------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Training time:",
          round(training_time, 4),
          "seconds")

    print("Accuracy:",
          round(accuracy, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "training_time_seconds": training_time
    })


# ------------------------------------------------------------
# SAVE COMPARISON
# ------------------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_path = os.path.join(
    REPORTS_DIR,
    "xgb_lightgbm_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\nSaved:", comparison_path)


# ------------------------------------------------------------
# SHAP ANALYSIS
# ------------------------------------------------------------

def shap_report(name, model, X_data):

    print("\n" + "-" * 60)
    print("SHAP ANALYSIS:", name.upper())
    print("-" * 60)

    # Use sample for faster SHAP analysis
    sample_size = min(2000, len(X_data))

    X_shap = X_data.sample(
        n=sample_size,
        random_state=42
    )

    print("SHAP samples:", len(X_shap))

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[:, :, -1]

    # Feature importance
    importance = np.mean(
        np.abs(values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_absolute_shap": importance
    })

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance_path = os.path.join(
        REPORTS_DIR,
        f"{name}_shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print("Saved:", importance_path)

    print("\nTop SHAP Features:")
    print(importance_df.head(10).to_string(index=False))


    # SHAP summary plot
    plt.figure()

    shap.summary_plot(
        values,
        X_shap,
        show=False
    )

    plt.tight_layout()

    summary_path = os.path.join(
        FIGURES_DIR,
        f"{name}_shap_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", summary_path)


    # SHAP dependence plot
    top_feature = importance_df.iloc[0]["feature"]

    plt.figure()

    shap.dependence_plot(
        top_feature,
        values,
        X_shap,
        show=False
    )

    plt.tight_layout()

    dependence_path = os.path.join(
        FIGURES_DIR,
        f"{name}_shap_dependence.png"
    )

    plt.savefig(
        dependence_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", dependence_path)


# Run SHAP
for name, model in models.items():

    shap_report(
        name,
        model,
        X_test
    )


print("\n" + "=" * 60)
print("LAB 10 COMPLETED SUCCESSFULLY!")
print("=" * 60)