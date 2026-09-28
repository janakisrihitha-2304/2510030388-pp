import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.multiclass import OneVsRestClassifier


# ============================================================
# LAB 7: LOGISTIC REGRESSION
# BINARY & MULTICLASS CLASSIFICATION
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "raw_placement_data.csv"
)

df = pd.read_csv(DATA_PATH)


print("=" * 65)
print("LAB 7: LOGISTIC REGRESSION")
print("=" * 65)

print("\nDataset loaded successfully")
print("Total records:", len(df))


# ============================================================
# PART A — BINARY CLASSIFICATION
# ============================================================

print("\n")
print("=" * 65)
print("PART A: BINARY CLASSIFICATION")
print("=" * 65)


# ------------------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------------------

binary_data = df[
    [
        "cgpa",
        "aptitude_score",
        "placement_status"
    ]
].dropna()


X_binary = binary_data[
    [
        "cgpa",
        "aptitude_score"
    ]
]

y_binary_text = binary_data[
    "placement_status"
]


# ------------------------------------------------------------
# 3. ENCODE PLACEMENT STATUS
# ------------------------------------------------------------

placement_encoder = LabelEncoder()

y_binary = placement_encoder.fit_transform(
    y_binary_text
)


print("\nBinary Classes")

for i, label in enumerate(
    placement_encoder.classes_
):
    print(f"{i} = {label}")


# ------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_binary,
    y_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_binary
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# ------------------------------------------------------------

binary_model = LogisticRegression(
    max_iter=1000
)

binary_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 6. PREDICTION
# ------------------------------------------------------------

y_pred = binary_model.predict(
    X_test
)


# ------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------

binary_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nBinary Accuracy:")
print(f"{binary_accuracy:.4f}")


print("\n--- Binary Classification Report ---")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=placement_encoder.classes_
    )
)


# ============================================================
# 8. BINARY DECISION BOUNDARY
# ============================================================

x_min = X_binary["cgpa"].min() - 0.2
x_max = X_binary["cgpa"].max() + 0.2

y_min = X_binary["aptitude_score"].min() - 2
y_max = X_binary["aptitude_score"].max() + 2


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)


grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


Z = binary_model.predict(
    grid
)

Z = Z.reshape(
    xx.shape
)


plt.figure(figsize=(9, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25
)

plt.scatter(
    X_test["cgpa"],
    X_test["aptitude_score"],
    c=y_test,
    edgecolors="black",
    s=25
)

plt.xlabel("CGPA")
plt.ylabel("Aptitude Score")

plt.title(
    "Binary Logistic Regression Decision Boundary"
)

plt.tight_layout()


# ------------------------------------------------------------
# 9. SAVE BINARY GRAPH
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


binary_graph = os.path.join(
    OUTPUT_DIR,
    "lab7_binary_decision_boundary.png"
)

plt.savefig(
    binary_graph,
    dpi=300
)

print("\nBinary decision boundary saved:")
print(binary_graph)

plt.show()


# ============================================================
# PART B — MULTICLASS CLASSIFICATION
# ============================================================

print("\n")
print("=" * 65)
print("PART B: MULTICLASS CLASSIFICATION")
print("=" * 65)


# ------------------------------------------------------------
# 10. SELECT MULTICLASS DATA
# ------------------------------------------------------------

multi_data = df[
    [
        "cgpa",
        "aptitude_score",
        "branch"
    ]
].dropna()


X_multi = multi_data[
    [
        "cgpa",
        "aptitude_score"
    ]
]


# ------------------------------------------------------------
# 11. ENCODE BRANCH
# ------------------------------------------------------------

branch_encoder = LabelEncoder()

y_multi = branch_encoder.fit_transform(
    multi_data["branch"]
)


print("\nBranch Classes")

for i, label in enumerate(
    branch_encoder.classes_
):
    print(f"{i} = {label}")


# ------------------------------------------------------------
# 12. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_multi,
    y_multi,
    test_size=0.20,
    random_state=42,
    stratify=y_multi
)


print("\nTraining samples:", len(X_m_train))
print("Testing samples :", len(X_m_test))


# ============================================================
# 13. MULTINOMIAL LOGISTIC REGRESSION
# ============================================================

multinomial_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

multinomial_model.fit(
    X_m_train,
    y_m_train
)


y_multi_pred = multinomial_model.predict(
    X_m_test
)


# ------------------------------------------------------------
# 14. MULTINOMIAL EVALUATION
# ------------------------------------------------------------

multinomial_accuracy = accuracy_score(
    y_m_test,
    y_multi_pred
)

print("\nMultinomial Accuracy:")
print(f"{multinomial_accuracy:.4f}")


print(
    "\n--- Multinomial Logistic Regression Report ---"
)

print(
    classification_report(
        y_m_test,
        y_multi_pred,
        target_names=branch_encoder.classes_
    )
)


# ============================================================
# 15. ONE-VS-REST LOGISTIC REGRESSION
# ============================================================

ovr_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000
    )
)

ovr_model.fit(
    X_m_train,
    y_m_train
)


y_ovr_pred = ovr_model.predict(
    X_m_test
)


ovr_accuracy = accuracy_score(
    y_m_test,
    y_ovr_pred
)


print("\nOne-vs-Rest Accuracy:")
print(f"{ovr_accuracy:.4f}")


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_m_test,
    y_multi_pred
)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cbar=False,
    xticklabels=branch_encoder.classes_,
    yticklabels=branch_encoder.classes_
)

plt.title(
    "Confusion Matrix - Multiclass Placement Prediction"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()


# ------------------------------------------------------------
# 17. SAVE CONFUSION MATRIX
# ------------------------------------------------------------

confusion_graph = os.path.join(
    OUTPUT_DIR,
    "lab7_confusion_matrix.png"
)

plt.savefig(
    confusion_graph,
    dpi=300
)

print("\nConfusion matrix saved:")
print(confusion_graph)

plt.show()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("LAB 7 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"\nBinary Accuracy      : {binary_accuracy:.4f}"
)

print(
    f"Multinomial Accuracy : {multinomial_accuracy:.4f}"
)

print(
    f"OvR Accuracy         : {ovr_accuracy:.4f}"
)

print("\nOutput files:")
print("1.", binary_graph)
print("2.", confusion_graph)

print("\n" + "=" * 65)