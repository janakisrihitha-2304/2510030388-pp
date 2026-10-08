from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"

OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Features and target
# ============================================================

F = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

T = "placement_status"


# ============================================================
# Load and preprocess data
# ============================================================

def load():

    df = pd.read_csv(DATA)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Select required columns
    df = df[F + [T]].dropna()

    # Convert college tier to numeric
    # Example: Tier 1 -> 1
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):

        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[F],
        columns=["branch"],
        dtype=float
    )

    # Convert target to numeric
    y = df[T]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError(
                "placement_status must be binary"
            )

        y = y.astype(str).map(
            {
                labels[0]: 0,
                labels[1]: 1
            }
        )

    return X, y


# ============================================================
# Evaluation function
# ============================================================

def score(model, X, y):

    prediction = model.predict(X)

    return [
        round(accuracy_score(y, prediction), 4),
        round(
            precision_score(
                y,
                prediction,
                zero_division=0
            ),
            4
        ),
        round(
            recall_score(
                y,
                prediction,
                zero_division=0
            ),
            4
        ),
        round(
            f1_score(
                y,
                prediction,
                zero_division=0
            ),
            4
        )
    ]


# ============================================================
# Main experiment
# ============================================================

def run():

    print("=" * 60)
    print("LAB 9 - RANDOM FOREST CLASSIFICATION")
    print("=" * 60)

    # Load data
    X, y = load()

    print("\nDataset loaded successfully.")
    print("Number of records:", len(X))
    print("Number of features:", X.shape[1])

    # Train-test split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training samples:", len(Xtr))
    print("Testing samples :", len(Xte))


    # ========================================================
    # 1. Decision Tree
    # ========================================================

    print("\n" + "=" * 60)
    print("1. DECISION TREE")
    print("=" * 60)

    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(Xtr, ytr)

    dt_score = score(
        dt,
        Xte,
        yte
    )

    print(
        "Decision Tree "
        "[accuracy, precision, recall, F1]:",
        dt_score
    )


    # ========================================================
    # 2. Random Forest
    # ========================================================

    print("\n" + "=" * 60)
    print("2. RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(Xtr, ytr)

    rf_score = score(
        rf,
        Xte,
        yte
    )

    print(
        "Random Forest "
        "[accuracy, precision, recall, F1]:",
        rf_score
    )

    print(
        "OOB score:",
        round(rf.oob_score_, 4)
    )

    print(
        "OOB error:",
        round(1 - rf.oob_score_, 4)
    )


    # ========================================================
    # 3. Effect of Number of Trees
    # ========================================================

    print("\n" + "=" * 60)
    print("3. EFFECT OF NUMBER OF TREES")
    print("=" * 60)

    counts = [10, 25, 50, 100, 200]

    oob = []
    acc = []

    for n in counts:

        print("Training Random Forest with", n, "trees...")

        model = RandomForestClassifier(
            n_estimators=n,
            max_features="sqrt",
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        oob_error = 1 - model.oob_score_

        test_accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        oob.append(oob_error)
        acc.append(test_accuracy)

        print(
            "Trees:",
            n,
            "| OOB Error:",
            round(oob_error, 4),
            "| Test Accuracy:",
            round(test_accuracy, 4)
        )


    # Plot number of trees

    plt.figure(figsize=(9, 6))

    plt.plot(
        counts,
        oob,
        marker="o",
        label="OOB Error"
    )

    plt.plot(
        counts,
        acc,
        marker="o",
        label="Test Accuracy"
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Value")

    plt.title(
        "Effect of Number of Trees - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_number_of_trees.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved:",
        OUT / "random_forest_number_of_trees.png"
    )


    # ========================================================
    # 4. Effect of Feature Subsampling
    # ========================================================

    print("\n" + "=" * 60)
    print("4. FEATURE SUBSAMPLING")
    print("=" * 60)

    options = [
        "sqrt",
        "log2",
        None
    ]

    labels = [
        "sqrt",
        "log2",
        "all features"
    ]

    values = []

    for option in options:

        model = RandomForestClassifier(
            n_estimators=100,
            max_features=option,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        values.append(accuracy)

        print(
            labels[len(values) - 1],
            ":",
            round(accuracy, 4)
        )


    # Plot feature subsampling

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Test Accuracy")

    plt.title(
        "Feature Subsampling - Placement Prediction"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_feature_subsampling.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved:",
        OUT / "random_forest_feature_subsampling.png"
    )


    # ========================================================
    # Final message
    # ========================================================

    print("\n" + "=" * 60)
    print("LAB 9 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ============================================================
# Run program
# ============================================================

if __name__ == "__main__":
    run()