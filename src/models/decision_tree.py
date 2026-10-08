from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


def load():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier to numbers
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=float
    )

    # Target variable
    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError("placement_status must be binary")

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


def run():

    X, y = load()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================================
    # 1. GINI VS ENTROPY
    # =========================================

    for name, criterion in [
        ("gini", "gini"),
        ("entropy", "entropy")
    ]:

        model = DecisionTreeClassifier(
            criterion=criterion,
            random_state=42
        )

        model.fit(Xtr, ytr)

        prediction = model.predict(Xte)

        accuracy = accuracy_score(
            yte,
            prediction
        )

        print(
            name,
            "test accuracy:",
            round(accuracy, 4)
        )

        plt.figure(figsize=(18, 10))

        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=[
                str(x)
                for x in sorted(y.unique())
            ],
            filled=True,
            max_depth=4,
            fontsize=7
        )

        plt.title(
            f"Placement Decision Tree - {name.title()}"
        )

        plt.tight_layout()

        plt.savefig(
            OUT / f"decision_tree_{name}.png",
            dpi=150
        )

        plt.close()


    # =========================================
    # 2. COST COMPLEXITY PRUNING
    # =========================================

    print("\nStarting pruning analysis...")

    path = DecisionTreeClassifier(
        random_state=42
    ).cost_complexity_pruning_path(
        Xtr,
        ytr
    )

    # Use a maximum of 10 representative alpha values
    alphas = path.ccp_alphas

    if len(alphas) > 10:

        indices = np.linspace(
            0,
            len(alphas) - 1,
            10,
            dtype=int
        )

        alphas = alphas[indices]

    rows = []

    for alpha in alphas:

        model = DecisionTreeClassifier(
            ccp_alpha=alpha,
            random_state=42
        )

        model.fit(Xtr, ytr)

        train_acc = accuracy_score(
            ytr,
            model.predict(Xtr)
        )

        test_acc = accuracy_score(
            yte,
            model.predict(Xte)
        )

        rows.append([
            alpha,
            train_acc,
            test_acc,
            model.tree_.node_count
        ])

    pr = pd.DataFrame(
        rows,
        columns=[
            "alpha",
            "train_accuracy",
            "test_accuracy",
            "nodes"
        ]
    )

    best = pr.loc[
        pr["test_accuracy"].idxmax()
    ]

    print(
        "Best CCP alpha:",
        best["alpha"]
    )

    print(
        "Pruned test accuracy:",
        best["test_accuracy"]
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        pr["alpha"],
        pr["train_accuracy"],
        marker="o",
        label="Training"
    )

    plt.plot(
        pr["alpha"],
        pr["test_accuracy"],
        marker="o",
        label="Testing"
    )

    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy")

    plt.title(
        "Cost Complexity Pruning - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_ccp.png",
        dpi=150
    )

    plt.close()


    # =========================================
    # 3. EFFECT OF TREE DEPTH
    # =========================================

    print("\nStarting tree depth analysis...")

    depths = range(1, 16)

    train_accuracy = []
    test_accuracy = []

    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(Xtr, ytr)

        train_accuracy.append(
            accuracy_score(
                ytr,
                model.predict(Xtr)
            )
        )

        test_accuracy.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    best_depth = list(depths)[
        int(np.argmax(test_accuracy))
    ]

    print(
        "Best depth:",
        best_depth
    )

    print(
        "Best test accuracy:",
        max(test_accuracy)
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        depths,
        train_accuracy,
        marker="o",
        label="Training"
    )

    plt.plot(
        depths,
        test_accuracy,
        marker="o",
        label="Testing"
    )

    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")

    plt.title(
        "Effect of Tree Depth - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_depth.png",
        dpi=150
    )

    plt.close()

    print("\nLab 8 completed successfully!")


if __name__ == "__main__":
    run()