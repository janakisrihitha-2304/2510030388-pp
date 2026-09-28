import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Import validated ingestion function from Lab 2
from src.data.ingest import load_and_validate_data


# ============================================================
# 1. COST FUNCTION
# ============================================================

def compute_cost(X, y, w):
    """
    Computes the Mean Squared Error cost function
    divided by 2.
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


# ============================================================
# 2. GRADIENT DESCENT
# ============================================================

def gradient_descent(X, y, w, alpha, num_iters):
    """
    Implements Gradient Descent optimization from scratch.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # Calculate predictions
        predictions = np.dot(X, w)

        # Calculate errors
        errors = predictions - y

        # Calculate gradient
        gradient = (1 / m) * np.dot(
            X.T,
            errors
        )

        # Update weights
        w = w - alpha * gradient

        # Calculate current cost
        cost = compute_cost(
            X,
            y,
            w
        )

        # Store cost
        cost_history.append(cost)

    return w, cost_history


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def run_gradient_descent_experiment():

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION USING GRADIENT DESCENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Select Feature and Target
    # --------------------------------------------------------

    feature_cols = [
        "cgpa"
    ]

    target_col = "salary_package_lpa"

    # Remove rows with missing values
    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[
        feature_cols
    ].values

    y_raw = df_clean[
        target_col
    ].values.reshape(-1, 1)

    print(
        f"\nTotal valid data points: "
        f"{len(df_clean)}"
    )

    # --------------------------------------------------------
    # 80/20 Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------

    scaler_x = StandardScaler()

    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    X_test_scaled = scaler_x.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )

    # --------------------------------------------------------
    # Add Intercept Column
    # --------------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones(
                (X_train_scaled.shape[0], 1)
            ),
            X_train_scaled
        ]
    )

    # --------------------------------------------------------
    # Learning Rate Experiment
    # --------------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    results = {}

    # --------------------------------------------------------
    # Plot Cost History
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    for alpha in learning_rates:

        # Initialize weights to zero
        w_init = np.zeros(
            (
                X_train_design.shape[1],
                1
            )
        )

        # Run Gradient Descent
        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        # Store results
        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        # Plot cost history
        plt.plot(
            cost_history,
            label=f"Alpha (α) = {alpha}"
        )

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost Function E(w) - MSE"
    )

    plt.title(
        "Effect of Different Learning Rates "
        "on Gradient Descent Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\n-> Saved learning-rate cost graph to:"
    )

    print(
        output_path
    )

    # --------------------------------------------------------
    # Select Best Learning Rate
    # --------------------------------------------------------

    best_alpha = 0.1

    final_w = results[
        best_alpha
    ]["weights"]

    print("\n" + "=" * 60)

    print(
        f"--- CUSTOM GRADIENT DESCENT "
        f"PARAMETERS (α = {best_alpha}) ---"
    )

    print("=" * 60)

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.4f}"
    )

    print(
        f"Coefficient (w1): "
        f"{final_w[1, 0]:.4f}"
    )

    # --------------------------------------------------------
    # Compare with Scikit-learn
    # --------------------------------------------------------

    sklearn_model = SklearnLinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n" + "=" * 60)

    print(
        "--- SCIKIT-LEARN COMPARISON ---"
    )

    print("=" * 60)

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.4f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.4f}"
    )

    print("\n" + "=" * 60)
    print("LAB 5 EXECUTION COMPLETE")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_gradient_descent_experiment()