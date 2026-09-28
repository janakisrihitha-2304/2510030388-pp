import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the ingestion function from Lab 2
from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION - STANDARD LEAST SQUARES")
    print("=" * 60)

    # ============================================================
    # 2. SELECT VALID DATA
    # ============================================================

    # Your actual dataset contains:
    # cgpa
    # communication_skill_score
    # salary_package_lpa

    required_columns = [
        "cgpa",
        "communication_skill_score",
        "salary_package_lpa"
    ]

    # Check that required columns exist
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove rows containing missing values
    df_clean = df.dropna(
        subset=required_columns
    ).copy()

    # ============================================================
    # 3. DEFINE INPUT AND OUTPUT
    # ============================================================

    # L = 2 input features
    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    # M = 1 output/target
    target_col = "salary_package_lpa"

    # Input matrix
    X_raw = df_clean[feature_cols].to_numpy(
        dtype=float
    )

    # Output vector
    y = df_clean[target_col].to_numpy(
        dtype=float
    ).reshape(-1, 1)

    N = X_raw.shape[0]

    print(f"\nNumber of valid data points: {N}")
    print(f"Input dimension L: {X_raw.shape[1]}")
    print(f"Output dimension M: {y.shape[1]}")

    # ============================================================
    # 4. CREATE DESIGN MATRIX
    # ============================================================

    # Add a column of ones for the intercept/bias.
    #
    # X_design =
    #
    # [1  x1  x2]
    # [1  x1  x2]
    # [1  x1  x2]
    # ...

    X_design = np.hstack(
        (
            np.ones((N, 1)),
            X_raw
        )
    )

    # ============================================================
    # 5. STANDARD LEAST SQUARES
    # ============================================================

    # Normal equation:
    #
    # w = (X^T X)^(-1) X^T y
    #
    # Use pseudo-inverse for numerical stability.

    XT_X = X_design.T @ X_design

    XT_y = X_design.T @ y

    w_optimal = np.linalg.pinv(XT_X) @ XT_y

    # ============================================================
    # 6. DISPLAY MODEL PARAMETERS
    # ============================================================

    intercept = w_optimal[0, 0]
    cgpa_coefficient = w_optimal[1, 0]
    communication_coefficient = w_optimal[2, 0]

    print("\n" + "-" * 60)
    print("OPTIMAL MODEL PARAMETERS")
    print("-" * 60)

    print(
        f"Intercept (w0): "
        f"{intercept:.6f}"
    )

    print(
        f"Coefficient for CGPA (w1): "
        f"{cgpa_coefficient:.6f}"
    )

    print(
        f"Coefficient for Communication Skill Score (w2): "
        f"{communication_coefficient:.6f}"
    )

    # ============================================================
    # 7. MAKE PREDICTIONS
    # ============================================================

    y_pred = X_design @ w_optimal

    # ============================================================
    # 8. CALCULATE ERROR
    # ============================================================

    # Lab 4 error:
    #
    # E_w = 0.5 * sum((y_pred - y)^2)

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print("\n" + "-" * 60)
    print("MODEL ERROR")
    print("-" * 60)

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.6f}"
    )

    # ============================================================
    # 9. CREATE 3D REGRESSION PLANE
    # ============================================================

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # ------------------------------------------------------------
    # Actual data points
    # ------------------------------------------------------------

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.5,
        label="Actual Data"
    )

    # ------------------------------------------------------------
    # Create grid for regression plane
    # ------------------------------------------------------------

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    # ------------------------------------------------------------
    # Calculate regression-plane values
    # ------------------------------------------------------------

    y_mesh = (
        intercept
        + cgpa_coefficient * x1_mesh
        + communication_coefficient * x2_mesh
    )

    # ------------------------------------------------------------
    # Plot regression plane
    # ------------------------------------------------------------

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )

    # ============================================================
    # 10. LABEL GRAPH
    # ============================================================

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Communication Skill Score"
    )

    ax.set_zlabel(
        "Salary Package (LPA)"
    )

    ax.set_title(
        "Linear Regression using Standard Least Squares"
    )

    ax.legend()

    plt.tight_layout()

    # ============================================================
    # 11. SAVE GRAPH
    # ============================================================

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"3D regression plot saved to:\n"
        f"{output_path}"
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    train_linear_regression_ls()