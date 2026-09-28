import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ============================================================
# LAB 6: REGULARIZATION (RIDGE / L2)
# Polynomial Regression on Placement Dataset
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "raw_placement_data.csv"
)

print("Loading dataset...")
print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 2. SELECT INPUT AND TARGET
# ------------------------------------------------------------

# X = CGPA
# y = Salary Package

required_columns = [
    "cgpa",
    "salary_package_lpa"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' was not found in the dataset.\n"
            f"Available columns are:\n{list(df.columns)}"
        )

# Remove missing values
data = df[required_columns].dropna()

X = data[["cgpa"]].values
y = data["salary_package_lpa"].values


print("\nSelected Features")
print("-----------------")
print("Input feature  : cgpa")
print("Target         : salary_package_lpa")
print("Usable samples :", len(data))


# ------------------------------------------------------------
# 3. SPLIT DATASET 80% / 20%
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData Split")
print("----------")
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ------------------------------------------------------------
# 4. CREATE DEGREE-15 POLYNOMIAL FEATURES
# ------------------------------------------------------------

poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\nPolynomial Transformation")
print("-------------------------")
print("Polynomial degree :", poly_degree)
print("Original features:", X_train.shape[1])
print("New features     :", X_train_poly.shape[1])


# ------------------------------------------------------------
# 5. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_poly
)

X_test_scaled = scaler.transform(
    X_test_poly
)


# ------------------------------------------------------------
# 6. DEFINE REGULARIZATION PARAMETERS
# ------------------------------------------------------------

lambdas = np.logspace(
    -4,
    4,
    200
)

train_errors = []
test_errors = []


# ------------------------------------------------------------
# 7. TRAIN RIDGE MODEL FOR EACH LAMBDA
# ------------------------------------------------------------

print("\nTraining Ridge models...")

for lam in lambdas:

    ridge = Ridge(
        alpha=lam
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    # Training prediction
    y_train_pred = ridge.predict(
        X_train_scaled
    )

    # Testing prediction
    y_test_pred = ridge.predict(
        X_test_scaled
    )

    # Calculate MSE
    train_mse = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    train_errors.append(
        train_mse
    )

    test_errors.append(
        test_mse
    )


# ------------------------------------------------------------
# 8. FIND BEST LAMBDA
# ------------------------------------------------------------

best_index = np.argmin(test_errors)

best_lambda = lambdas[best_index]
best_test_mse = test_errors[best_index]

print("\nBest Regularization Parameter")
print("-----------------------------")
print("Best alpha / lambda :", best_lambda)
print("Minimum test MSE    :", best_test_mse)


# ------------------------------------------------------------
# 9. CREATE OUTPUT DIRECTORY
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


# ------------------------------------------------------------
# 10. PLOT ERROR CURVES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error (Ew)",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error (Ew)",
    linewidth=2,
    linestyle="--"
)

plt.xscale("log")

plt.xlabel(
    "Regularization Parameter (λ / alpha)"
)

plt.ylabel(
    "Mean Squared Error (Ew)"
)

plt.title(
    "Regularization Path: Ridge Regression "
    "Overfitting Control (Degree 15)"
)

plt.legend()

plt.grid(
    True,
    which="both",
    linestyle="--"
)

plt.tight_layout()


# ------------------------------------------------------------
# 11. SAVE GRAPH
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "lab6_error_curve.png"
)

plt.savefig(
    output_file,
    dpi=300
)

print("\nGraph saved to:")
print(output_file)

plt.show()


# ------------------------------------------------------------
# 12. FINAL MESSAGE
# ------------------------------------------------------------

print("\n======================================")
print("LAB 6 COMPLETED SUCCESSFULLY")
print("======================================")