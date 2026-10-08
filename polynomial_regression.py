import os
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

ROLL_NO = "BT2024150"

DATA_DIR = "."
OUTPUT_DIR = "."

MAX_DEGREE_VAR1 = 10
MAX_DEGREE_VAR2 = 20


# ============================================================
# Polynomial Regression Function
# ============================================================

def train_polynomial_model(
    train_file,
    test_file,
    output_file,
    problem_name,
    max_degree
):

    print("\n" + "=" * 60)
    print(f"Processing {problem_name}")
    print("=" * 60)

    # Load datasets
    train_path = os.path.join(DATA_DIR, train_file)
    test_path = os.path.join(DATA_DIR, test_file)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training data shape:", train_df.shape)
    print("Testing data shape :", test_df.shape)

    # Separate features and target
    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    X_test = test_df.copy()

    print("Features:", list(X_train.columns))
    print("Number of features:", X_train.shape[1])

    # 5-Fold Cross Validation
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    print("\nTesting polynomial degrees...\n")

    for degree in range(1, max_degree + 1):

        model = Pipeline([
            (
                "polynomial",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "regressor",
                Ridge(alpha=1.0)
            )
        ])

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="neg_mean_squared_error"
        )

        mse = -scores.mean()

        results.append({
            "degree": degree,
            "cv_mse": mse
        })

        print(
            f"Degree {degree:2d}  "
            f"CV MSE = {mse:.6f}"
        )

    # Select best degree
    results_df = pd.DataFrame(results)

    best_index = results_df["cv_mse"].idxmin()

    best_degree = int(
        results_df.loc[best_index, "degree"]
    )

    best_cv_mse = results_df.loc[
        best_index,
        "cv_mse"
    ]

    print("\nBest polynomial degree:", best_degree)
    print("Best CV MSE:", best_cv_mse)

    # Train final model
    final_model = Pipeline([
        (
            "polynomial",
            PolynomialFeatures(
                degree=best_degree,
                include_bias=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            Ridge(alpha=1.0)
        )
    ])

    final_model.fit(
        X_train,
        y_train
    )

    # Training performance
    train_predictions = final_model.predict(X_train)

    train_mse = mean_squared_error(
        y_train,
        train_predictions
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    print("\nFinal Model Performance")
    print("-----------------------")
    print("Training MSE:", train_mse)
    print("Training R² :", train_r2)

    # Test predictions
    test_predictions = final_model.predict(X_test)

    # Create prediction CSV
    submission = pd.DataFrame({
        "y": test_predictions
    })

    submission.to_csv(
        output_file,
        index=False
    )

    print("\nPrediction file created:")
    print(output_file)

    print("\nFirst 5 predictions:")
    print(submission.head())

    return {
        "problem": problem_name,
        "best_degree": best_degree,
        "cv_mse": best_cv_mse,
        "train_mse": train_mse,
        "train_r2": train_r2
    }


# ============================================================
# VAR 1
# ============================================================

var1_results = train_polynomial_model(
    train_file=f"{ROLL_NO}_train_var1.csv",
    test_file=f"{ROLL_NO}_test_var1.csv",
    output_file=f"{ROLL_NO}_pred_var1.csv",
    problem_name="VAR1",
    max_degree=MAX_DEGREE_VAR1
)


# ============================================================
# VAR 2
# ============================================================

var2_results = train_polynomial_model(
    train_file=f"{ROLL_NO}_train_var2.csv",
    test_file=f"{ROLL_NO}_test_var2.csv",
    output_file=f"{ROLL_NO}_pred_var2.csv",
    problem_name="VAR2",
    max_degree=MAX_DEGREE_VAR2
)


# ============================================================
# Print Final Summary
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print("\nVAR1")
print("-" * 30)
print("Best Degree :", var1_results["best_degree"])
print("CV MSE      :", var1_results["cv_mse"])
print("Training MSE:", var1_results["train_mse"])
print("Training R² :", var1_results["train_r2"])

print("\nVAR2")
print("-" * 30)
print("Best Degree :", var2_results["best_degree"])
print("CV MSE      :", var2_results["cv_mse"])
print("Training MSE:", var2_results["train_mse"])
print("Training R² :", var2_results["train_r2"])

