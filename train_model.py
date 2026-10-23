from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "student_model.pkl"


FEATURES = [
    "attendance",
    "study_hours",
    "assignment_score",
    "previous_score"
]

TARGET = "final_score"


def train_model():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run generate_dataset.py first."
        )

    data = pd.read_csv(DATA_FILE)

    # 1. Check required columns
    required_columns = FEATURES + [TARGET]

    # Normalize column names to lowercase
    data.columns = data.columns.str.strip().str.lower()

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )

    # 2. Select required columns and remove missing records
    data = data[required_columns].dropna()

    if len(data) < 10:
        raise ValueError(
            "At least 10 valid records are required."
        )

    # 3. Separate features and target
    X = data[FEATURES]
    y = data[TARGET]

    # 4. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # 5. Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Evaluate model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # 8. Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 9. Save model and metadata
    model_data = {
        "model": model,
        "features": FEATURES,
        "mae": float(mae),
        "r2": float(r2)
    }

    with MODEL_FILE.open("wb") as file:
        pickle.dump(model_data, file)

    # 10. Display results
    print("\nModel training completed successfully!")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")
    print("Model saved at:", MODEL_FILE)


if __name__ == "__main__":
    train_model()