from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "aqi_sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "aqi_model.joblib"

FEATURES = [
    "max_wind_speed",
    "max_temperature",
    "min_temperature",
    "average_temperature",
    "average_wind_speed",
]
TARGET = "aqi"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset missing. First run: python generate_dataset.py")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"MAE:  {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.2f}")
    print(f"R2:   {r2_score(y_test, predictions):.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
