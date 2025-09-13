import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from LoadDataset import Extract


def preprocess_data(df):
    y = df["price"]

    drop_cols = ["price", "property_id", "page_url"]
    X = df.drop(columns=drop_cols, errors="ignore")

    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, categorical_cols, encoders


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, verbose=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    return model, rmse, r2


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "House_Price_dataset.csv")

    df = Extract(dataset_path)

    X, y, categorical_cols, encoders = preprocess_data(df)

    
    experiment_name = f"house_price_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Training"):

        model, rmse, r2 = train_model(X, y)

        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

       
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("random_state", 42)

        
        mlflow.sklearn.log_model(model, "random_forest_model")

       
        model_package = {
            "model": model,
            "categorical_cols": categorical_cols,
            "encoders": encoders
        }
        joblib.dump(model_package, "house_price_model.pkl")

        print("💾 Model + encoders saved as house_price_model.pkl")
        print(f"Categorical columns tracked: {categorical_cols}")
        print(f"📊 MLflow experiment logged under: {experiment_name}")
