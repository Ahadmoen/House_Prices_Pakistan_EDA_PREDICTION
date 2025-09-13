import os
import pandas as pd
import joblib
from LoadDataset import Extract
from train import preprocess_data  


def load_model(path="house_price_model.pkl"):
    """Load trained model and encoders"""
    return joblib.load(path)


def preprocess_input(df, categorical_cols, encoders):
    """Apply same preprocessing as training"""
    for col in categorical_cols:
        if col in df:
            le = encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df


def evaluate_last_10_percent(dataset_path, output_file="results.csv"):
    """Take last 10% of dataset, predict, compare with actual, and save to CSV"""
    
    df = Extract(dataset_path)

   
    package = load_model()
    model = package["model"]
    categorical_cols = package["categorical_cols"]
    encoders = package["encoders"]

    
    cutoff = int(len(df) * 0.9)
    df_test = df.iloc[cutoff:].copy()

    # Keep actual prices
    actual_prices = df_test["price"].values

    
    drop_cols = ["price", "property_id", "page_url"]
    X_test = df_test.drop(columns=drop_cols, errors="ignore")
    X_test = preprocess_input(X_test, categorical_cols, encoders)

    
    predictions = model.predict(X_test)

   
    results = df_test.copy()
    results["Predicted Price"] = predictions

    
    results[["price", "Predicted Price"]].to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "House_Price_dataset.csv")

    evaluate_last_10_percent(dataset_path, output_file="results.csv")
