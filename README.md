# Price Prediction Project

This project predicts house prices using a **Random Forest Regressor** on a structured dataset.  
It includes **data preprocessing, model training, evaluation, prediction, experiment tracking with MLflow, and result saving**.

---

## 📂 Project Structure

Project_HousePrices/
│
├── pycache/ # Compiled Python cache files
├── mlruns/ # MLflow experiment tracking logs
├── plots/ # (Optional) Folder for visualizations/plots
│
├── ApacheSQL.ipynb # SQL-related analysis in Jupyter Notebook
├── trendsRawData.ipynb # Raw data analysis and exploration
│
├── House_Price_dataset.csv # Main dataset for training
├── House_Price_dataset.csv.dvc # DVC tracking file for dataset versioning
│
├── house_price_model.pkl # Trained Random Forest model + encoders (Joblib)
├── results.csv # Predictions on last 10% test data (actual vs predicted)
│
├── house_prices_sql_analysis.ipynb # SQL exploration and analysis of dataset
├── LoadDataset.py # Data loading/cleaning utilities
├── train.py # Model training script (with MLflow logging)
├── predict.py # Prediction script for new inputs
│
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders ignored by Git
├── .dvcignore # Files ignored by DVC
└── README.md # Project documentation

---

## ⚙️ Setup

### 1️⃣ Create Virtual Environment (recommended)
```bash
python -m venv venv
# Activate venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### Install Dependencies
pip install -r requirements.txt


### Usage
python train.py

### Track Experiment 
mlflow ui

