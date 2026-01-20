import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time

def train_models():
    print("Loading data...")
    # Load processed data
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    except FileNotFoundError:
        print("Error: Processed data not found. Run preprocess.py first.")
        return

    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- Model 1: Logistic Regression ---
    print("\nTraining Logistic Regression (Baseline)...")
    lr_model = LogisticRegression(max_iter=1000)
    
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    print(f"Logistic Regression trained in {time.time() - start_time:.2f} seconds.")
    
    print("Evaluating Logistic Regression on Test Set:")
    lr_pred = lr_model.predict(X_test)
    print(classification_report(y_test, lr_pred))
    
    joblib.dump(lr_model, 'models/logreg.pkl')
    print("Saved models/logreg.pkl")

    # --- Model 2: Random Forest ---
    print("\nTraining Random Forest (Advanced)...")
    # Reduced n_estimators and max_depth for faster training in this demo context
    # class_weight='balanced' is not strictly needed because we SMOTEd, but good practice
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    print(f"Random Forest trained in {time.time() - start_time:.2f} seconds.")
    
    print("Evaluating Random Forest on Test Set:")
    rf_pred = rf_model.predict(X_test)
    print(classification_report(y_test, rf_pred))
    
    joblib.dump(rf_model, 'models/rf.pkl')
    print("Saved models/rf.pkl")

if __name__ == "__main__":
    train_models()
