import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import os

def preprocess_data():
    print("Starting Preprocessing...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        print("Error: data/creditcard.csv not found. Please run download_data.py first.")
        return

    # 2. Scaling
    # RobustScaler is less prone to outliers
    print("Scaling 'Amount' and 'Time'...")
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Move scaled columns to the beginning for easier viewing (optional but nice)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # 3. Splitting
    print("Splitting data into Train and Test sets...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratify ensure the test set has the same proportion of fraud as the original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. SMOTE (Oversampling)
    # Apply SMOTE ONLY to the training data to prevent data leakage!
    print("Applying SMOTE to Training set...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"Original Train shape: {y_train.shape}, Fraud count: {y_train.sum()}")
    print(f"Resampled Train shape: {y_train_res.shape}, Fraud count: {y_train_res.sum()}")
    
    # 5. Save Processed Data
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving processed files...")
    X_train_res.to_csv(f'{output_dir}/X_train.csv', index=False)
    y_train_res.to_csv(f'{output_dir}/y_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    print(f"Preprocessing Complete. Files saved to {output_dir}")

if __name__ == "__main__":
    preprocess_data()
