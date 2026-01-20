import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def verify_eda():
    print("Verifying EDA steps...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/creditcard.csv')
        print(f"Dataset loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # 2. Check Missing Values
    missing_values = df.isnull().sum().max()
    print(f"Max missing values: {missing_values}")
    
    # 3. Class Imbalance
    class_counts = df['Class'].value_counts()
    print("Class distribution:\n", class_counts)
    
    # 4. Generate Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.yscale('log')
    plt.savefig('class_distribution.png')
    print("Saved class_distribution.png")
    
    # 5. Generate Time/Amount Distribution Plot
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))
    amount_val = df['Amount'].values
    time_val = df['Time'].values

    sns.histplot(amount_val, ax=ax[0], color='r', bins=50, kde=True)
    ax[0].set_title('Distribution of Transaction Amount')
    ax[0].set_xlim([min(amount_val), 20000])

    sns.histplot(time_val, ax=ax[1], color='b', bins=50, kde=True)
    ax[1].set_title('Distribution of Transaction Time')
    
    plt.savefig('distributions.png')
    print("Saved distributions.png")

if __name__ == "__main__":
    verify_eda()
