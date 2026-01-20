import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

def evaluate_models():
    print("Loading data and models...")
    try:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        lr_model = joblib.load('models/logreg.pkl')
        rf_model = joblib.load('models/rf.pkl')
    except FileNotFoundError as e:
        print(f"Error: Required files not found. {e}")
        return

    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')

    # --- 1. Confusion Matrix ---
    def plot_cm(model, name, filename):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'results/{filename}')
        plt.close()
        print(f"Saved results/{filename}")

    plot_cm(lr_model, 'Logistic Regression', 'confusion_matrix_logreg.png')
    plot_cm(rf_model, 'Random Forest', 'confusion_matrix_rf.png')

    # --- 2. ROC Curve ---
    plt.figure(figsize=(8, 6))
    
    models = [(lr_model, 'Logistic Regression'), (rf_model, 'Random Forest')]
    
    for model, name in models:
        # Get probabilities for the positive class (Fraud)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    plt.close()
    print("Saved results/roc_curve.png")

if __name__ == "__main__":
    evaluate_models()
