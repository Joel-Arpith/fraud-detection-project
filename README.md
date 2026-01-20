# Fraud Detection using Machine Learning (Imbalanced Data)

## Overview
This project focuses on **detecting fraudulent transactions** using classical Machine Learning techniques on a **highly imbalanced dataset**.

The primary objective is to understand **why accuracy fails in fraud detection** and how alternative evaluation metrics improve real-world performance.

The project is designed to be:
- Reproducible
- GitHub-ready
- Focused on fundamentals before deep learning

---

## Problem Statement
Fraud detection is a **binary classification problem** where:
- Class `0` → Legitimate transaction
- Class `1` → Fraudulent transaction

The key challenge is **extreme class imbalance**, where fraudulent transactions form a very small fraction of the data.  
This makes naïve models appear accurate while being practically useless.

---

## Dataset
- **Source**: Kaggle – Fraud Detection Dataset
- **Access Method**: `kagglehub` (Python API)
- **Target Column**: `Class`
- **Imbalance Ratio**: ~99.8% non-fraud vs ~0.2% fraud

> The dataset is not committed to the repository.  
> It is downloaded programmatically for reproducibility.

---

## Project Structure
fraud-detection-ml/
│
├── notebooks/
│ └── 01_data_exploration.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── eda.py
│ └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md


---

## Tools & Libraries
- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- kagglehub

---

## Workflow
1. Download dataset using Kaggle API
2. Perform exploratory data analysis (EDA)
3. Convert data to NumPy arrays
4. Train baseline machine learning models
5. Evaluate using fraud-appropriate metrics
6. Improve performance using imbalance-aware techniques

---

## Why This Project Matters
This project emphasizes **real-world machine learning practices**:
- Business-critical metrics over raw accuracy
- Proper handling of imbalanced datasets
- Reproducible pipelines
- Clean project organization

---

## Future Improvements
- SMOTE and resampling techniques
- Isolation Forest for anomaly detection
- Precision–Recall curve analysis
- Hyperparameter tuning
- Deployment-ready pipeline

---

## Author
**Joel Arpith**  
Applied Machine Learning | Fraud Detection | Anomaly Detection

---

## License
This project is intended for educational and research purposes.
