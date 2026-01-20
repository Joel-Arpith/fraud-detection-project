import pandas as pd
import numpy as np

def verify_preprocessing():
    print("Verifying Preprocessing...")
    
    files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    base_path = 'data/processed/'
    
    # 1. Check if files exist
    for f in files:
        try:
            df = pd.read_csv(base_path + f)
            print(f"Loaded {f}: Shape {df.shape}")
            
            if 'y_' in f:
                counts = df.iloc[:,0].value_counts()
                print(f"Class Distribution for {f}:\n{counts}\n")
                
                # Check balance
                if f == 'y_train.csv':
                    ratio = counts[0] / counts[1]
                    if 0.9 <= ratio <= 1.1:
                        print("✅ Train set is balanced.")
                    else:
                        print(f"❌ Train set is NOT balanced. Ratio: {ratio}")
                
                if f == 'y_test.csv':
                    ratio = counts[0] / counts[1]
                    if ratio > 100:
                         print("✅ Test set preserves original imbalance (Good).")
                    else:
                        print(f"❌ Test set seems balanced! (Bad - data leakage?) Ratio: {ratio}")

        except FileNotFoundError:
            print(f"❌ Error: {f} not found.")

if __name__ == "__main__":
    verify_preprocessing()
