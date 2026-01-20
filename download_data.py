import kagglehub
import shutil
import os

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    
    print("Path to dataset files:", path)
    
    # Create data directory if it doesn't exist
    target_dir = "data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
        
    # Move files to data directory
    # The path returned by kagglehub is a directory containing the csv
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            source_file = os.path.join(path, file_name)
            destination_file = os.path.join(target_dir, file_name)
            
            # Check if file already exists
            if os.path.exists(destination_file):
                print(f"File {file_name} already exists in {target_dir}. Skipping move.")
            else:
                shutil.move(source_file, destination_file)
                print(f"Moved {file_name} to {target_dir}")

if __name__ == "__main__":
    download_dataset()
