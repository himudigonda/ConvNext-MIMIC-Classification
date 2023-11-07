import os
import pandas as pd
import shutil

# Function to create directories given a root directory and a list of sub-directories
def create_directories(root_dir, sub_dirs):
    for subdir in sub_dirs:
        os.makedirs(os.path.join(root_dir, str(subdir)), exist_ok=True)

# Function to move images based on DataFrame entries
def move_images(df, source_dir, target_root_dir):
    for _, row in df.iterrows():
        img_path = row["path"]
        img_labels = row["labels"].strip("[]").split(", ")
        
        # Extract directories with label "1.0" from the image labels
        directories = [i for i, x in enumerate(img_labels) if x == "1.0"]
        
        # Get the image filename from the path
        image_filename = os.path.basename(img_path)
        
        # Move the image to the appropriate target directories
        for label in directories:
            target_path = os.path.join(target_root_dir, str(label), image_filename)
            shutil.copy(img_path, target_path)

# Main function where the script execution starts
def main():
    # Set the root directory of your dataset
    dataset_root = 'mimic-cxr-jpg'
    new_dataset_root = 'modified-' + dataset_root
    
    # Path to the annotations CSV files
    train_annotations_file = os.path.join(dataset_root, 'train_fin.csv')
    test_annotations_file = os.path.join(dataset_root, 'test.csv')
    
    # Read the CSV files into pandas DataFrames
    train_df = pd.read_csv(train_annotations_file)
    test_df = pd.read_csv(test_annotations_file)
    
    # Create directories for train and test data (0 to 13 labels)
    train_dir = os.path.join(new_dataset_root, 'train')
    test_dir = os.path.join(new_dataset_root, 'test')
    create_directories(train_dir, range(14))
    create_directories(test_dir, range(14))
    
    # Move images for training and testing datasets based on annotations
    move_images(train_df, dataset_root, train_dir)
    move_images(test_df, dataset_root, test_dir)
    
    print("Dataset conversion completed.")

# Check if the script is being run directly (not imported) and then call the main function
if __name__ == "__main__":
    main()
