import os
import pandas as pd
import shutil

# Set the root directory of your dataset
dataset_root = 'mimic-cxr-jpg'
new_dataset_root = 'modified-' +  dataset_root
# Path to the annotations.csv file
annotations_file = os.path.join(dataset_root, 'test.csv')

# Read the CSV file into a pandas DataFrame
annotations_df = pd.read_csv(annotations_file)

# Create directories for train and validation data
# train_dir = os.path.join(new_dataset_root, 'train')
val_dir = os.path.join(new_dataset_root, 'val')

for i in range(14):
    os.makedirs(os.path.join(val_dir, str(i)), exist_ok=True)
        
# Iterate through the DataFrame and move images to appropriate directories

for _, row in annotations_df.iterrows():
    directories = []
    img_path = row["path"]
    img_labels = row["labels"].strip("[").strip("]").split(", ")
    # print(img_labels[-1])
    # print(type(img_labels))
    for i, x in enumerate(img_labels):
        # print(x + "$")
        if x == "1.0":
            directories.append(i)
    print(directories)
    # image_filename = row['Image Index']  # Assuming the image filenames are in a column named 'Image Index'
    # patient_id = row['Patient ID']        # Assuming the patient IDs are in a column named 'Patient ID'
    # target_directory = train_dir

    # Create subdirectories if they don't exist
    # patient_directory = os.path.join(target_directory, f'p{patient_id}')
    # series_directory = os.path.join(patient_directory, f'p{image_filename[:7]}')
    
    image_filename = img_path.split("/")[-1]
        
    for label in directories:
        target_path = os.path.join(val_dir, str(label), image_filename)
        shutil.copy(source_path, target_path)
    # Move the image to the appropriate directory
    # source_path = os.path.join(img_path)
    # target_path = os.path.join(series_directory, image_filename)
    


print("Dataset conversion completed.")