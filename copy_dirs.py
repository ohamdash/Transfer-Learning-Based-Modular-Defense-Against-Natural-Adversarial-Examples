import os
import shutil

def copy_class_dirs(classes_file, dataset_dir, dest_dir):
    # Read class ids from the text file (assumes each line like: "n01498041 stingray")
    with open(classes_file, 'r') as f:
        class_ids = [line.strip().split()[0] for line in f if line.strip()]
    
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    for class_id in class_ids:
        src_path = os.path.join(dataset_dir, class_id)
        dest_path = os.path.join(dest_dir, class_id)
        if os.path.isdir(src_path):
            # If destination directory already exists, you can choose to skip or overwrite.
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            else:
                print(f"Destination {dest_path} already exists, skipping.")
        else:
            print(f"Source directory {src_path} does not exist, skipping.")

if __name__ == "__main__":
    # Update these paths with your actual file locations
    classes_file = "./imagenet-a-200-classes-v2.txt"  # The file with class ids and names
    dataset_dir = "dataset/val/val"         # The directory containing the class-id folders
    dest_dir = "dataset-v2/val-v4"  # The external directory to copy the folders to
    
    copy_class_dirs(classes_file, dataset_dir, dest_dir)
