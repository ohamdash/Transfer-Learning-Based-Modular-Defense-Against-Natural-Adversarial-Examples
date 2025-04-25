import random
import os
import shutil
import tarfile

# Paths
imagenet_a_plus_path = "./original-datasets/imagenet-a-plus/IN-A-Plus"
imagenet_a_path = "./original-datasets/imagenet-a"
imagenet_path = "./original-datasets/imagenet/train"

new_dataset_dir = "dataset-test"

if os.path.exists(new_dataset_dir):
    answer = input(f"dataset directory \"{new_dataset_dir}\" already exists, delete it? (y/n): ").lower()
    if answer == "y":
        shutil.rmtree(new_dataset_dir)
    else:
        pass
os.makedirs(new_dataset_dir, exist_ok=True)

i = 1
# Iterate over ImageNet-A classes
for class_dir in os.listdir(imagenet_a_path):
    imagenet_a_class_path = os.path.join(imagenet_a_path, class_dir)
    imagenet_a_plus_class_path = os.path.join(imagenet_a_plus_path, class_dir)

    if not os.path.isdir(imagenet_a_class_path):
        continue

    # Get image lists for both datasets
    imagenet_a_images = os.listdir(imagenet_a_class_path)
    imagenet_a_plus_images = os.listdir(imagenet_a_plus_class_path) if os.path.isdir(imagenet_a_plus_class_path) else []

    # Convert to sets for easier manipulation
    imagenet_a_set = set(imagenet_a_images)
    imagenet_a_plus_set = set(imagenet_a_plus_images)

    # All unique images across both datasets
    all_images = list(imagenet_a_set)  # Since imagenet-a-plus is a subset, imagenet-a contains all images

    # Shuffle all images
    random.shuffle(all_images)
    total_images = len(all_images)
    split_idx = int(0.8 * total_images)

    # Unified train/test split
    train_images_all = all_images[:split_idx]
    test_images_all = all_images[split_idx:]

    # Split into imagenet-a and imagenet-a-plus while respecting the unified split
    train_images_a = [img for img in train_images_all if img in imagenet_a_set]
    test_images_a = [img for img in test_images_all if img in imagenet_a_set]
    train_images_a_plus = [img for img in train_images_all if img in imagenet_a_plus_set]
    test_images_a_plus = [img for img in test_images_all if img in imagenet_a_plus_set]

    # Debugging output
    print(f"{i} - Class: {class_dir}")
    print(f"Total images: {total_images}, Split idx: {split_idx}")
    print(f"Imagenet-A - Train: {len(train_images_a)}, Test: {len(test_images_a)}")
    print(f"Imagenet-A-Plus - Train: {len(train_images_a_plus)}, Test: {len(test_images_a_plus)}")
    i += 1

    # Handle benign images from ImageNet
    tar_file_path = os.path.join(imagenet_path, class_dir + ".tar")
    if not os.path.exists(tar_file_path):
        print(f"Tar file for class {class_dir} not found in ImageNet, skipping benign images.")
    else:
        with tarfile.open(tar_file_path, "r") as tar:
            benign_members = [
                member for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            random.shuffle(benign_members)
            benign_members = benign_members[:len(train_images_a)]  # Match count to imagenet-a train
            print(f"benign_members_len: {len(benign_members)}")
            os.makedirs(f"{new_dataset_dir}/train/imagenet/{class_dir}", exist_ok=True)
            for idx, member in enumerate(benign_members):
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    benign_dest = f"{new_dataset_dir}/train/imagenet/{class_dir}/benign_{idx}.jpg"
                    with open(benign_dest, "wb") as out_f:
                        out_f.write(extracted_file.read())

    # Create directories for both datasets
    os.makedirs(f"{new_dataset_dir}/train/imagenet-a/{class_dir}", exist_ok=True)
    os.makedirs(f"{new_dataset_dir}/test/imagenet-a/{class_dir}", exist_ok=True)
    os.makedirs(f"{new_dataset_dir}/train/imagenet-a-plus/{class_dir}", exist_ok=True)
    os.makedirs(f"{new_dataset_dir}/test/imagenet-a-plus/{class_dir}", exist_ok=True)

    # Copy ImageNet-A images
    for img in train_images_a:
        shutil.copy(os.path.join(imagenet_a_class_path, img), f"{new_dataset_dir}/train/imagenet-a/{class_dir}/{img}")
    for img in test_images_a:
        shutil.copy(os.path.join(imagenet_a_class_path, img), f"{new_dataset_dir}/test/imagenet-a/{class_dir}/{img}")

    # Copy ImageNet-A-Plus images (only if the class exists in imagenet-a-plus)
    if os.path.isdir(imagenet_a_plus_class_path):
        for img in train_images_a_plus:
            shutil.copy(os.path.join(imagenet_a_plus_class_path, img), f"{new_dataset_dir}/train/imagenet-a-plus/{class_dir}/{img}")
        for img in test_images_a_plus:
            shutil.copy(os.path.join(imagenet_a_plus_class_path, img), f"{new_dataset_dir}/test/imagenet-a-plus/{class_dir}/{img}")

print("Dataset preparation complete!")
