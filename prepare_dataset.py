import random
import os
import shutil
import tarfile

# Paths
imagenet_a_plust_path = "./original-datasets/imagenet-a-plus/IN-A-Plus"
imagenet_a_path = "./original-datasets/imagenet-a"
imagenet_path = "./original-datasets/imagenet/train"

new_dataset_dir = "dataset-test"

if os.path.exists(new_dataset_dir):
    answer = input(f"dataset directory \"{new_dataset_dir}\" already exists, delete it? (y/n): ").lower()
    if answer == "y":
        shutil.rmtree(new_dataset_dir)
    else:
        #exit()
        pass
os.makedirs(new_dataset_dir, exist_ok=True)


i = 1
# Iterate over ImageNet-A classes
for class_dir in os.listdir(imagenet_a_path):
    class_path = os.path.join(imagenet_a_path, class_dir)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    print(images)

    # Split into 80% train, 20% test
    random.shuffle(images)
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Instead of a class directory for benign images, we expect a tar file named <class_dir>.tar
    tar_file_path = os.path.join(imagenet_path, class_dir + ".tar")
    print(f"{i} - {tar_file_path}\nimages_len: {len(images)}\nsplit_idx: {split_idx}\ntrain_images_len: {len(train_images)}\ntest_images_len: {len(test_images)}")
    i += 1

    '''
    if tar_file_path == "./original-datasets/imagenet/train/n12267677.tar":
        continue
    elif tar_file_path == "./original-datasets/imagenet/train/n12144580.tar":
        continue
    elif tar_file_path == "./original-datasets/imagenet/train/n11879895.tar":
        continue
    elif tar_file_path == "./original-datasets/imagenet/train/n09835506.tar":
        continue
    elif tar_file_path == "./original-datasets/imagenet/train/n12057211.tar":
        continue
    elif tar_file_path == "./original-datasets/imagenet/train/n09472597.tar":
        continue
    '''
    
    if not os.path.exists(tar_file_path):
        print(f"Tar file for class {class_dir} not found in ImageNet-Mini, skipping benign images.")
    else:
        # Open the tar file and get a list of benign image members (in the tar's root)
        with tarfile.open(tar_file_path, "r") as tar:
            benign_members = [
                member for member in tar.getmembers()
                #if member.isfile() and os.path.basename(member.name) == member.name
                if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            random.shuffle(benign_members)
            benign_members = benign_members[:len(train_images)]  # Limit to the same count as train_images
            print(f"benign_members_len: {len(benign_members)}")
            # Create the benign destination directory
            os.makedirs(f"{new_dataset_dir}/train/imagenet/{class_dir}", exist_ok=True)

            # Extract each selected benign image and write to disk without fully extracting the tar file
            for idx, member in enumerate(benign_members):
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    benign_dest = f"{new_dataset_dir}/train/imagenet/{class_dir}/benign_{idx}.jpg"
                    with open(benign_dest, "wb") as out_f:
                        out_f.write(extracted_file.read())

    # Check if the class exists for ImageNet-A images (the original benign code assumed a directory; now we use tar files)
    # Create directories for ImageNet-A train and test splits
    os.makedirs(f"{new_dataset_dir}/train/imagenet-a/{class_dir}", exist_ok=True)
    os.makedirs(f"{new_dataset_dir}/test/imagenet-a/{class_dir}", exist_ok=True)

    # Copy ImageNet-A images for training and testing
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), f"{new_dataset_dir}/train/imagenet-a/{class_dir}/{img}")

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), f"{new_dataset_dir}/test/imagenet-a/{class_dir}/{img}")

print("Dataset preparation complete!")
