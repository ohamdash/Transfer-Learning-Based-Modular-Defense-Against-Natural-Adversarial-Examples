from helpers import *

models_path = "./models"
features_path = "./data/features"
gan_path = f"{models_path}/SF_GAN_large"

benign_features_train_path_v4 = f"{features_path}/benign_features-v4.npy"
benign_features_train_path_resnet_v4 = f"{features_path}/benign_features_resnet-v4.npy"
benign_features_val_full_path = f"{features_path}/benign_features_val-full.npy"
benign_features_val_new_path = f"{features_path}/benign_features_val-new.npy"
benign_features_val_new_path_resnet = f"{features_path}/benign_features_val_resnet-new.npy"
benign_labels_train_path_v4 = f"{features_path}/benign_labels-v4.npy"
benign_labels_train_path_resnet_v4 = f"{features_path}/benign_labels_resnet-v4.npy"
benign_labels_val_full_path = f"{features_path}/benign_labels_val-full.npy"
benign_labels_val_new_path = f"{features_path}/benign_labels_val-new.npy"
benign_labels_val_new_path_resnet = f"{features_path}/benign_labels_val_resnet-new.npy"
adversarial_features_train_path_v4 = f"{features_path}/adversarial_features-v4.npy"
adversarial_features_train_path_resnet_v4 = f"{features_path}/adversarial_features_resnet-v4.npy"
adversarial_features_plus_train_path_v4 = f"{features_path}/adversarial_features_plus-v4.npy"
adversarial_features_plus_train_path_resnet_v4 = f"{features_path}/adversarial_features_plus_resnet-v4.npy"
adversarial_features_test_path_v4 = f"{features_path}/adversarial_features_test-v4.npy"
adversarial_features_test_path_resnet_v4 = f"{features_path}/adversarial_features_test_resnet-v4.npy"
adversarial_features_plus_test_path_v4 = f"{features_path}/adversarial_features_plus_test-v4.npy"
adversarial_features_plus_test_path_resnet_v4 = f"{features_path}/adversarial_features_plus_test_resnet-v4.npy"
adversarial_labels_train_path_v4 = f"{features_path}/adversarial_labels-v4.npy"
adversarial_labels_train_path_resnet_v4 = f"{features_path}/adversarial_labels_resnet-v4.npy"
adversarial_labels_plus_train_path_v4 = f"{features_path}/adversarial_labels_plus-v4.npy"
adversarial_labels_plus_train_path_resnet_v4 = f"{features_path}/adversarial_labels_plus_resnet-v4.npy"
adversarial_labels_test_path_v4 = f"{features_path}/adversarial_labels_test-v4.npy"
adversarial_labels_test_path_resnet_v4 = f"{features_path}/adversarial_labels_test_resnet-v4.npy"
adversarial_labels_plus_test_path_v4 = f"{features_path}/adversarial_labels_plus_test-v4.npy"
adversarial_labels_plus_test_path_resnet_v4 = f"{features_path}/adversarial_labels_plus_test_resnet-v4.npy"

print_step("Loading Data")

benign_features_train_v4 = np.load(benign_features_train_path_v4)
benign_features_train_resnet_v4 = np.load(benign_features_train_path_resnet_v4)
benign_features_val_full = np.load(benign_features_val_full_path)
benign_features_val_new = np.load(benign_features_val_new_path)
benign_features_val_new_resnet = np.load(benign_features_val_new_path_resnet)
benign_labels_train_v4 = np.load(benign_labels_train_path_v4)
benign_labels_train_resnet_v4 = np.load(benign_labels_train_path_resnet_v4)
benign_labels_val_full = np.load(benign_labels_val_full_path)
benign_labels_val_new = np.load(benign_labels_val_new_path)
benign_labels_val_new_resnet = np.load(benign_labels_val_new_path_resnet)
adversarial_features_train_v4 = np.load(adversarial_features_train_path_v4)
adversarial_features_train_resnet_v4 = np.load(adversarial_features_train_path_resnet_v4)
adversarial_features_plus_train_v4 = np.load(adversarial_features_plus_train_path_v4)
adversarial_features_plus_train_resnet_v4 = np.load(adversarial_features_plus_train_path_resnet_v4)
adversarial_features_test_v4 = np.load(adversarial_features_test_path_v4)
adversarial_features_test_resnet_v4 = np.load(adversarial_features_test_path_resnet_v4)
adversarial_features_plus_test_v4 = np.load(adversarial_features_plus_test_path_v4)
adversarial_features_plus_test_resnet_v4 = np.load(adversarial_features_plus_test_path_resnet_v4)
adversarial_labels_train_v4 = np.load(adversarial_labels_train_path_v4)
adversarial_labels_train_resnet_v4 = np.load(adversarial_labels_train_path_resnet_v4)
adversarial_labels_plus_train_v4 = np.load(adversarial_labels_plus_train_path_v4)
adversarial_labels_plus_train_resnet_v4 = np.load(adversarial_labels_plus_train_path_resnet_v4)
adversarial_labels_test_v4 = np.load(adversarial_labels_test_path_v4)
adversarial_labels_test_resnet_v4 = np.load(adversarial_labels_test_path_resnet_v4)
adversarial_labels_plus_test_v4 = np.load(adversarial_labels_plus_test_path_v4)
adversarial_labels_plus_test_resnet_v4 = np.load(adversarial_labels_plus_test_path_resnet_v4)

benign_labels_val_new_1000 = map_200_to_1000(benign_labels_val_new)
benign_labels_val_new_resnet_1000 = map_200_to_1000(benign_labels_val_new_resnet)
adversarial_labels_test_v4_1000 = map_200_to_1000(adversarial_labels_test_v4)
adversarial_labels_train_v4_1000 = map_200_to_1000(adversarial_labels_train_v4)
benign_labels_train_v4_1000 = map_200_to_1000(benign_labels_train_v4)
benign_labels_train_resnet_v4_1000 = map_200_to_1000(benign_labels_train_resnet_v4)
adversarial_labels_plus_train_v4_1000 = map_200_to_1000(adversarial_labels_plus_train_v4)
adversarial_labels_plus_test_v4_1000 = map_200_to_1000(adversarial_labels_plus_test_v4)
adversarial_labels_train_resnet_v4_1000 = map_200_to_1000(adversarial_labels_train_resnet_v4)
adversarial_labels_test_resnet_v4_1000 = map_200_to_1000(adversarial_labels_test_resnet_v4)
adversarial_labels_plus_test_resnet_v4_1000 = map_200_to_1000(adversarial_labels_plus_test_resnet_v4)
adversarial_labels_plus_train_resnet_v4_1000 = map_200_to_1000(adversarial_labels_plus_train_resnet_v4)

latent_dim = 2048