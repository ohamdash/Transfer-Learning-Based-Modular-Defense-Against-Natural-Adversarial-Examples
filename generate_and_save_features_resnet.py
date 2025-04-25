from helpers import *

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
    #tf.get_logger().setLevel('ERROR')
    

    _, feature_model, _ = load_resnet152()


    benign_features, benign_labels, adversarial_features, adversarial_labels = load_imagenet_data_resnet152(
        benign_dir='./dataset-v4/train/imagenet',
        adversarial_dir='./dataset-v4/train/imagenet-a',
        feature_model=feature_model, 
        generate_benign=True,
        generate_adversarial=True)
    
    print_step("Data loaded")
    print(benign_features.shape)
    print(benign_labels.shape)
    print(adversarial_features.shape)
    print(adversarial_labels.shape)

    features_path = "./data/features"

    os.makedirs(features_path, exist_ok=True)

    np.save(f"{features_path}/benign_features_resnet-v4.npy", benign_features)
    np.save(f"{features_path}/benign_labels_resnet-v4.npy", benign_labels)
    np.save(f"{features_path}/adversarial_features_resnet-v4.npy", adversarial_features)
    np.save(f"{features_path}/adversarial_labels_resnet-v4.npy", adversarial_labels)

    print("Features and labels saved successfully.")