from onelayer import build_onelayer_classifier
from helpers import *

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
    #tf.get_logger().setLevel('ERROR')

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

    print_step("Loading Data")

    base_dir = "./data/features"

    benign_features = np.load(f"{base_dir}/benign_features_resnet-v4.npy")
    benign_labels = np.load(f"{base_dir}/benign_labels_resnet-v4.npy")
    adversarial_features = np.load(f"{base_dir}/adversarial_features_plus_resnet-v4.npy")
    adversarial_labels = np.load(f"{base_dir}/adversarial_labels_plus_resnet-v4.npy")


    log_path = "./logs"
    model_path = "./models"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)


    benign_labels = map_200_to_1000(benign_labels)
    adversarial_labels = map_200_to_1000(adversarial_labels)

    print(benign_labels.shape)
    print(adversarial_labels.shape)


    model = build_onelayer_classifier(cnn="resnet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'reconstructed_features': 'mse',
            'predictions': 'categorical_crossentropy'
        },
        loss_weights={
            'reconstructed_features': 1.0,
            'predictions': 1.0
        },
        metrics={'predictions': 'accuracy'}
    )

    model.fit(
        benign_features,
        {
            'reconstructed_features': benign_features,
            'predictions': benign_labels
        },
        batch_size=64,
        epochs=5
    )

    model.fit(
        adversarial_features,
        {
            'reconstructed_features': adversarial_features,
            'predictions': adversarial_labels
        },
        batch_size=64,
        epochs=10
    )


    model.save(f"{model_path}/generator_classifier_onelayer_plus_new_resnet")