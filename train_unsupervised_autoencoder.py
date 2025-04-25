import unsupervised_autoencoder as autoenc
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

    benign_features = np.load(f"{base_dir}/benign_features_resnet-v4.npy") # load benign features, either from inc-v3 or resnet.
    adversarial_features = np.load(f"{base_dir}/adversarial_features_plus_resnet-v4.npy") # load adv features, either from inc-v3 or resnet, in-a or in-a+


    all_features = np.concatenate((benign_features, adversarial_features), axis=0)

    #all_features = benign_features

    latent_dim = 2048

    autoencoder, encoder, decoder = autoenc.build_model(latent_dim)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    autoencoder.fit(all_features, all_features, epochs=100, batch_size=32, validation_split=0.32)

    autoencoder.save("models/unsupervised_autoencoder_plus_resnet-v4")


