from GAN_imagenet import *
from helpers import *
from config import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_epochs', type=int, default=25001, help='Number of GAN epochs')
    parser.add_argument('--name_suffix', type=str, default="", help='Name suffix for the gan model and logs')
    args = parser.parse_args()

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

    benign_features = benign_features_train_v4
    adversarial_features = adversarial_features_plus_train_v4

    log_path = "./logs"
    model_path = "./models"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)


    gan_epochs = args.gan_epochs
    gan_batchsize = 64
    feature_dim = benign_features.shape[1] # 2048

    print("feature_dim:", feature_dim)

    print_step("Training Salient Feature GAN (SF)")

    adv_shape = adversarial_features.shape[0]
    benign_features = benign_features[0:adv_shape]

    targeted_feature = np.concatenate((benign_features, benign_features))
    input_block = np.concatenate((benign_features, adversarial_features))
    print("input_block shape:", np.shape(input_block), "targeted_feature shape:", np.shape(targeted_feature))
        
    sf_gan = GAN(input_shape=feature_dim, input_latent_dim=feature_dim, G_data=input_block, D_data=targeted_feature,
                    image_path='./f2f/SF')

    sf_gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)


    name_suffix = args.name_suffix
    if name_suffix != "":
        name_suffix = "-" + name_suffix
    
    sf_gan.showlogs(log_path + f"/SF_GAN-{gan_epochs}{name_suffix}")

    sf_gan.save_model(model_path + f"/SF_GAN-{gan_epochs}{name_suffix}")



    print_step("Training Trivial Feature GAN (TF)")

    targeted_feature = np.concatenate((adversarial_features, adversarial_features))
    input_block = np.concatenate((benign_features, adversarial_features))
    print("input_block shape:", np.shape(input_block), "targeted_feature shape:", np.shape(targeted_feature))

    tf_gan = GAN(input_shape=feature_dim, input_latent_dim=feature_dim, G_data=input_block, D_data=targeted_feature,
                    image_path='./f2f/NSF')

    tf_gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)

    tf_gan.showlogs(log_path + f"/TF_GAN-{gan_epochs}{name_suffix}")

    tf_gan.save_model(model_path + f"/TF_GAN-{gan_epochs}{name_suffix}")

    

