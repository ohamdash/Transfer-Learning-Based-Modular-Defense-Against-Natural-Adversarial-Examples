from GAN_imagenet import *
from helpers import *
import argparse
from config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="", help='Name of the GAN model to evaluate', required=True)
    parser.add_argument('--adv_dataset', type=str, default="", help='Name of the adversarial dataset to test on', choices=['in-a', 'in-a-plus'], required=True)
    parser.add_argument('--finetune_dataset', type=str, default="", help='Name of the finetuning dataset', choices=['in-a', 'in-a-plus'])
    parser.add_argument('--cnn', type=str, default="inc-v3", help='Name of the CNN', choices=['inc-v3', 'resnet'], required=True)
    parser.add_argument('--ben_epochs', type=int, default=10, help='number of benign finetuning epochs')
    parser.add_argument('--adv_epochs', type=int, default=30, help='number of adversarial finetuning epochs')

    args = parser.parse_args()

    finetune = True
    if args.finetune_dataset == "":
        finetune = False

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

    feature_dim = 2048
    models_path = "./models"
    gan_path = f"{models_path}/{args.model_name}"
    gan = GAN(input_shape=feature_dim, input_latent_dim=feature_dim, G_data=None, D_data=None, image_path='./f2f/SF')
    gan.load_model(gan_path)

    if args.cnn == "inc-v3":
        base_model, _ = load_inception_v3()
        classifier = get_sub_model(base_model)

        benign_features_train = benign_features_train_v4
        benign_labels_train = benign_labels_train_v4_1000

        benign_features_test = benign_features_val_new
        benign_labels_test = benign_labels_val_new_1000

        if finetune:
            if args.finetune_dataset == "in-a":
                adversarial_features_train = adversarial_features_train_v4
                adversarial_labels_train = adversarial_labels_train_v4_1000
            elif args.finetune_dataset == "in-a-plus":
                adversarial_features_train = adversarial_features_plus_train_v4
                adversarial_labels_train = adversarial_labels_plus_train_v4_1000

        if args.adv_dataset == "in-a":
            adversarial_features_test = adversarial_features_test_v4
            adversarial_labels_test = adversarial_labels_test_v4_1000
        elif args.adv_dataset == "in-a-plus":
            adversarial_features_test = adversarial_features_plus_test_v4
            adversarial_labels_test = adversarial_labels_plus_test_v4_1000

    elif args.cnn == "resnet":
        _, _, classifier = load_resnet152()

        benign_features_train = benign_features_train_resnet_v4
        benign_labels_train = benign_labels_train_resnet_v4_1000

        benign_features_test = benign_features_val_new_resnet
        benign_labels_test = benign_labels_val_new_resnet_1000

        if finetune:
            if args.finetune_dataset == "in-a":
                adversarial_features_train = adversarial_features_train_resnet_v4
                adversarial_labels_train = adversarial_labels_train_resnet_v4_1000
            elif args.finetune_dataset == "in-a-plus":
                adversarial_features_train = adversarial_features_plus_train_resnet_v4
                adversarial_labels_train = adversarial_labels_plus_train_resnet_v4_1000

        if args.adv_dataset == "in-a":
            adversarial_features_test = adversarial_features_test_resnet_v4
            adversarial_labels_test = adversarial_labels_test_resnet_v4_1000
        elif args.adv_dataset == "in-a-plus":
            adversarial_features_test = adversarial_features_plus_test_resnet_v4
            adversarial_labels_test = adversarial_labels_plus_test_resnet_v4_1000
 

    classifier.trainable = False

    latent_input = tf.keras.Input(shape=(gan.latent_dim,))

    generated_features = gan.generator(latent_input)
    
    logits = classifier(generated_features)
    
    combined_model = tf.keras.Model(latent_input, logits)
    
    combined_model = tf.keras.Model(
        inputs=latent_input,
        outputs={
            'generated_features': generated_features,
            'predictions': logits
        }
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9)

    combined_model.compile(
        optimizer=optimizer,
        loss={
            'generated_features': 'mse',
            'predictions': 'categorical_crossentropy'
        },
        loss_weights={
            'generated_features': 1.0,
            'predictions': 1.0
        },
        metrics={'predictions': 'accuracy'}
    )


    if finetune:
        combined_model.fit(
            benign_features_train,
            {
                'generated_features': benign_features_train,
                'predictions': benign_labels_train
            },
            batch_size=64,
            epochs=args.ben_epochs
        )

        combined_model.fit(
            adversarial_features_train,
            {
                'generated_features': adversarial_features_train,
                'predictions': adversarial_labels_train
            },
            batch_size=64,
            epochs=args.adv_epochs
        )

    combined_model.evaluate(
        benign_features_test,
        {
            'generated_features': benign_features_test,
            'predictions': benign_labels_test
        }
    )

    combined_model.evaluate(
        adversarial_features_test,
        {
            'generated_features': adversarial_features_test,
            'predictions': adversarial_labels_test
        }
    )

