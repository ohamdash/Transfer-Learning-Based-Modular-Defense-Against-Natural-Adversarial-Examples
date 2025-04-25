# Finetune and test the pretrained unsupervised autoencoder


import unsupervised_autoencoder as autoenc
from helpers import *
from keras.models import load_model, Model
from config import *


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



    ###################################################
    # ONLY change the 6 vars below and the models !!! #
    ###################################################

    train_features = np.concatenate((benign_features_train_v4, adversarial_features_plus_train_v4), axis=0)
    train_labels = np.concatenate((benign_labels_train_v4_1000, adversarial_labels_plus_train_v4_1000), axis=0)
    
    test_features_ben = benign_features_val_new
    test_labels_ben = benign_labels_val_new_1000
    test_features_adv = adversarial_features_plus_test_v4
    test_labels_adv = adversarial_labels_plus_test_v4_1000


    autoencoder = load_model("models/unsupervised_autoencoder_plus-v4")

    cnn = "incv3" # options: incv3, resnet

    ################
    # END EDITABLE #
    ################

    autoencoder.summary()

    classifier = None

    if cnn == "incv3":
        base_model, _ = load_inception_v3()
        classifier = get_sub_model(base_model)
    elif cnn == "resnet":
        _, _, classifier = load_resnet152()
    else:
        print("Invalid CNN model specified.")
        exit(1)
    
    classifier.trainable = False

    classifier.summary()

    input_layer = Input(shape=(2048,))

    refined_features = autoencoder(input_layer)

    predictions = classifier(refined_features)

    combined_model = Model(inputs=input_layer, outputs=predictions)

    combined_model = Model(
        inputs=input_layer,
        outputs={
            'reconstructed_features': refined_features,
            'predictions': predictions
        }
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    combined_model.compile(
        optimizer=optimizer,
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
    

    is_train_plus = True
    is_test_plus = True

    suffix_train = ""
    suffix_test = ""

    if is_train_plus:
        suffix_train = "_plus"

    if is_test_plus:
        suffix_test = "_plus"



    classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    classifier.evaluate(
        test_features_ben,
        test_labels_ben
    )
    classifier.evaluate(
        test_features_adv,
        test_labels_adv
    )


    combined_model.fit(
        train_features,
        {
            'reconstructed_features': train_features,
            'predictions': train_labels
        },
        epochs=10,
        batch_size=32
    )

    output = combined_model.evaluate(
        test_features_adv,
        {
            'reconstructed_features': test_features_adv,
            'predictions': test_labels_adv
        }
    )


    output = combined_model.evaluate(
        test_features_ben,
        {
            'reconstructed_features': test_features_ben,
            'predictions': test_labels_ben
        }
    )



