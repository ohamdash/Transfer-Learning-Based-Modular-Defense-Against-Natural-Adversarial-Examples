import tensorflow as tf
from helpers import load_inception_v3, get_sub_model, load_resnet152, print_step

latent_dim = 2048
feature_dim = latent_dim
img_shape = latent_dim

models_path = "./models"



def build_onelayer(name="onelayer"):
    print_step("Building generator")
    model = tf.keras.Sequential(name=name)
    
    model.add(tf.keras.layers.Dense(2048, input_dim=latent_dim))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.summary()

    input = tf.keras.Input(shape=(latent_dim, ))
    x = model(input)

    return tf.keras.Model(input, x)




# cnn options: incv3, resnet
def build_onelayer_classifier(cnn="incv3") -> tf.keras.Model:

    classifier = None
    if cnn == "incv3":
        base_model, _ = load_inception_v3()
        classifier = get_sub_model(base_model)
    elif cnn == "resnet":
        _, _, classifier = load_resnet152()
    else:
        raise ValueError("Invalid cnn option")
    

    generator = build_onelayer()

    print(generator.summary())

    classifier.trainable = False

    # Connect the models
    input_features = tf.keras.Input(shape=(latent_dim,))
    refined_features = generator(input_features)
    predictions = classifier(refined_features)

    model = tf.keras.Model(inputs=input_features, outputs={
            'reconstructed_features': refined_features,
            'predictions': predictions
        })

    return model