from helpers import *
import numpy as np
import time
from keras.models import load_model
from config import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    refined_classifier = load_model(f"{models_path}/generator_classifier_onelayer_plus_new_resnet")

    refined_classifier.compile(
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

    output = refined_classifier.evaluate(
        adversarial_features_test_resnet_v4,
        {
            'reconstructed_features': adversarial_features_test_v4,
            'predictions': adversarial_labels_test_v4_1000
        }
    )

    output = refined_classifier.evaluate(
        adversarial_features_plus_test_resnet_v4,
        {
            'reconstructed_features': adversarial_features_plus_test_v4,
            'predictions': adversarial_labels_plus_test_v4_1000
        }
    )

    output = refined_classifier.evaluate(
        benign_features_val_new_resnet,
        {
            'reconstructed_features': benign_features_val_new,
            'predictions': benign_labels_val_new_1000
        }
    )