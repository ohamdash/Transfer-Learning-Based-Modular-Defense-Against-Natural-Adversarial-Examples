import datetime
import math
import os

import io
from contextlib import redirect_stdout

from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.utils import to_categorical

import tensorflow_datasets as tfds

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet

from keras.applications.resnet_v2 import ResNet152V2

def load_inception_v3():
    """
    Loads the pre-trained Inception v3 model and prepares the feature extraction model.
    
    :return: Tuple of (base_model, feature_model)
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)
    # Extract features from 'avg_pool' layer
    feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return base_model, feature_model

def load_resnet152():
    """
    Loads the pre-trained ResNet152 model and prepares the feature extraction model and classifier.

    :return: Tuple of (base_model, feature_model, classifier)
    """
    base_model = ResNet152V2(weights='imagenet', include_top=True)
    
    # Extract features from 'avg_pool' layer
    feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    avg_pool = base_model.get_layer("avg_pool").output
    predictions = base_model.get_layer("predictions")(avg_pool)

    # Create the sub-model with avg_pool as input and predictions as output
    classifier = Model(inputs=avg_pool, outputs=predictions)
    
    return base_model, feature_model, classifier

def preprocess_image(image, label):
    """
    Preprocesses the input image for Inception v3.

    :param image: TensorFlow image tensor
    :param label: Corresponding label
    :return: Tuple of (preprocessed image, label)
    """
    image = tf.image.resize(image, (299, 299))  # Resize to InceptionV3 input size
    image = preprocess_input(image)  # Normalize for InceptionV3
    return image, label

def load_imagenet_a(batch_size=64, test_size=0.2):
    """
    Loads the ImageNet-A dataset, applies preprocessing, and returns a TensorFlow dataset.

    :param batch_size: Batch size for dataset loading
    :return: Preprocessed TF dataset
    """
    dataset = tfds.load("imagenet_a", split="test", as_supervised=True, data_dir="./tensorflow_datasets/")
    
    # Shuffle the dataset to ensure randomness
    dataset = dataset.shuffle(50, seed=42)

    # Apply the preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Determine the number of test samples
    total_size = 0
    for _ in dataset:
        total_size += 1
    test_size = int(total_size * test_size)

    # Split the dataset into train and test
    test_dataset = dataset.take(test_size)  # Take the first portion for testing
    train_dataset = dataset.skip(test_size)  # Skip the test portion for training

    # Batch and prefetch for efficiency
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


import time

def log(message):
    print(f"{datetime.datetime.now()} - {message}")

def print_step(message):
    print(f"\n\n##################################\n##################################\n===> {message}\n##################################\n##################################\n\n")


def get_sub_model(base_model) -> tf.keras.Model:
    # Get the last two layers
    avg_pool = base_model.get_layer("avg_pool").output
    predictions = base_model.get_layer("predictions")(avg_pool)

    # Create the sub-model with avg_pool as input and predictions as output
    sub_model = Model(inputs=avg_pool, outputs=predictions)

    # Print summary to verify
    sub_model.summary()

    return sub_model


def scheduler(epoch):
    if epoch <= 80:
        return 0.01
    if epoch <= 140:
        return 0.005
    return 0.001





def eval_inception_v3(base_model, dataset):

    # Evaluate on ImageNet-A
    correct_predictions = 0
    total_samples = 0

    print_step("Starting evaluation on imagenet-a")

    for images, labels in dataset:
        predictions = base_model.predict(images)  # Get predictions
        predicted_labels = np.argmax(predictions, axis=1)  # Convert to class indices
        correct_predictions += np.sum(predicted_labels == labels.numpy())  # Compare with true labels
        total_samples += labels.shape[0]
        print(f"[{total_samples}] - Batch accuracy: {correct_predictions / total_samples:.4f}")

    accuracy = correct_predictions / total_samples
    print(f"Model accuracy on ImageNet-A: {accuracy:.4f}")


def eval_submodel(sub_model, features, true_labels):
    """
    Evaluates the sub-model's performance.
    
    :param features: 2048-dimensional feature vectors (from avg_pool)
    :param sub_model: The sub-model for evaluation
    :param true_labels: True labels for evaluation (one-hot encoded)
    """
    
    # Get predictions (logits)
    logits = sub_model.predict(features)

    # Convert logits to class predictions (indices)
    predicted_class_indices = tf.argmax(logits, axis=1)
    
    # Convert predicted indices to one-hot encoding
    predicted_one_hot = tf.one_hot(predicted_class_indices, depth=logits.shape[1])
    

    true_labels_max = np.argmax(true_labels, axis=1)
    predictions_max = np.argmax(predicted_one_hot, axis=1)

    boolean_matrix = true_labels_max == predictions_max

    # Compare the two arrays element-wise and calculate the accuracy
    accuracy = np.mean(boolean_matrix)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(f"Correct Predictions: {np.sum(boolean_matrix)}")
    print(f"Wrong Predictions: {np.sum(~boolean_matrix)}")
    
    return accuracy



def convert_one_hot(label_200_class, imagenet_a_to_imagenet_2012_mapping):
    label_1000_class = np.zeros(1000)
    
    # Map the 200-class label to the 1000-class label using the mapping
    if label_200_class in imagenet_a_to_imagenet_2012_mapping:
        mapped_class = imagenet_a_to_imagenet_2012_mapping[label_200_class]
        label_1000_class[mapped_class] = 1
    else:
        print(f"Warning: Mapping for label {label_200_class} not found!")
    
    return label_1000_class


def load_dataset(benign_dir, adversarial_dir, batch_size=64, img_size=(299, 299)):
    """
    Loads and preprocesses ImageNet data from benign and adversarial directories.
    The images are preprocessed for InceptionV3 and returned as raw image arrays along with their labels.

    :param benign_dir: Path to the directory containing benign ImageNet images
    :param adversarial_dir: Path to the directory containing adversarial ImageNet images
    :param batch_size: Number of samples per batch
    :param img_size: Target image size for InceptionV3 (default is 299x299)
    :return: Tuple of (benign_images, benign_labels, adversarial_images, adversarial_labels)
    """
    
    # Create an ImageDataGenerator that applies InceptionV3 preprocessing.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Generator for benign images.
    benign_generator = datagen.flow_from_directory(
        benign_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    # Calculate the number of steps required to go through the benign dataset.
    benign_steps = math.ceil(benign_generator.samples / batch_size)
    benign_images = []
    benign_labels = []
    
    # Iterate through the benign dataset.
    for _ in range(benign_steps):
        imgs, labels = benign_generator.next()
        benign_images.append(imgs)
        benign_labels.append(labels)
    
    benign_images = np.vstack(benign_images)
    benign_labels = np.vstack(benign_labels)
    
    # Generator for adversarial images.
    adversarial_generator = datagen.flow_from_directory(
        adversarial_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    adversarial_steps = math.ceil(adversarial_generator.samples / batch_size)
    adversarial_images = []
    adversarial_labels = []
    
    # Iterate through the adversarial dataset.
    for _ in range(adversarial_steps):
        imgs, labels = adversarial_generator.next()
        adversarial_images.append(imgs)
        adversarial_labels.append(labels)
    
    adversarial_images = np.vstack(adversarial_images)
    adversarial_labels = np.vstack(adversarial_labels)
    
    return benign_images, benign_labels, adversarial_images, adversarial_labels


def map_200_to_1000(labels):
    with open("imagenet-1000-classes.txt", "r") as f:
        imagenet_classes = [line.strip().split(" ")[0] for line in f.readlines()]

    # Load ImageNet-A class labels (200 classes) from the txt file
    with open("imagenet-a-200-classes.txt", "r") as f:
        imagenet_a_classes = [line.strip().split(" ")[0] for line in f.readlines()]

    # Create a mapping from ImageNet-A to ImageNet 2012 class IDs
    # Here we assume that the classes in imagenet_a_classes correspond directly to those in imagenet_classes.
    # If needed, adjust this mapping based on how ImageNet-A and ImageNet 2012 classes correspond.
    imagenet_a_to_imagenet_2012_mapping = {i: imagenet_classes.index(cls_id) for i, cls_id in enumerate(imagenet_a_classes)}

    lables_1000 = np.array([convert_one_hot(np.argmax(label), imagenet_a_to_imagenet_2012_mapping) for label in labels])

    return lables_1000


def load_imagenet_data(benign_dir, adversarial_dir, feature_model, batch_size=64, img_size=(299, 299), generate_benign=True, generate_adversarial=True):
    """
    Loads and preprocesses ImageNet data from benign and adversarial directories.

    :param benign_dir: Path to benign ImageNet images
    :param adversarial_dir: Path to adversarial ImageNet images
    :param feature_model: Pretrained model to extract features
    :param batch_size: Number of samples per batch
    :param img_size: Target image size for Inception v3
    :return: Tuple of (benign_features, benign_labels, adversarial_features, adversarial_labels)
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    if generate_benign:
        benign_generator = datagen.flow_from_directory(
            benign_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        
        benign_steps = benign_generator.samples // batch_size
        benign_features = []
        benign_labels = []
        for _ in range(benign_steps):
            imgs, labels = benign_generator.next()
            features = feature_model.predict(imgs)
            benign_features.append(features)
            benign_labels.append(labels)
        benign_features = np.vstack(benign_features)
        benign_labels = np.vstack(benign_labels)
    else:
        benign_features = []
        benign_labels = []

    if generate_adversarial:
        adversarial_generator = datagen.flow_from_directory(
            adversarial_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        
        adversarial_steps = adversarial_generator.samples // batch_size
        adversarial_features = []
        adversarial_labels = []
        for _ in range(adversarial_steps):
            imgs, labels = adversarial_generator.next()
            features = feature_model.predict(imgs)
            adversarial_features.append(features)
            adversarial_labels.append(labels)
        adversarial_features = np.vstack(adversarial_features)
        adversarial_labels = np.vstack(adversarial_labels)
    else:
        adversarial_features = []
        adversarial_labels = []

    return benign_features, benign_labels, adversarial_features, adversarial_labels





def load_imagenet_data_resnet152(benign_dir, adversarial_dir, feature_model, batch_size=64, img_size=(224, 224), generate_benign=True, generate_adversarial=True):
    """
    Loads and preprocesses ImageNet data from benign and adversarial directories.

    :param benign_dir: Path to benign ImageNet images
    :param adversarial_dir: Path to adversarial ImageNet images
    :param feature_model: Pretrained model to extract features
    :param batch_size: Number of samples per batch
    :param img_size: Target image size for Inception v3
    :return: Tuple of (benign_features, benign_labels, adversarial_features, adversarial_labels)
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input_resnet)

    if generate_benign:
        benign_generator = datagen.flow_from_directory(
            benign_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        
        benign_steps = benign_generator.samples // batch_size
        benign_features = []
        benign_labels = []
        for _ in range(benign_steps):
            imgs, labels = benign_generator.next()
            features = feature_model.predict(imgs)
            benign_features.append(features)
            benign_labels.append(labels)
        benign_features = np.vstack(benign_features)
        benign_labels = np.vstack(benign_labels)
    else:
        benign_features = []
        benign_labels = []

    if generate_adversarial:
        adversarial_generator = datagen.flow_from_directory(
            adversarial_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        
        adversarial_steps = adversarial_generator.samples // batch_size
        adversarial_features = []
        adversarial_labels = []
        for _ in range(adversarial_steps):
            imgs, labels = adversarial_generator.next()
            features = feature_model.predict(imgs)
            adversarial_features.append(features)
            adversarial_labels.append(labels)
        adversarial_features = np.vstack(adversarial_features)
        adversarial_labels = np.vstack(adversarial_labels)
    else:
        adversarial_features = []
        adversarial_labels = []

    return benign_features, benign_labels, adversarial_features, adversarial_labels