import datetime
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

def print_step(message):
    print(f"\n\n##################################\n##################################\n===> {message}\n##################################\n##################################\n\n")


class GAN():
    def __init__(self, input_shape, input_latent_dim, G_data, D_data, image_path):
        """
        :param input_shape: Shape of the input image
        :param input_latent_dim: Shape of the input noise for G, should be 1-D array
        :param G_data: Generator training datasets, should be numpy array
        :param D_data: Discriminator training datasets, should be numpy array
        :param image_path: Image save path during training
        """

        #self.strategy = tf.distribute.MirroredStrategy()

        self.img_shape = input_shape
        self.latent_dim = input_latent_dim
        self.G_datasets = G_data
        self.D_datasets = D_data
        self.image_path = image_path
        self.log = []

        #with self.strategy.scope():
        optimizer = tf.keras.optimizers.legacy.Adam(0.00001, 0.5)
        discriminator_optimizer = tf.keras.optimizers.legacy.Adam(0.000002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                optimizer=discriminator_optimizer,
                                metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates images
        z = Input(shape=(self.latent_dim,))
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        reconstructed_z = self.generator(z)
        validity = frozen_D(reconstructed_z)

        # The discriminator takes generated images as input and determines validity
        self.discriminator.trainable = False

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, [reconstructed_z, validity])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                            loss_weights=[0.999, 0.001],
                            optimizer=optimizer)

    def build_generator(self, name="generator"):
        print_step("Building generator")
        model = tf.keras.Sequential(name=name)
        
        model.add(tf.keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

        model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(tf.keras.layers.Dense(self.img_shape))

        model.summary()

        noise = tf.keras.Input(shape=(self.latent_dim,))
        img = model(noise)

        return tf.keras.Model(noise, img)

    def build_discriminator(self, name="discriminator"):
        print_step("Building discriminator")

        model = tf.keras.Sequential(name=name)
        print_step(f"Input shape: {self.img_shape}")
        #model.add(tf.keras.layers.Flatten(input_shape=self.img_shape))
        model.add(tf.keras.layers.Dense(512, input_dim=self.img_shape))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        img = tf.keras.Input(shape=self.img_shape)
        validity = model(img)

        return tf.keras.Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, rescale=False, expand_dims=True):
        """
        :param epochs: Number of training iterations
        :param batch_size: Batch size
        :param sample_interval: Print the loss of G and D every sample_interval
        :param rescale: If True, rescale D_img to [-1,1]
        :param expand_dims: If True, expand image channel.
        :return: None
        """

        # Load the dataset
        D_train = self.D_datasets
        G_train = self.G_datasets

        if rescale:
            # Rescale to -1 to 1
            D_train = D_train / 127.5 - 1.
        if expand_dims:
            D_train = np.expand_dims(D_train, axis=-1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, D_train.shape[0], batch_size)
            D_imgs = D_train[idx]  # targeted feature
            G_feature = G_train[idx]  # input feature

            noise_add = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = G_feature  # + noise_add

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(D_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator to fool the discriminator
            g_loss = self.combined.train_on_batch(noise, [D_imgs, valid])


            #print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f}]")
            # Print the progress
            if epoch % sample_interval == 0:
                message = f"{datetime.datetime.now()} - {epoch}/{epochs} [D loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.2f}] [G loss: {g_loss[0]:.5f}, mse: {g_loss[1]:.5f}]"
                self.log.append([epoch, d_loss[0], d_loss[1], g_loss[0], g_loss[1]])
                self.create_str_to_txt('gan', datetime.datetime.now().strftime('%Y-%m-%d'), message)
                print(message)

    def showlogs(self, path):
        logs = np.array(self.log)
        names = ["d_loss", "d_acc", "g_loss", "g_mse"]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(logs[:, 0], logs[:, i + 1])
            plt.xlabel("iteration")
            plt.ylabel(names[i])
            plt.grid()
        plt.tight_layout()
        plt.savefig(path + ".png")
        plt.close()
        np.save(path + ".npy", logs)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)
        fig.savefig(os.path.join(self.image_path, f"{epoch}.png"))
        plt.close()

    def save_model(self, path):
        self.combined.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.combined = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def get_generator(self):
        return self.generator

    def calculateMSE(self, Y, Y_hat):
        """
        Calculate Mean Squared Error (MSE) and R-squared (R²)
        """
        MSE = np.mean(np.power((Y - Y_hat), 2))
        R2 = 1 - MSE / np.var(Y)
        return MSE, R2

    def create_str_to_txt(self, model_name, date, str_data):
        """
        Create a txt file and append the given data to it.
        """
        directory = f'./logs/training/{model_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_file_name = f'{directory}/{model_name}_{date}.txt'

        with open(path_file_name, "a") as f:
            f.write(str_data + '\n')
        print(f"Data saved to {path_file_name}")
