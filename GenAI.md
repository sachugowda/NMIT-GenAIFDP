
---

# Tutorial: Building a Simple Generative AI Model Using Variational Autoencoder (VAE) on MNIST Data

This tutorial guides you through building a basic Variational Autoencoder (VAE) using TensorFlow to generate new handwritten digits based on the MNIST dataset. VAEs are a type of generative model that can create realistic data samples by learning a compressed latent representation of the data.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Preprocess Data](#setup-and-preprocess-data)
4. [Define the VAE Model](#define-the-vae-model)
5. [Compile and Train the VAE](#compile-and-train-the-vae)
6. [Generate and Visualize New Images](#generate-and-visualize-new-images)
7. [Complete Code](#complete-code)

---

## Overview

Variational Autoencoders (VAEs) use a probabilistic approach to learn a compressed representation of data. VAEs consist of three main components:

1. **Encoder**: Encodes the input data into a latent representation.
2. **Sampling Layer**: Samples from the latent space to generate variations.
3. **Decoder**: Reconstructs the data from the sampled latent representation.

In this example, we use the MNIST dataset of handwritten digits to train a VAE model that can generate new digit images.

---

## Requirements

To run this example, you need TensorFlow installed. Install it using the following command:

```bash
pip install tensorflow
```

---

## Setup and Preprocess Data

1. **Data Loading**: We use TensorFlow to load the MNIST dataset, which consists of grayscale images of handwritten digits.
2. **Normalization**: Scale pixel values to [0, 1] by dividing by 255.
3. **Flattening**: Reshape each `28x28` image into a 784-dimensional vector required by the VAE.

```python
import tensorflow as tf
import numpy as np

# Load MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize data to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images to 784 dimensions for the encoder input
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
```

---

## Define the VAE Model

A Variational Autoencoder (VAE) consists of three main components:

1. **Encoder**: Encodes the input image into a latent distribution represented by mean and log variance.
2. **Sampling Layer**: Uses the reparameterization trick to sample from the latent distribution, allowing gradients to flow through the model.
3. **Decoder**: Decodes the sampled latent vector back into the original input space.

### Encoder

The encoder maps the input image to a latent distribution with mean and log variance.

```python
from tensorflow.keras import layers, Model

class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu")
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
```

- **Layers**: The encoder has two dense layers, followed by outputs for the `mean` and `log_var` that define the latent distribution.

### Sampling Layer

The sampling layer applies the reparameterization trick, which is essential to make the VAE differentiable.

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * epsilon
```

- **Sampling**: Generates a sample from the latent space by applying the reparameterization trick, `z = mean + exp(log_var / 2) * epsilon`.

### Decoder

The decoder maps the sampled latent vector back to the original input space (784-dimensional flattened image).

```python
class Decoder(Model):
    def __init__(self, original_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(256, activation="relu")
        self.output_layer = layers.Dense(original_dim, activation="sigmoid")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
```

- **Decoder**: Decodes the latent vector back into the 784-dimensional image space.

### Complete VAE Model

The VAE class combines the encoder, sampling layer, and decoder. During training, it calculates the KL divergence as part of the loss to ensure the latent distribution remains close to a normal distribution.

```python
class VAE(Model):
    def __init__(self, original_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(original_dim)

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder(z)
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        self.add_loss(kl_loss)
        return reconstructed
```

- **KL Divergence**: Regularizes the latent space to approximate a standard normal distribution.

---

## Compile and Train the VAE

Now that we have defined our VAE, we can compile it with an optimizer and a loss function and then train it on the MNIST dataset.

```python
original_dim = 784
latent_dim = 2
vae = VAE(original_dim, latent_dim)

# Compile with a loss function and optimizer
vae.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# Train the VAE
vae.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))
```

- **Mean Squared Error Loss**: Measures the difference between the input and the reconstructed output.
- **Training**: We train the VAE for 20 epochs with a batch size of 128, using the MNIST images as both input and output for reconstruction.

---

## Generate and Visualize New Images

To generate new images, we can sample random points from the latent space and decode them.

```python
import matplotlib.pyplot as plt

def generate_and_plot_digits(model, num_images=10):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = model.decoder(random_latent_vectors)
    generated_images = generated_images.numpy().reshape(num_images, 28, 28)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i], cmap="gray")
        plt.axis("off")
    plt.show()

generate_and_plot_digits(vae)
```

- **Image Generation**: We sample random latent vectors and pass them through the decoder to generate synthetic images of handwritten digits.

---

## Complete Code

```python
# Install TensorFlow
# pip install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize data to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images to 784 dimensions for the encoder input
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

latent_dim = 2  # Size of the latent space

class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu")
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * epsilon

class Decoder(Model):
    def __init__(self, original_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(

256, activation="relu")
        self.output_layer = layers.Dense(original_dim, activation="sigmoid")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class VAE(Model):
    def __init__(self, original_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(original_dim)

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        self.add_loss(kl_loss)
        return reconstructed

original_dim = 784
vae = VAE(original_dim, latent_dim)

# Compile with a loss function and optimizer
vae.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# Train the VAE
vae.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))

# Generate new images by sampling random points in the latent space
def generate_and_plot_digits(model, num_images=10):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = model.decoder(random_latent_vectors)
    generated_images = generated_images.numpy().reshape(num_images, 28, 28)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i], cmap="gray")
        plt.axis("off")
    plt.show()

generate_and_plot_digits(vae)
```

This tutorial walks you through a simple example of a Variational Autoencoder to generate new handwritten digits, using the MNIST dataset and TensorFlow.
