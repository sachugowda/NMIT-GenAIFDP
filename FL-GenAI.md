
---

# Tutorial: Federated Learning with a Variational Autoencoder (VAE) on the MNIST Dataset

This tutorial demonstrates how to simulate federated learning using TensorFlow, where multiple client models are trained separately on a Variational Autoencoder (VAE) and then aggregated on a central server model.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Preprocess Data](#setup-and-preprocess-data)
3. [Define the VAE Model](#define-the-vae-model)
4. [Helper Function for Federated Averaging](#helper-function-for-federated-averaging)
5. [Simulate Federated Learning](#simulate-federated-learning)
6. [Generate and Visualize New Images](#generate-and-visualize-new-images)
7. [Complete Code](#complete-code)

---

## Overview

Federated learning is a distributed approach to training machine learning models, where data remains on client devices (like phones or IoT devices), and only model updates are shared with a central server. Here, we simulate federated learning by:

- Splitting the MNIST dataset across two clients.
- Training a VAE model on each client's data.
- Aggregating client models on a central server.

---

## Setup and Preprocess Data

```python
import tensorflow as tf
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
```

- **Load Data**: We load the MNIST dataset, which contains grayscale images of handwritten digits.
- **Normalize**: Rescale pixel values to [0, 1].
- **Flatten**: Reshape each `28x28` image into a 784-dimensional vector.

### Split Data for Clients

```python
# Split data into two client datasets
client_1_data = x_train[:30000]
client_2_data = x_train[30000:60000]

# Define a preprocessing function for batching
def preprocess(dataset):
    return tf.data.Dataset.from_tensor_slices((dataset, dataset)).batch(32)

# Prepare the data for each client
client_1_data = preprocess(client_1_data)
client_2_data = preprocess(client_2_data)
test_data = preprocess(x_test)
```

1. **Client Splits**: Divide the dataset into two non-overlapping subsets to simulate two clients.
2. **Batching**: Use `tf.data.Dataset` to batch the data for efficient processing during training.

---

## Define the VAE Model

A Variational Autoencoder (VAE) consists of three main components:

1. **Encoder**: Encodes the input into a latent distribution.
2. **Sampling Layer**: Samples from the latent distribution using the reparameterization trick.
3. **Decoder**: Decodes the latent representation back into the original input space.

```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.log_var = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
```

- **Layers**: The encoder has two dense layers, followed by two outputs: `mean` and `log_var` which parameterize the latent distribution.

```python
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * epsilon
```

- **Sampling**: The reparameterization trick enables gradient-based optimization by sampling from a distribution defined by `mean` and `log_var`.

```python
class Decoder(tf.keras.layers.Layer):
    def __init__(self, original_dim):
        super(Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.output_layer = tf.keras.layers.Dense(original_dim, activation="sigmoid")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
```

- **Decoder**: Takes a sampled latent vector and reconstructs the original input.

```python
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(original_dim)
        self.original_dim = original_dim

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(x, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
```

- **Custom `train_step`**: Calculates both the reconstruction loss and KL divergence, using them to compute the total loss.

---

## Helper Function for Federated Averaging

This function averages model weights from multiple clients:

```python
def average_weights(weight_list):
    avg_weights = copy.deepcopy(weight_list[0])
    for i in range(len(avg_weights)):
        for j in range(1, len(weight_list)):
            avg_weights[i] += weight_list[j][i]
        avg_weights[i] = avg_weights[i] / len(weight_list)
    return avg_weights
```

- **Purpose**: Computes the element-wise average of each weight tensor from multiple client models.

---

## Simulate Federated Learning

1. **Server Initialization**: Initialize the central model that will aggregate weights.
2. **Client Training**: Each client trains locally and sends updated weights to the server.
3. **Weight Aggregation**: The server updates its weights by averaging client weights.

```python
server_model = VAE(original_dim, latent_dim)
server_model(tf.zeros((1, original_dim)))  # Initialize weights
server_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

for round_num in range(1, num_rounds + 1):
    print(f"Round {round_num}")
    client_1_model = VAE(original_dim, latent_dim)
    client_1_model(tf.zeros((1, original_dim)))
    client_1_model.set_weights(server_model.get_weights())
    client_1_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    client_2_model = VAE(original_dim, latent_dim)
    client_2_model(tf.zeros((1, original_dim)))
    client_2_model.set_weights(server_model.get_weights())
    client_2_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    client_1_model.fit(client_1_data, epochs=1, verbose=0)
    client_2_model.fit(client_2_data, epochs=1, verbose=0)

    client_1_weights = client_1_model.get_weights()
    client_2_weights = client_2_model.get_weights()
    averaged_weights = average_weights([client_1_weights, client_2_weights])
    server_model.set_weights(averaged_weights)
    test_loss = server_model.evaluate(test_data, verbose=0)
    print(f"Test Loss after round {round_num}: {test_loss}")
```

---

## Generate and Visualize New Images

The trained VAE is used to generate images by sampling random points from the latent space.

```python
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
```

- **Image Generation**: Random latent vectors are decoded to generate synthetic digit images.

---

## Complete Code

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy

# Set up latent dimension and other parameters
latent_dim = 2  # Dimension of the latent space for VAE
num_rounds = 10  # Number of federated learning rounds
original_dim = 784  # MNIST images are 28x28 pixels, flattened

# Step 1: Load and Preprocess Data

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Split data into two client datasets
client_1_data = x_train[:30000]
client_2_data = x_train[30000:60000]

# Define a preprocessing function for batching
def preprocess(dataset):
    return tf.data.Dataset.from_tensor_slices((dataset, dataset)).batch(32)

# Prepare the data for each client
client_1_data = preprocess(client_1_data)
client_2_data = preprocess(client_2_data)
test_data = preprocess(x_test)

# Step 2: Define the VAE Model

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.log_var = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * epsilon

class Decoder(tf.keras.layers.Layer):
    def __init__(self, original_dim):
        super(Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.output_layer = tf.keras.layers.Dense(original_dim, activation="sigmoid")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(original_dim)
        self.original_dim = original_dim

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = self.sampling((mean, log_var))
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(x, training=True)
            # Calculate per-sample reconstruction loss (sum over pixel dimensions)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=1)
            )
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))
            # Total loss
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

# Step 3: Helper Function for Federated Averaging

def average_weights(weight_list):
    """Averages a list of weight sets (from different models)."""
    avg_weights = copy.deepcopy(weight_list[0])
    for i in range(len(avg_weights)):
        for j in range(1, len(weight_list)):
            avg_weights[i] += weight_list[j][i]
        avg_weights[i] = avg_weights[i] / len(weight_list)
    return avg_weights

# Step 4: Simulate Federated Learning

# Initialize a "server" model
server_model = VAE(original_dim, latent_dim)
# Build the model to initialize weights
server_model(tf.zeros((1, original_dim)))
server_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Run federated training for multiple rounds
for round_num in range(1, num_rounds + 1):
    print(f"Round {round_num}")

    # Initialize and build models for each client with server weights
    client_1_model = VAE(original_dim, latent_dim)
    client_1_model(tf.zeros((1, original_dim)))  # Build model to initialize weights
    client_1_model.set_weights(server_model.get_weights())
    client_1_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    client_2_model = VAE(original_dim, latent_dim)
    client_2_model(tf.zeros((1, original_dim)))  # Build model to initialize weights
    client_2_model.set_weights(server_model.get_weights())
    client_2_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # Train each client on its local data
    client_1_model.fit(client_1_data, epochs=1, verbose=0)
    client_2_model.fit(client_2_data, epochs=1, verbose=0)

    # Collect weights from each client
    client_1_weights = client_1_model.get_weights()
    client_2_weights = client_2_model.get_weights()

    # Average the client weights and update the server model
    averaged_weights = average_weights([client_1_weights, client_2_weights])
    server_model.set_weights(averaged_weights)

    # Evaluate the server model on the test set after each round
    test_loss = server_model.evaluate(test_data, verbose=0)
    print(f"Test Loss after round {round_num}: {test_loss}")

# Step 5: Generate and Visualize New Images

# Helper function to generate images
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

# Generate and visualize new images from the trained server model
print("Generated images from trained model:")
generate_and_plot_digits(server_model)

# Additional Evaluation - Reconstruction Loss on Test Set
test_loss = server_model.evaluate(x_test, x_test)
print(f"Final Reconstruction Loss on Test Set: {test_loss}")

```

This tutorial gives a detailed explanation for each section of the code, enabling you to understand the workings of federated

 learning with a VAE model.
