
---

# Tutorial: Federated Learning with Image Classification Using TensorFlow and MNIST

In this tutorial, we’ll introduce federated learning, build a simple neural network for image classification, and simulate federated training using TensorFlow with two clients and ten communication rounds. The dataset is split between two clients, and they train locally, sharing only model updates with a central server.

## Table of Contents

1. [What is Federated Learning?](#what-is-federated-learning)
2. [Federated Learning Setup with MNIST](#federated-learning-setup-with-mnist)
3. [Building and Distributing the Model](#building-and-distributing-the-model)
4. [Federated Training with Multiple Rounds](#federated-training-with-multiple-rounds)
5. [Complete Code](#complete-code)

---

## What is Federated Learning?

**Federated Learning (FL)** is a decentralized approach to machine learning where multiple clients (e.g., phones, IoT devices) train a shared model on their local data without sharing that data with a central server. Instead, they only share model updates, which are aggregated on a central server.

### Key Characteristics of Federated Learning:

- **Data Privacy**: Clients keep their data locally, which can help with data privacy concerns.
- **Decentralized Training**: Model training is performed locally on each client, with only the model updates being sent to the server.
- **Communication Rounds**: The model is trained over several rounds, where clients train locally and send updates to the server to be averaged and redistributed.

In this tutorial, we’ll simulate federated learning using two clients and ten communication rounds, training a model to classify MNIST digits.

---

## Federated Learning Setup with MNIST

To simulate federated learning, we’ll:
1. Split the MNIST dataset between two clients.
2. Each client trains its model on its local data.
3. After each communication round, the server aggregates the client models’ weights and updates the global model.
4. The global model is then sent back to each client for the next round.

---

## Building and Distributing the Model

We’ll use the same model structure as in centralized training:
1. **Input Layer**: Flatten the 28x28 image to a 784-dimensional vector.
2. **Hidden Layers**: Two dense layers with ReLU activations.
3. **Output Layer**: A dense layer with softmax activation for 10 classes.

---

## Federated Training with Multiple Rounds

In federated training, we:
1. Initialize a global model on the server.
2. For each communication round:
   - Send the global model weights to each client.
   - Each client trains on its own local data.
   - Clients send their updated weights back to the server.
   - The server averages the weights and updates the global model.

---

## Complete Code

Here’s the full code to simulate federated learning with two clients and ten communication rounds on the MNIST dataset.

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import copy

# Step 1: Load and Preprocess the MNIST Dataset

# Load MNIST data from TensorFlow datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1 by dividing by 255.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split data for two clients
client_1_data = (x_train[:30000], y_train[:30000])
client_2_data = (x_train[30000:], y_train[30000:])

# Step 2: Define the Neural Network Model Function

def create_model():
    # Initialize a Sequential model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),   # Flatten layer
        layers.Dense(128, activation='relu'),   # First hidden layer
        layers.Dense(64, activation='relu'),    # Second hidden layer
        layers.Dense(10, activation='softmax')  # Output layer
    ])
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Federated Averaging Function

def federated_average(weight_list):
    """Average the weights from two clients."""
    avg_weights = copy.deepcopy(weight_list[0])
    for i in range(len(avg_weights)):
        avg_weights[i] = (weight_list[0][i] + weight_list[1][i]) / 2
    return avg_weights

# Step 4: Federated Training Simulation

num_rounds = 10  # Number of federated learning rounds

# Initialize a global model
global_model = create_model()

# Perform federated training over multiple rounds
for round_num in range(num_rounds):
    print(f"\nCommunication Round {round_num + 1}")

    # Create two copies of the model for each client
    client_1_model = create_model()
    client_2_model = create_model()

    # Set the weights of each client model to the global model's weights
    client_1_model.set_weights(global_model.get_weights())
    client_2_model.set_weights(global_model.get_weights())

    # Train each client's model on their local data
    client_1_model.fit(client_1_data[0], client_1_data[1], epochs=1, batch_size=32, verbose=0)
    client_2_model.fit(client_2_data[0], client_2_data[1], epochs=1, batch_size=32, verbose=0)

    # Collect the weights from each client model
    client_1_weights = client_1_model.get_weights()
    client_2_weights = client_2_model.get_weights()

    # Federated Averaging: Average the weights from both clients
    averaged_weights = federated_average([client_1_weights, client_2_weights])

    # Update the global model with the averaged weights
    global_model.set_weights(averaged_weights)

    # Evaluate the global model on the test data after each round
    test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Round {round_num + 1} - Test accuracy: {test_accuracy:.4f}")

# Final evaluation of the global model
final_loss, final_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy after Federated Training: {final_accuracy:.4f}")
```

---

## Explanation of the Code

1. **Loading and Preprocessing the Data**:
   - The MNIST dataset is loaded, normalized, and split into two parts for the two clients.

2. **Model Definition**:
   - The `create_model` function defines a simple neural network for image classification, compiled with the Adam optimizer and sparse categorical crossentropy loss.

3. **Federated Averaging**:
   - The `federated_average` function averages the weights from two clients by taking the element-wise mean of the weights.

4. **Federated Training Simulation**:
   - A global model is initialized on the server.
   - For each communication round:
     - Two client models are created, each initialized with the global model’s weights.
     - Each client trains locally on its own dataset for one epoch.
     - The weights of both client models are averaged, and the global model is updated with these averaged weights.
     - The global model’s performance is evaluated on the test set after each round.

5. **Final Evaluation**:
   - After all communication rounds, the final global model is evaluated on the test set to determine its overall accuracy.

---

## Summary

In this tutorial, we simulated federated learning for image classification using TensorFlow and the MNIST dataset. We used two clients and ten communication rounds to train a shared model while keeping data decentralized. Here’s what we covered:

- **Federated Learning Concept**: Clients train locally on their data, sending only model updates to a central server.
- **Federated Averaging**: Used to aggregate the weights from clients to update the global model.
- **Communication Rounds**: Repeatedly train clients and update the global model over multiple rounds to improve accuracy.

**Key Takeaway**: Federated learning allows for decentralized training, which can enhance data privacy by keeping the data local to each client while still achieving a shared global model.

This simulation provides a foundation for understanding federated learning in a simple setting. You can extend this by adding more clients, increasing the number of communication rounds, or using different model architectures.
