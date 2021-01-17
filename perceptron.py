#From tutorial video: https://www.youtube.com/watch?v=kft1AJ9WVDk&ab_channel=Polycode

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array(
    [
        [0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]
    ]
)

training_outputs = np.array([[0,1,1,0]]).T #transposes it to become a 4 by 1 matrix

#assigning random values as weights but to ensure those random values are the same every time
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for iteration in range(5000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("\nSynaptic weights after training: ")
print(synaptic_weights)

print("\nOutputs after training: ")
print(outputs)