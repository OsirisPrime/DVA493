import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import datasets


# Import data and sort it
def import_file():

    # Open and read txt file
    with open('Diabetic.txt', "r") as file:
        patients = file.read().splitlines()

    # Remove the first 24 lines
    for i in range(0, 24):
        patients.pop(0)

    patient_attr = []

    # Split all 19 attributes by ","
    for i, patent in enumerate(patients):
        patient_attr.append(patent.split(','))

    # Take the first 18 attributes as training data
    training_data = [attribute[0:18] for attribute in patient_attr]
    training_data = np.array(training_data, dtype=float)

    # Take the last attribute as training result
    training_result = [[attribute[-1] for attribute in patient_attr]][0]
    training_result = np.array(training_result, dtype=float)

    return training_data, training_result
#---------------------------------------------------------------------------------------#
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize network structure
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def feedforward(self, training_data):
        # Forward pass through the network layers
        self.input = training_data

        # Hidden layer
        self.hidden_input = np.dot(training_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backpropagate(self, training_data, training_result):
        # Calculate the error in output
        error_output = training_result - self.output
        d_output = error_output * sigmoid_derivative(self.output)

        # Backpropagate the error to the hidden layer
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += training_data.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, training_data, training_result, epochs):
        for epoch in range(epochs):
            self.feedforward(training_data)
            self.backpropagate(training_data, training_result)
            if epoch % 100 == 0:
                loss = np.mean(np.square(training_result - self.output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

#---------------------------------------------------------------------------------------#

training_data, training_result = import_file()

training_result_one_hot = np.zeros((training_result.size, 2))
training_result_one_hot[np.arange(training_result.size), training_result.astype(int)] = 1

input_size = 18
hidden_size = 2
output_size = 2

training_data_set = training_data[0:863]
training_result_set = training_result_one_hot[0:863]
validation_set = training_data[864:978]
validation_result_set = training_result_one_hot[864:978]
test_set = training_data[979:1151]
test_result_set = training_result_one_hot[979:1151]

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)

nn.train(training_data_set, training_result_set, epochs=864)

test_input = test_set[0]
print("Test output:", nn.feedforward(test_input))
print("Real output:", test_result_set[0])
#---------------------------------------------------------------------------------------#