import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import data and sort it
def import_file():

    # Open and read txt file
    with open('Diabetic.txt', "r") as file:
        patients = file.read().splitlines()

    # Remove the first 24 lines
    patients = patients[24:]

    # Split all 19 attributes by ","
    patient_attr = [patient.split(',') for patient in patients]

    # Take the first 19 attributes as training data
    patient_data = np.array([attribute[0:19] for attribute in patient_attr], dtype=float)

    # Take the last attribute as training result
    patient_result = np.array([attribute[-1] for attribute in patient_attr], dtype=float)

    return patient_data, patient_result
#---------------------------------------------------------------------------------------#
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
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
        training_losses = []

        for epoch in range(epochs):
            self.feedforward(training_data)
            self.backpropagate(training_data, training_result)

            # Calculate the training loss
            loss = np.mean(np.square(training_result - self.output))
            training_losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return training_losses

    def validate(self, validation_set, validation_result):
        # Feedforward to get predictions
        prediction = self.feedforward(validation_set)
        predicted_classes = np.argmax(prediction, axis=1)
        actual_classes = np.argmax(validation_result, axis=1)

        # Calculate accuracy
        correct_predictions = np.sum(predicted_classes == actual_classes)
        accuracy = correct_predictions / len(validation_result)
        print(f"Validation Accuracy: {accuracy:.4f}")

    def plot_progress(self, training_losses):
        epochs = range(len(training_losses))

        # Plot training loss
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, training_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')

        plt.tight_layout()
        plt.show()

    def testing(self, test_data, test_result):
        accuracy = []
        for sample in range(len(test_data)):

            test = np.round(1)
            if test == test_result[sample]:
                accuracy.append(1)
            else:
                accuracy.append(0)

        numAccurate = 0
        for i in range(len(accuracy)):
            if accuracy[i] == 1:
                numAccurate += 1

        print('Accuracy:', numAccurate / len(accuracy))

#---------------------------------------------------------------------------------------#

# Load data
patient_data, patient_result = import_file()

patient_result_one_hot = np.zeros((patient_result.size, 2))
patient_result_one_hot[np.arange(patient_result.size), patient_result.astype(int)] = 1

# Define network structure
input_size = 19
hidden_size = 2
output_size = 2
learning_rate = 0.1
epochs = 500

# Split the data into training, validation and testing
split_idx1 = int(0.75 * len(patient_data))
split_idx2 = int(0.875 * len(patient_data))

training_data_set = patient_data[:split_idx1]
training_result_set = patient_result_one_hot[:split_idx1]

validation_data_set = patient_data[split_idx1:split_idx2]
validation_result_set = patient_result_one_hot[split_idx1:split_idx2]

test_data_set = patient_data[split_idx2:]
test_result_set = patient_result_one_hot[split_idx2:]


# Normalize the data
scaler = StandardScaler()
training_data_set = scaler.fit_transform(training_data_set)
validation_set = scaler.transform(validation_data_set)
test_data_set = scaler.transform(test_data_set)


# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
training_losses = nn.train(training_data_set, training_result_set, epochs)

# Validate the model
nn.validate(validation_data_set, validation_result_set)

# Plot the training and validation progress
nn.plot_progress(training_losses)

nn.testing(test_data_set, test_result_set)

#---------------------------------------------------------------------------------------#
