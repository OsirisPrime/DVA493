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

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize network structure
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Calculate the error term of the output layer
    def error_term_output_layer(self, output, target):
        return output * (1 - output) * (target - output)

    # Calculate the error term of the hidden layer
    def error_term_hidden_layer(self, output, errorTerms, weights):
        weighted_sum = 0
        for i in range(len(weights)):
            weighted_sum += errorTerms[i] * weights[i]
        return (1 - output) * output * weighted_sum

    # Feed the patient data (inputs) into the model
    def feedforward(self, data):
        # Input to hidden layer
        self.hidden_activation = np.dot(data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)

        # Hidden to output layer
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    # Update the weight and biases
    def backward(self, data, target):
        # Compute the output layer error term
        output_error_term = self.error_term_output_layer(self.predicted_output, target)

        hidden_error_terms = []
        for i in range(self.hidden_size):
            hidden_error = self.error_term_hidden_layer(
                self.hidden_output[0, i],
                output_error_term[0],
                self.weights_hidden_output[i]
            )
            hidden_error_terms.append(hidden_error)

        # Reshape the array
        hidden_error_terms = np.array(hidden_error_terms).reshape(1, -1)

        # Update output weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, output_error_term)
        self.bias_output += self.learning_rate * output_error_term

        # Update hidden weights and biases
        self.weights_input_hidden += self.learning_rate * np.dot(data.T, hidden_error_terms)
        self.bias_hidden += self.learning_rate * hidden_error_terms

    # Train the model
    def train(self, data, target, validation_set, validation_result, epochs):
        # The loss
        self.training_loss = []
        self.validation_loss = []

        # Train the model
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(data)):
                # Get a single patient's data and result
                patient_data = data[i].reshape(1, -1)
                patient_result = target[i].reshape(1, -1)

                # Feedforward the patient data
                self.feedforward(patient_data)

                # Compute the loss
                loss = np.mean(np.square(patient_result - self.predicted_output))
                total_loss += loss

                # Backward to update the weights and biases
                self.backward(patient_data, patient_result)

            # Save average training loss after each epoch
            self.training_loss.append(total_loss / len(data))

            # Validate the model after every epoch
            validation_predictions = self.feedforward(validation_set)
            validation_loss = np.mean(np.square(validation_result - validation_predictions))
            self.validation_loss.append(validation_loss)

            # Print the average loss for each epoch
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Average Training Loss: {total_loss / len(data):.4f}, Total Validation Loss: {validation_loss:.4f}")

        return self.training_loss, self.validation_loss

    # Calculate the finished model accuracy
    def testing(self, test_data, test_result):
        # Feedforward to get predictions
        predictions = self.feedforward(test_data)

        # Round the predictions to 0 or 1
        predicted_classes = np.round(predictions).astype(int)
        actual_classes = test_result.astype(int)

        # Calculate the accuracy
        correct_predictions = np.sum(predicted_classes == actual_classes)
        accuracy = correct_predictions / len(test_result)
        print(f"Validation Accuracy: {accuracy:.4f}")

#---------------------------------------------------------------------------------------#

# Load data
patient_data, patient_result = import_file()

# Define network structure
input_size = 19
hidden_size = 10
output_size = 1
learning_rate = 0.1
epochs = 500

# Normalize patient data
scaler = StandardScaler()
patient_data = scaler.fit_transform(patient_data)

# Split the data into training, validation and testing set
split_idx1 = int(0.75 * len(patient_data))
split_idx2 = int(0.875 * len(patient_data))

training_data_set = patient_data[:split_idx1]
training_result_set = patient_result[:split_idx1]
validation_data_set = patient_data[split_idx1:split_idx2]
validation_result_set = patient_result[split_idx1:split_idx2]
test_data_set = patient_data[split_idx2:]
test_result_set = patient_result[split_idx2:]

# Initialize the model
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# Train the model
training_loss, validation_loss = nn.train(training_data_set, training_result_set, validation_data_set, validation_result_set, epochs)

# Test the finished model
nn.testing(test_data_set, test_result_set)

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss', color='red')
plt.plot(validation_loss, label='Validation Loss', color='blue', linestyle='--')
plt.title("Training and Validation Loss Over Epochs", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
