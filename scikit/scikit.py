import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('../data/hello_world_dataset.csv')

# Convert greetings to numerical labels
greetings = df['Greeting'].unique()
greeting_to_label = {greeting: label for label, greeting in enumerate(greetings)}
df['Label'] = df['Greeting'].map(greeting_to_label)

# Split data into features and labels
X = df['Greeting'].values.reshape(-1, 1)  # type: ignore # Features (reshape for OneHotEncoder)
y = df['Label'].values                     # Labels

# One-hot encode the features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Define a simple neural network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.output = self.softmax(self.output_input)
        return self.output

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss (cross-entropy)
            m = y.shape[0]
            log_probs = -np.log(output[range(m), y])
            loss = np.sum(log_probs) / m

            # Backpropagation
            d_output = output
            d_output[range(m), y] -= 1
            d_output /= m

            d_hidden = np.dot(d_output, self.weights2.T) * (self.hidden_output * (1 - self.hidden_output))

            # Update weights and biases
            self.weights2 -= learning_rate * np.dot(self.hidden_output.T, d_output)
            self.bias2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.weights1 -= learning_rate * np.dot(X.T, d_hidden)
            self.bias1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Create and train the neural network
input_size = X_encoded.shape[1]  # Number of input features (one-hot encoded)
output_size = len(greetings)     # Number of unique greetings
nn = SimpleNeuralNetwork(input_size, hidden_size=10, output_size=output_size)
nn.train(X_encoded, y, epochs=1000, learning_rate=0.1)

# Example prediction
new_greeting = "Hello, World!"
new_greeting_encoded = encoder.transform(np.array([new_greeting]).reshape(-1, 1))
predicted_label = nn.predict(new_greeting_encoded)
predicted_greeting = greetings[predicted_label[0]]

print(f"Predicted greeting: {predicted_greeting}")