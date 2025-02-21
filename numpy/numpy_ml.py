import numpy as np


# Input features (4 samples, 2 features each)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Labels (XOR of the inputs)
y = np.array([[0], [1], [1], [0]])



# Define the neural network
class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights and biases randomly
        np.random.seed(42)  # For reproducibility
        self.weights1 = np.random.randn(2, 2)  # Input to hidden layer weights
        self.bias1 = np.random.randn(1, 2)     # Hidden layer bias
        self.weights2 = np.random.randn(2, 1)  # Hidden to output layer weights
        self.bias2 = np.random.randn(1, 1)     # Output layer bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.output = self.sigmoid(self.output_input)
        return self.output

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backpropagation
            error = y - output
            d_output = error * self.sigmoid_derivative(output)
            error_hidden = d_output.dot(self.weights2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

            # Update weights and biases
            self.weights2 += self.hidden_output.T.dot(d_output) * learning_rate
            self.bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.weights1 += X.T.dot(d_hidden) * learning_rate
            self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

            if epoch % 1000 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))



# Create an instance of the neural network
nn = SimpleNeuralNetwork()

# Train the network
nn.train(X, y, epochs=10000, learning_rate=0.1)        



# Make predictions
predictions = nn.predict(X)
print("Predictions:")
print(predictions)