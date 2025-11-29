"""
MNIST Digit Classifier - Neural Network Implementation
Built from scratch using only NumPy for mathematical operations.
"""

import numpy as np
import pickle


class NeuralNetwork:
    """
    3-layer neural network for MNIST digit classification.
    Architecture: 784 (input) -> 128 (hidden1) -> 64 (hidden2) -> 10 (output)
    """
    
    def __init__(self, layer_sizes=[784, 128, 64, 10], learning_rate=0.01):
        """
        Initialize neural network with random weights.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, hidden2, output]
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases with He initialization
        self.weights = []
        self.biases = []
        
        np.random.seed(42)
        for i in range(self.num_layers - 1):
            # He initialization for ReLU activation
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Cache for backpropagation
        self.cache = {}
        
    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation for output layer."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, 784)
            
        Returns:
            Output probabilities of shape (batch_size, 10)
        """
        self.cache['A0'] = X
        
        # Forward through hidden layers with ReLU
        for i in range(self.num_layers - 2):
            Z = np.dot(self.cache[f'A{i}'], self.weights[i]) + self.biases[i]
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = self.relu(Z)
        
        # Output layer with softmax
        i = self.num_layers - 2
        Z = np.dot(self.cache[f'A{i}'], self.weights[i]) + self.biases[i]
        self.cache[f'Z{i+1}'] = Z
        self.cache[f'A{i+1}'] = self.softmax(Z)
        
        return self.cache[f'A{self.num_layers - 1}']
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (batch_size, 10)
            y_true: True labels as one-hot vectors (batch_size, 10)
            
        Returns:
            Average cross-entropy loss
        """
        batch_size = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size
        return loss
    
    def backward(self, y_true):
        """
        Backpropagation to compute gradients.
        
        Args:
            y_true: True labels as one-hot vectors (batch_size, 10)
        """
        batch_size = y_true.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient (softmax + cross-entropy)
        dA = self.cache[f'A{self.num_layers - 1}'] - y_true
        
        # Backpropagate through all layers
        for i in range(self.num_layers - 2, -1, -1):
            # Gradient w.r.t weights and biases
            dW = np.dot(self.cache[f'A{i}'].T, dA) / batch_size
            dB = np.sum(dA, axis=0, keepdims=True) / batch_size
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
            
            # Gradient w.r.t previous layer (if not input layer)
            if i > 0:
                dA = np.dot(dA, self.weights[i].T)
                # Apply ReLU derivative
                dA = dA * self.relu_derivative(self.cache[f'Z{i}'])
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """
        Update weights using gradient descent.
        
        Args:
            gradients_w: List of weight gradients
            gradients_b: List of bias gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train_step(self, X_batch, y_batch):
        """
        Perform one training step (forward + backward + update).
        
        Args:
            X_batch: Input batch (batch_size, 784)
            y_batch: True labels as one-hot vectors (batch_size, 10)
            
        Returns:
            loss: Training loss for this batch
        """
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_pred, y_batch)
        
        # Backward pass
        gradients_w, gradients_b = self.backward(y_batch)
        
        # Update weights
        self.update_weights(gradients_w, gradients_b)
        
        return loss
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data (batch_size, 784)
            
        Returns:
            Predicted class labels (batch_size,)
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate accuracy on a dataset.
        
        Args:
            X: Input data (num_samples, 784)
            y: True labels (num_samples,) as integers
            
        Returns:
            accuracy: Classification accuracy
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def save(self, filepath):
        """Save model weights to file."""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.learning_rate = model_data['learning_rate']
        print(f"Model loaded from {filepath}")
