import numpy as np
import pandas as pd


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def softmax(z):#z: vector resultante antes de funcion de activación en la última capa
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)#retorna las probabilidades de las posibles clases

#mide la diferencia entre distribucion de prob. creada por softmax y los targets reales
def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]

class MultiLayerNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1. / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, inputs):
        self.activations = [inputs]
        a = inputs

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = leaky_relu(z)
            self.activations.append(a)

        # Output uses softmax for multiclass classification
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.activations.append(a)
        return a

    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        a = inputs

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = leaky_relu(z)
        
        # Output with softmax
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = softmax(z)

        return a

    def backward(self, targets, learning_rate, clip_value=None):
        delta_weights = [0] * len(self.weights)
        delta_biases = [0] * len(self.biases)
        
        # Calculate the initial error (difference between prediction and target for the output layer)
        error = self.activations[-1] - targets
        
        for i in reversed(range(len(self.weights))):
            # Calculate the delta for the current layer
            delta = error
            delta_weights[i] = np.dot(self.activations[i].T, delta)
            delta_biases[i] = np.sum(delta, axis=0, keepdims=True)
            
            if i != 0:
                # Propagate the error to the previous layer
                error = np.dot(delta, self.weights[i].T) * leaky_relu_derivative(self.activations[i])
            
            # Clip gradients if clip_value is provided
            if clip_value:
                delta_weights[i] = np.clip(delta_weights[i], -clip_value, clip_value)
                delta_biases[i] = np.clip(delta_biases[i], -clip_value, clip_value)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * delta_weights[i]
            self.biases[i] -= learning_rate * delta_biases[i]

        #print(self.weights)
        
    def train(self, inputs, targets, epochs, learning_rate, clip_value=None):
        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2)
        errors = []

        for epoch in range(epochs):
            predictions = self.forward(inputs)
            
            self.backward(targets, learning_rate, clip_value)
            error = cross_entropy_loss(predictions, targets)
            errors.append(error)

            print(predictions)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Error: {error}')

        return errors
    
#Obtener datos (estoy usando pd porque anda considerablemente más rápido que np)
train_data = pd.read_csv(r'C:/Users/kueru/Documents/VSCode/semestre_9/Deep_Learning/T2/train_data_2.csv')
train_data = train_data.to_numpy()
    
#Cortar en features y labales
train_samples = train_data.shape[0]
features = train_data[:train_samples, 1:-1]  # Features for training    
labels = train_data[:train_samples, -1]  #Labels for training

labels = labels.reshape(-1, 1)  # Reshape to (299, 1)

X_train = np.array(features)
y_train = np.array(labels, ndmin=2)

print(train_data.shape)
print(features.shape)
print(labels.shape)

num_classes = 10
input_size = 3072
hidden_layers = [256,256,256]  # Tamaños de las capas ocultas
output_size = 10
layer_sizes = [input_size] + hidden_layers + [output_size]

model = MultiLayerNetwork(layer_sizes)
epochs = 100
learning_rate = 0.0001

errors = model.train(X_train, y_train, epochs, learning_rate, 2.5)
#Evaluar
predictions = model.predict(X_train)
accuracy = np.mean(predictions.argmax(axis=1) == y_train)
print(f'Test Accuracy: {accuracy}')

