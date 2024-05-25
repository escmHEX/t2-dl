import cupy as cp
import numpy as np

def xavier_initialization(N_in, N_out):
    return cp.random.randn(N_in, N_out) * cp.sqrt(2.0 / (N_in + N_out))

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(z):#z: vector resultante antes de funcion de activación en la última capa
    exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))  
    return exp_z / cp.sum(exp_z, axis=1, keepdims=True) #retorna las probabilidades de las posibles clases

#mide la diferencia entre las probs p_pred creada por softmax y las etiquetas Y reales
def cross_entropy_loss(p_pred, Y):
    epsilon = 1e-15
    prob_predicted = cp.clip(p_pred, epsilon, 1 - epsilon)

    return -cp.sum(Y * cp.log(prob_predicted))/Y.shape[0]

def one_hot(y, num_classes):
    if y.ndim > 1:  # Flatten the array if necessary
        y = y.flatten()
        
    one_hot_labels = cp.zeros((y.shape[0], num_classes))
    one_hot_labels[cp.arange(y.shape[0]), y] = 1
    return one_hot_labels

class MultiLayerNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            
            # Usando inicialización Xavier para fn sigmoide
            weight = xavier_initialization(layer_sizes[i], layer_sizes[i + 1])
            bias = cp.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, inputs):
        self.activations = [inputs]
        a = inputs

        for i in range(len(self.weights) - 1):
            z = cp.dot(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.activations.append(a)

        # Output uses softmax for multiclass classification
        z = cp.dot(a, self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.activations.append(a)
        return a

    def predict(self, inputs):
        inputs = cp.array(inputs, ndmin=2)
        a = inputs

        for i in range(len(self.weights) - 1):
            z = cp.dot(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)
        
        # Output with softmax
        z = cp.dot(a, self.weights[-1]) + self.biases[-1]
        a = softmax(z)

        return a

    def backward(self, targets, learning_rate):
        m = targets.shape[0]  # number of training examples
        delta_weights = [0] * len(self.weights)
        delta_biases = [0] * len(self.biases)

        # Calculate the initial error (difference between prediction and target for the output layer)
        error = self.activations[-1] - targets

        for i in reversed(range(len(self.weights))):
            # Calculate the delta for the current layer
            delta = error
            delta_weights[i] = cp.dot(self.activations[i].T, delta) / m
            delta_biases[i] = cp.sum(delta, axis=0, keepdims=True) / m

            if i != 0:
                # Propagate the error to the previous layer
                error = cp.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])

            # Update weights and biases
            self.weights[i] -= learning_rate * delta_weights[i]
            self.biases[i] -= learning_rate * delta_biases[i]
        
    def train(self, inputs, targets, epochs, learning_rate):
        targets = one_hot(targets, 10)

        errors = []
        
        num_equal = 0
        last_error = 0
        i = 0 

        for epoch in range(epochs):
            predictions = self.forward(inputs)
            error = cross_entropy_loss(predictions, targets)
            self.backward(targets, learning_rate)
            errors.append(error)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Error: {error}')
            
                if (last_error - error) < 0.0001: # si el costo mejora menos que esto durante 40 epocas seguidas, nos salimos
                    num_equal += 1
                    
                if num_equal >= 4:
                    return errors
                
                last_error = error
                
            if i == 0:
                last_error = error
                i +=1
                continue
            
            
        return errors
    
