import cupy as cp
import numpy as np
from hyp_opt import activation_function

def xavier_initialization(N_in, N_out):
    return cp.random.randn(N_in, N_out) * cp.sqrt(2.0 / (N_in + N_out))

def he_initialization(N_in, N_out):
        return cp.random.randn(N_in, N_out) * cp.sqrt(2.0 /N_in)

#mide la diferencia entre las probs p_pred creada por softmax y las etiquetas Y reales
def cross_entropy_loss(p_pred, Y):
    epsilon = 1e-15
    prob_predicted = cp.clip(p_pred, epsilon, 1 - epsilon)

    return -cp.sum(Y * cp.log(prob_predicted))/Y.shape[0]

def one_hot(y, num_classes):
    if y.ndim > 1: 
        y = y.flatten()
        
    one_hot_labels = cp.zeros((y.shape[0], num_classes))
    one_hot_labels[cp.arange(y.shape[0]), y] = 1
    return one_hot_labels

class MultiLayerNetwork:
    def __init__(self, layer_sizes, n): # fn - funcion activacion
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.fn = activation_function(n)
        self.softMax = activation_function('softmax')
        
        self.momentums = []
        self.velocities = []
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            
            # Usando inicialización Xavier, asumiendo sigmoide
            weight = he_initialization(layer_sizes[i], layer_sizes[i + 1])
            bias = cp.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            self.momentums.append(np.zeros_like(self.weights[-1]))  
            self.velocities.append(np.zeros_like(self.weights[-1]))

    def forward(self, inputs):
        self.activations = [inputs]
        a = inputs

        for i in range(len(self.weights) - 1):
            z = cp.dot(a, self.weights[i]) + self.biases[i]
            a = self.fn.value(z)
            self.activations.append(a)

        # Output uses softmax for multiclass classification
        z = cp.dot(a, self.weights[-1]) + self.biases[-1]
        a = self.softMax.value(z)
        self.activations.append(a)
        return a

    def predict(self, inputs):
        inputs = cp.array(inputs, ndmin=2)
        a = inputs

        for i in range(len(self.weights) - 1):
            z = cp.dot(a, self.weights[i]) + self.biases[i]
            a = self.fn.value(z)
        
        # Output with softmax
        z = cp.dot(a, self.weights[-1]) + self.biases[-1]
        a = self.softMax.value(z)

        return a

    def backward(self, targets, learning_rate, t):
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
                error = cp.dot(delta, self.weights[i].T) * self.fn.derivative(self.activations[i])
            
            # Actualizar pesos usando adam
            self.momentums[i] = self.beta1 * self.momentums[i] + (1 - self.beta1) * delta_weights[i]
            self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * delta_weights[i]**2
            m_hat = self.momentums[i] / (1 - self.beta1**(t + 1))
            v_hat = self.velocities[i] / (1 - self.beta2**(t + 1))
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
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
            self.backward(targets, learning_rate, epoch)
            errors.append(error)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Error: {error}')
            
                if (last_error - error) < 0.0001: # si el costo mejora menos que esto durante 40 epocas seguidas, nos salimos
                    num_equal += 1
                    
                if num_equal >= 6:
                    return errors
                
                last_error = error
                
            if i == 0:
                last_error = error
                i +=1
                continue
            
            
        return errors
    
