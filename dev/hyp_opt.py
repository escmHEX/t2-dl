# Se busca optimizar: Cantidad Capas, Cantidad Neuronas por capas, epochs y learning rate

from scipy.stats import norm
import numpy as np
import train
import model as m
from scipy.optimize import minimize
import cupy as cp

img_size = 32
channels_amount = 3
input_size = (img_size * img_size) * channels_amount
output_size = 10

def train_evaluate(epochs, learning_rate, hidden_layers_num):
    # Aquí iría el código para entrenar y evaluar tu red neuronal.
    # Por simplicidad, esta función devolverá una métrica de evaluación aleatoria.
    # En un caso real, debes entrenar la red y devolver la precisión o pérdida en el conjunto de validación.
    
    epochs = int(epochs)
    hidden_layers_num = int(hidden_layers_num)
    
    print(epochs, learning_rate, hidden_layers_num)
    
    hidden_layers = [256] * hidden_layers_num # Tamaños de las capas ocultas
    layer_sizes = [input_size] + hidden_layers + [output_size]
    model = m.MultiLayerNetwork(layer_sizes)
    
    errors = model.train(train.X_train, train.y_train, epochs, learning_rate)
    
    predictions =  model.predict(train.X_train)
    accuracy = train.cp.mean(predictions.argmax(axis=1) == train.y_train)
    
    print(accuracy)
    return accuracy

def expected_improvement(mean, std, y_max, xi=0.01):
    z = (mean - y_max - xi) / std
    return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

class GaussianProcessRegressor:
    def __init__(self, alpha=1e-10):
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.N = len(X)
        self.K = self.kernel(X, X) + self.alpha * np.eye(self.N)
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_test):
        K_star = self.kernel(self.X_train, X_test)
        mu = K_star.T @ self.K_inv @ self.y_train
        K_star_star = self.kernel(X_test, X_test)
        cov = K_star_star - K_star.T @ self.K_inv @ K_star
        return mu, np.sqrt(np.diag(cov))

    def kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

import numpy as np

import numpy as np

class BayesianOptimizer:
    def __init__(self, f, bounds, num_layers_bounds, n_initial_points=5, n_iter=25, xi=0.01):
        self.f = f
        self.bounds = bounds
        self.num_layers_bounds = num_layers_bounds
        self.n_initial_points = n_initial_points
        self.n_iter = n_iter
        self.xi = xi
        self.X = []
        self.Y = []

    def initialize(self):
        for _ in range(self.n_initial_points):
            x = [np.random.uniform(low, high) for low, high in self.bounds]
            num_layers = np.random.randint(self.num_layers_bounds[0], self.num_layers_bounds[1] + 1)
            x.append(num_layers)
            y = self.f(*x)
            self.X.append(x)
            self.Y.append(y)

    def optimize(self):
        self.initialize()
        for _ in range(self.n_iter):
            x_new = self.propose_location()
            y_new = self.f(*x_new)
            self.X.append(x_new)
            self.Y.append(y_new)
        best_index = np.argmax(self.Y)
        return self.X[best_index], self.Y[best_index]

    def propose_location(self):
        # Minimizar la mejora esperada negativa
        def min_obj(X):
            X = np.array(X).reshape(-1, len(self.bounds) + 1)  # +1 para incluir num_layers
            mean, std = self.gp_predict(X)
            y_max = np.max(self.Y)
            return -expected_improvement(mean, std, y_max, self.xi)
        
        x_initial = [np.random.uniform(low, high) for low, high in self.bounds]
        num_layers = np.random.randint(self.num_layers_bounds[0], self.num_layers_bounds[1] + 1)
        x_initial.append(num_layers)
        
        result = minimize(min_obj, x_initial, bounds=self.bounds + [self.num_layers_bounds], method='L-BFGS-B')
        return result.x

    def gp_predict(self, X):
        gpr = GaussianProcessRegressor(alpha=1e-5)
        gpr.fit(np.array(self.X), np.array(self.Y))
        return gpr.predict(np.array(X))

    def __init__(self, f, bounds, num_layers_bounds, n_initial_points=10, n_iter=25, xi=0.01):
        self.f = f
        self.bounds = bounds
        self.num_layers_bounds = num_layers_bounds
        self.n_initial_points = n_initial_points
        self.n_iter = n_iter
        self.xi = xi
        self.X = []
        self.Y = []

    def initialize(self):
        for _ in range(self.n_initial_points):
            x = [np.random.uniform(low, high) for low, high in self.bounds]
            num_layers = np.random.randint(self.num_layers_bounds[0], self.num_layers_bounds[1] + 1)
            x.append(num_layers)
            y = self.f(*x)
            self.X.append(x)
            self.Y.append(y)

    def optimize(self):
        self.initialize()
        for _ in range(self.n_iter):
            x_new = self.propose_location()
            y_new = self.f(*x_new)
            self.X.append(x_new)
            self.Y.append(y_new)
        best_index = np.argmax(self.Y)
        return self.X[best_index], self.Y[best_index]

    def propose_location(self):
        # Minimizar la mejora esperada negativa
        def min_obj(X):
            X = np.array(X).reshape(-1, len(self.bounds) + 1)  # +1 para incluir num_layers
            mean, std = self.gp_predict(X)
            y_max = np.max(self.Y)
            return -expected_improvement(mean, std, y_max, self.xi)
        
        x_initial = [np.random.uniform(low, high) for low, high in self.bounds]
        num_layers = np.random.randint(self.num_layers_bounds[0], self.num_layers_bounds[1] + 1)
        x_initial.append(num_layers)
        
        result = minimize(min_obj, x_initial, bounds=self.bounds + [self.num_layers_bounds], method='L-BFGS-B')
        return result.x

    def gp_predict(self, X):
        # Convertir los datos de entrada a matrices de NumPy si son de CuPy
        X_np = np.array(self.X) if isinstance(self.X[0], cp.ndarray) else np.array(self.X)
        Y_np = np.array(self.Y) if isinstance(self.Y[0], cp.ndarray) else np.array(self.Y)
        
        gpr = GaussianProcessRegressor(alpha=1e-5)
        gpr.fit(X_np, Y_np)
        return gpr.predict(np.array(X))

# Definir los límites de los hiperparámetros
bounds = [(100, 800), (0.001, 0.1)]  # (epochs, learning rate)
num_layers_bounds = (3, 7)

# Crear el optimizador
optimizer = BayesianOptimizer(train_evaluate, bounds, num_layers_bounds)

# Optimizar los hiperparámetros
best_params, best_score = optimizer.optimize()
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")