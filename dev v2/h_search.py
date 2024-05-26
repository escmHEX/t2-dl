import data_prep as dp
import cupy as np
import model as m

img_size = 32
channels_amount = 3
input_size = (img_size * img_size) * channels_amount

# train ds al 90%
trainAndVal = dp.TrainAndVal('./sets/train_data.csv', train_percentage=0.85)
train_dataset = trainAndVal.get_train_data()
X_train = train_dataset['X']
y_train = train_dataset['Y']

# normalizar xtrain
X_train = dp.normalize('minmax', X_train)

# val ds al 10%
val_dataset = trainAndVal.get_val_data()
X_val = val_dataset['X']
y_val = val_dataset['Y']

# normalizar xval
X_val = dp.normalize('minmax', X_val)

# Definir la función objetivo (a ser definida por ti)
def objective_function(params):
    
    learning_rate, epochs, hidden_layers_size = params
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    hidden_layers_size = int(hidden_layers_size)
    hidden_layers = [128] * hidden_layers_size
    
    layer_sizes = [input_size] + hidden_layers + [10]
    model = m.MultiLayerNetwork(layer_sizes, 'sigmoid')
    
    # entrenar
    errors = model.train(X_train, y_train, epochs, learning_rate)
    
    # evaluar
    predictions = model.predict(X_val)    
    accuracy = np.mean(predictions.argmax(axis=1) == y_val)
    
    return accuracy

# Definir la función de adquisición
def acquisition_function(mu, sigma, beta=2.0):
    return mu + beta * sigma

# Definir el kernel RBF
def rbf_kernel(X1, X2, length_scale=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / length_scale**2 * sqdist)

# Implementar la regresión gaussiana
class GaussianProcessRegressor:
    def __init__(self, kernel=rbf_kernel, alpha=1e-6):
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(X, X) + self.alpha * np.eye(len(X))
        self.L = np.linalg.cholesky(self.K)

    def predict(self, X):
        K_s = self.kernel(self.X_train, X)
        Lk = np.linalg.solve(self.L, K_s)
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y_train))
        K_ss = self.kernel(X, X)
        s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
        sigma = np.sqrt(s2)
        return mu, sigma


# Espacio de hiperparámetros
bounds = {
    'learning_rate': (0.0001, 0.1),
    'epochs': (200, 500),
    'hidden_layers': (2, 6)
}

# Inicializar con puntos aleatorios
def random_sample(bounds, n_samples=5):
    samples = []
    for _ in range(n_samples):
        sample = [
            np.random.uniform(bounds['learning_rate'][0], bounds['learning_rate'][1]),
            np.random.uniform(bounds['epochs'][0], bounds['epochs'][1]),
            np.random.uniform(bounds['hidden_layers'][0], bounds['hidden_layers'][1])
        ]
        samples.append(sample)
    return np.array(samples)

# Optimización bayesiana
def bayesian_optimization(objective_function, bounds, n_iter=25, init_points=5):
    # Inicializar puntos aleatorios
    X_init = random_sample(bounds, n_samples=init_points)
    y_init = np.array([objective_function(x) for x in X_init])

    gp = GaussianProcessRegressor()
    X_sample = X_init
    y_sample = y_init

    for i in range(n_iter):
        gp.fit(X_sample, y_sample)
        
        # Crear una malla de puntos para predecir
        X_grid = random_sample(bounds, n_samples=1000)
        mu, sigma = gp.predict(X_grid)
        acquisition_values = acquisition_function(mu, sigma)
        
        # Seleccionar el siguiente punto
        next_point = X_grid[np.argmax(acquisition_values)]

        # Evaluar el siguiente punto
        next_value = objective_function(next_point)
        
        # Actualizar muestras
        X_sample = np.vstack((X_sample, next_point))
        y_sample = np.append(y_sample, next_value)

        print(f"Iteration {i+1}/{n_iter} - Best Value: {np.max(y_sample)}")

    best_index = np.argmax(y_sample)
    best_params = X_sample[best_index]
    return best_params, np.max(y_sample)


# Ejecución del optimizador
best_params, best_score = bayesian_optimization(objective_function, bounds)
print(f"Mejores Hiperparámetros: {best_params}")
print(f"Mejor Puntuación: {best_score}")
