import cupy as cp
# Se busca optimizar: Cantidad Capas, Cantidad Neuronas por capas, epochs y learning rate

class activation_function:
    def __init__(self, n): #n = nombre fn a usar
       self.name = n
    
    def relu(self, x):
     return cp.maximum(0, x)

    def relu_derivative(self, x):
     return (x > 0).astype(x.dtype)
 
    def sigmoid(self, x):
     return 1 / (1 + cp.exp(-x))

    def sigmoid_derivative(self, x):
     return x * (1 - x)
 
    def elu(self, x, alpha=1.0):
        return cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

    def elu_derivative(self, x, alpha=1.0):
        return cp.where(x > 0, 1, alpha * cp.exp(x))

    def leaky_elu(self, x, alpha=0.01):
        return cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

    def leaky_elu_derivative(self, x, alpha=0.01):
        return cp.where(x > 0, 1, alpha * cp.exp(x))
    
    def softmax(self, x):#z: vector resultante antes de funcion de activación en la última capa
        exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))  
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True) #retorna las probabilidades de las posibles clases
    
    def value(self, z):
        if self.name == 'relu':
            return self.relu(z)
        
        elif self.name == 'sigmoid':
            return self.sigmoid(z)
        
        elif self.name == 'elu':
            return self.elu(z)
        
        elif self.name == 'leakyelu':
            return self.leaky_elu(z)   
        
        elif self.name == 'softmax':
            return self.softmax(z) 
        
        else:
            print('fn inexistente')
            return z            
    
    def derivative(self, z):
        if self.name == 'relu':
            return self.relu_derivative(z)
        
        elif self.name == 'sigmoid':
            return self.sigmoid_derivative(z)
        
        elif self.name == 'elu':
            return self.elu_derivative(z)
        
        elif self.name == 'leakyelu':
            return self.leaky_elu_derivative(z)    
        
        else:
            print('derivada inexistente')
            return z  
