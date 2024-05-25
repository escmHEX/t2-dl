import pandas as pd
import cupy as cp

class Dataset:
    def __init__(self, path, isTrainAndVal=False):
        self.data = pd.read_csv(path).to_numpy()
        self.samples = self.data.shape[0]
        self.features = self.data[:self.samples, 1:-1]
        self.labels = self.data[:self.samples, -1]
        self.labels = self.labels.reshape(1, -1).T 
        
    def X_data(self):
        return cp.array(self.features)
    
    def Y_data(self):
        return cp.array(self.labels).flatten() 

class TrainAndVal:
    def __init__(self, path, train_percentage):
        self.data = pd.read_csv(path).to_numpy()
        self.samples = self.data.shape[0]
        self.train_samples = int(self.samples * train_percentage)
        
        self.train_set = {
            'features': self.data[:self.train_samples, 1:-1],
            'labels': (self.data[:self.train_samples, -1]).reshape(1, -1).T
        }
        
        self.val_set= {
            'features': self.data[:(self.samples - self.train_samples), 1:-1],
            'labels': (self.data[:(self.samples - self.train_samples ), -1]).reshape(1, -1).T
        }
        
    def get_train_data(self):
        return {
            'X': cp.array(self.train_set['features']),
            'Y': cp.array(self.train_set['labels']).flatten()
        }
        
    def get_val_data(self):
        return {
            'X': cp.array(self.val_set['features']),
            'Y': cp.array(self.val_set['labels']).flatten()
        }
        

