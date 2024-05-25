import pandas as pd
import cupy as cp
from skimage.transform import resize

class Dataset:
    def __init__(self, path):
        self.data = pd.read_csv(path).to_numpy()
        self.samples = self.data.shape[0]
        self.features = self.data[:self.samples, 1:-1]
    
        self.labels = self.data[:self.samples, -1]
        self.labels = self.labels.reshape(-1, 1)
        
    def X_data(self):
        return cp.array(self.features)
    
    def Y_data(self):
        return cp.array(self.labels).flatten() 


