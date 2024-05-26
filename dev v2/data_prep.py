import pandas as pd
import cupy as cp
import numpy as np
from skimage.transform import rotate, AffineTransform, warp
from skimage.exposure import adjust_gamma

class Test:
    def __init__(self, path):
        self.data = pd.read_csv(path).to_numpy()
        self.samples = self.data.shape[0]
        self.features = self.data[:self.samples, 1:]
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
        
        
def normalize(method, dataset):
    
  if method == 'zscore':
    """ ds = dataset.copy()
    u = ds.mean(axis=0)
    std = ds.std(axis=0)

    arr_normalized = (ds - u)/std 
    
    return arr_normalized
    """
      
    X = dataset.copy()
      
    # Reshape X_train to separate the color channels
    X_red = X[:, :1024]
    X_green = X[:, 1024:2048]
    X_blue = X[:, 2048:]

    # Normalize each channel separately
    def normalize_channel(channel):
        # axis=1 porque calculamos la media y std para cada imagen por separado a la vez de separados por canal
        mean = cp.mean(channel, axis=1, keepdims=True) 
        std = cp.std(channel, axis=1, keepdims=True)
        return (channel - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

    X_red_norm = normalize_channel(X_red)
    X_green_norm = normalize_channel(X_green)
    X_blue_norm = normalize_channel(X_blue)

    # Concatenate the normalized channels back together
    X_norm = cp.concatenate((X_red_norm, X_green_norm, X_blue_norm), axis=1)
    
    return X_norm

  elif method == 'minmax':
      ds = dataset.copy()
      min_cols = ds.min(axis=0)
      max_cols = ds.max(axis=0)
      
      arr_normalized = (ds - min_cols)/(max_cols - min_cols)
      
      return arr_normalized

  else:
      print('Método no disponible')
      return dataset
  
  
  
def translate_image(image, tx, ty):
    transform = AffineTransform(translation=(tx, ty))
    return warp(image, transform, mode='wrap')

def rotate_image(image, angle):
    return rotate(image, angle, mode='wrap')

def flip_image(image):
    return np.fliplr(image)

def adjust_brightness(image, factor):
    return adjust_gamma(image, gamma=factor)

def augment_data(X_train, y_train):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(X_train, y_train):
        img = cp.asnumpy(img).reshape(32, 32, 3) / 255.0  # Convertir a numpy y normalizar a [0, 1]
        label = cp.asnumpy(label)

        # Original image
        augmented_images.append((img * 255).astype(np.uint8).flatten())
        augmented_labels.append(label)

        # Translations
        for tx, ty in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            translated_img = translate_image(img, tx, ty)
            augmented_images.append((translated_img * 255).astype(np.uint8).flatten())
            augmented_labels.append(label)

        # Rotations
        for angle in [-15, 15]:
            rotated_img = rotate_image(img, angle)
            augmented_images.append((rotated_img * 255).astype(np.uint8).flatten())
            augmented_labels.append(label)

        # Horizontal flip
        flipped_img = flip_image(img)
        augmented_images.append((flipped_img * 255).astype(np.uint8).flatten())
        augmented_labels.append(label)

        # Brightness adjustments
        for factor in [0.9, 1.1]:
            bright_img = adjust_brightness(img, factor)
            augmented_images.append((bright_img * 255).astype(np.uint8).flatten())
            augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Convertir de nuevo a cupy
    return cp.array(augmented_images), cp.array(augmented_labels)