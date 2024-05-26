import data_prep as dp
import model
import cupy as cp
import numpy as np
import csv

def write_to_csv(data, filename):
    # Abre el archivo CSV en modo de escritura
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(['ID', 'label'])
        for i, value in enumerate(data):
            csv_writer.writerow([str(i), str(value)])

def toClass(predictions):
 indexes = []
 for curClassPredictions in predictions:
    curMax = np.max(curClassPredictions)
    firstIndex = np.where(curClassPredictions == curMax)[0][0]
    
    indexes.append(firstIndex)
 return indexes
    

# const
img_size = 32
channels_amount = 3
input_size = (img_size * img_size) * channels_amount

# vars
num_classes = 10
hidden_layers = [128] * 3 # Tamaños de las capas ocultas
output_size = 10
layer_sizes = [input_size] + hidden_layers + [output_size]

# best params 0
# relu, epochs=1000, [128] * 3, learning_rate = 0.0001. Adam, He, Sin L2
# best params 1
# relu, epochs=1300, [128] * 4, lr = 0.0001. Adam, He. Con L2 y lambd = 0.01

# relu, epochs=3000, [128] * 4, lr = 0.0001, Adam, He, L2 con lambda = 0.02

# hyperparams
fn_activation = 'relu'
model = model.MultiLayerNetwork(layer_sizes, fn_activation, lambd=0.02)
epochs = 1500
learning_rate = 0.00015

########### SETS TRAIN, VAL, TEST ####################

# train ds al 90%
trainAndVal = dp.TrainAndVal('./sets/train_data.csv', train_percentage=0.8)
train_dataset = trainAndVal.get_train_data()
X_train = train_dataset['X']
y_train = train_dataset['Y']

X_train_augmented, y_train_augmented = dp.augment_data(X_train, y_train)

# normalizar xtrain
X_train_augmented = dp.normalize('minmax', X_train_augmented)
X_train = dp.normalize('minmax', X_train)

print(X_train_augmented.shape)

# val ds al 10%
val_dataset = trainAndVal.get_val_data()
X_val = val_dataset['X']
y_val = val_dataset['Y']

# normalizar xval
X_val = dp.normalize('minmax', X_val)


# test ds
test_dataset = dp.Test('./sets/test_data.csv')
X_test = test_dataset.X_data()
X_test = dp.normalize('minmax', X_test)
y_test = test_dataset.Y_data() 


######################################################

errors = model.train(X_train_augmented, y_train_augmented, epochs, learning_rate)

#Evaluar
predictions = model.predict(X_test)

filename = 'output2.csv'  # Nombre del archivo CSV de salida
predictions_list = toClass(predictions)
write_to_csv(predictions_list, filename) 

#accuracy = cp.mean(predictions.argmax(axis=1) == y_val)
#print(f'Test Accuracy validation set: {accuracy}')  