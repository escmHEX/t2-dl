import data_prep as dp
import model
import cupy as cp

# const
img_size = 32
channels_amount = 3
input_size = (img_size * img_size) * channels_amount

# vars
num_classes = 10
hidden_layers = [128] * 3 # Tamaños de las capas ocultas
output_size = 10
layer_sizes = [input_size] + hidden_layers + [output_size]

# hyperparams
fn_activation = 'sigmoid'
model = model.MultiLayerNetwork(layer_sizes, fn_activation)
epochs = 600
learning_rate = 0.001

########### SETS TRAIN, VAL, TEST ####################

# train ds al 90%
trainAndVal = dp.TrainAndVal('./sets/train_data.csv', train_percentage=0.9)
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


# test ds
test_dataset = dp.Dataset('./sets/test_data.csv')
X_test = test_dataset.X_data()
y_test = test_dataset.Y_data() 


######################################################

errors = model.train(X_train, y_train, epochs, learning_rate)

#Evaluar
predictions = model.predict(X_val)
accuracy = cp.mean(predictions.argmax(axis=1) == y_val)
print(f'Test Accuracy validation set: {accuracy}')
