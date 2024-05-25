import data_prep as dp
import model
import cupy as cp

# const
img_size = 32
channels_amount = 3
input_size = (img_size * img_size) * channels_amount

# set ?
# 204 0.08114909455303297 3

# vars
num_classes = 10
hidden_layers = [256] * 3 # Tamaños de las capas ocultas
output_size = 10
layer_sizes = [input_size] + hidden_layers + [output_size]

# hyperparams
#model = model.MultiLayerNetwork(layer_sizes)
epochs = 204
learning_rate = 0.08114909455303297

train_dataset = dp.Dataset('./sets/train_data.csv')
X_train = train_dataset.X_data()
y_train = train_dataset.Y_data()

test_dataset = dp.Dataset('./sets/test_data.csv')
X_test = test_dataset.X_data()
y_test = test_dataset.Y_data()

#errors = model.train(X_train, y_train, epochs, learning_rate)


#Evaluar
#predictions = model.predict(X_train)
#accuracy = cp.mean(predictions.argmax(axis=1) == y_train)
#print(f'Test Accuracy: {accuracy}')