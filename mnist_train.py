import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf

from neural_network import NeuralNetwork


def create_confusion_matrix(pred, true, title):
    cm = confusion_matrix(pred, true, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='PuBu')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    plt.show()


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('mnist data loaded.\n')

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

x_train_norm_reshaped = x_train_norm.reshape(x_train_norm.shape[0], -1)
x_test_norm_reshaped = x_test_norm.reshape(x_test_norm.shape[0], -1)

y_train_one_hot = np.zeros((y_train.size, y_train.max() + 1))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, y_test.max() + 1))
y_test_one_hot[np.arange(y_test.size), y_test] = 1


INPUT_SIZE = 784
layers = [(32, 'relu'), (16, 'relu'), (8, 'relu'), (10, 'softmax')]

nn = NeuralNetwork(INPUT_SIZE, layers)

print("training neural network...\n")
nn.fit(x_train_norm_reshaped, y_train_one_hot,
       epoch=100, alpha=0.001, batch_size=32)

pred_train = nn.forward_prop(x_train_norm_reshaped)
pred_test = nn.forward_prop(x_test_norm_reshaped)

pred_train_categorical = np.argmax(pred_train, axis=1)
pred_test_categorical = np.argmax(pred_test, axis=1)

train_loss = nn.calculate_loss(pred_train, y_train_one_hot)
train_accuracy = np.mean(pred_train_categorical == y_train)

test_loss = nn.calculate_loss(pred_test, y_test_one_hot)
test_accuracy = np.mean(pred_test_categorical == y_test)

create_confusion_matrix(y_train, pred_train_categorical, 'Training Data')
create_confusion_matrix(y_test, pred_test_categorical, 'Testing Data')

print(f'\ntrain | loss: {train_loss} - accuracy: {train_accuracy}')
print(f'test | loss: {test_loss} - accuracy: {test_accuracy}')
