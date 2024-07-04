import numpy as np
from activations import activations


class Layer:
    def __init__(self, num_inputs, num_neurons, activation):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation

        self.weight = np.random.randn(num_inputs, num_neurons)
        self.bias = np.random.randn(num_neurons)

        self.z = 0
        self.output = 0


class NeuralNetwork:
    def __init__(self, num_inputs, layers):
        """
        num_inputs: number of inputs of the first layer
        layers: a list of tuples containing num_neurons and activation of layer
        """
        self.layers = []

        layer_inputs = num_inputs

        for i in range(len(layers)):
            l = Layer(layer_inputs, layers[i][0], layers[i][1])
            # next layer inputs are equal to previous layers number of neurons
            layer_inputs = layers[i][0]

            self.layers.append(l)

    def summary(self):
        for layer in self.layers:
            print(f'inputs: {layer.num_inputs} - neurons: {layer.num_neurons}')

    def forward_prop(self, inputs):
        current_input = inputs

        for layer in self.layers:
            layer.z = np.dot(current_input, layer.weight) + layer.bias

            layer.output = activations[layer.activation](layer.z)

            current_input = layer.output

        return self.layers[-1].output
