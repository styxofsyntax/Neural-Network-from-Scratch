import numpy as np
from activations import activations, deriv_activations


class Layer:
    def __init__(self, num_inputs, num_neurons, activation):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation

        std = np.sqrt(2. / num_inputs)
        self.weight = np.random.randn(
            num_inputs, num_neurons) * std
        self.bias = np.zeros(num_neurons)

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
            layer = Layer(layer_inputs, layers[i][0], layers[i][1])
            # Next layer inputs are equal to previous layers number of neurons
            layer_inputs = layers[i][0]

            self.layers.append(layer)

    def summary(self):
        for layer in self.layers:
            print(f'inputs: {layer.num_inputs} - neurons: {layer.num_neurons}')

    def forward_prop(self, inputs):
        current_input = np.copy(inputs)

        for layer in self.layers:
            layer.z = np.dot(current_input, layer.weight) + layer.bias

            layer.output = activations[layer.activation](layer.z)

            current_input = layer.output

        return self.layers[-1].output

    def backward_prop(self, inputs, outputs, alpha):
        diff = self.layers[-1].output - outputs
        dE_a = (2.0 * diff) / len(outputs)

        for i in range(len(self.layers))[::-1]:

            da_z = deriv_activations[self.layers[i].activation](
                self.layers[i].z)

            dz_a_prev = self.layers[i].weight

            dz_w = self.layers[i-1].output if i > 0 else np.copy(inputs)

            dE_z = dE_a * da_z

            dE_w = np.dot(dz_w.T, dE_z)

            dE_b = np.sum(dE_z, axis=0)
            dE_b = dE_b.squeeze()

            # Update weights and biases
            self.layers[i].weight -= alpha * dE_w
            self.layers[i].bias -= alpha * dE_b

            # Propagate the error to the previous layer
            dE_a = np.dot(dE_z, dz_a_prev.T)

    def fit(self, inputs, outputs, epoch, alpha, batch_size):
        for i in range(epoch):
            batches = len(inputs) // batch_size

            for j in range(batches):
                start_index = j * batch_size
                batch_input = inputs[start_index: start_index + batch_size]
                batch_output = outputs[start_index: start_index + batch_size]

                self.forward_prop(batch_input)
                self.backward_prop(batch_input, batch_output, alpha)

            if i % (epoch // 10) == 0:
                pred = self.forward_prop(inputs)
                loss = self.calculate_loss(pred, outputs)

                pred_categorical = np.argmax(pred, axis=1)
                outputs_categorical = np.argmax(outputs, axis=1)

                accuracy = np.mean(pred_categorical == outputs_categorical)
                print(f'loss: {loss} - accuracy: {accuracy}')

    def calculate_loss(self, outputs, targets):
        return np.mean(np.square(outputs - targets))
