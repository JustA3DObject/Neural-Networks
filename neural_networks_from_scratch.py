import numpy as np
np.random.seed(0)

X = [[1 ,2, 3, 2.5], 
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
# print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)


# ReLU Activation funtion:

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

#     # OR
#     output.append(max(0, i))

# # print(output)

# Manually implimenting the network

# inputs = [[1 ,2, 3, 2.5], 
#           [2.0, 5.0, -1.0, 2.0], 
#           [-1.5, 2.7, 3.3, -0.8]]

# # Layer 1
# weights = [[0.2, 0.8, -0.5, 1], 
#            [0.5, -0.91, 0.26, -0.5], 
#            [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# #  Layer 2
# weights2 = [[0.1, -0.14, 0.5], 
#            [-0.5, 0.12, -0.33], 
#            [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

# Another way to do the above function

# layer_outputs  = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)


