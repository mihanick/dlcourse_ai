import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input, hidden_layer_size, name="linear1"))
        self.layers.append(ReLULayer( name = "relu"))
        self.layers.append(FullyConnectedLayer(hidden_layer_size, n_output, name="linear2"))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        for layer in self.layers:
            for param_key in layer.params():
                param = layer.params()[param_key]
                param.grad = np.zeros_like(param.grad)
                
        prev_layer_input = X
        for layer in self.layers:
            layer_output = layer.forward(prev_layer_input)
            prev_layer_input = layer_output
        predictions = layer_output
        
        target_index = np.arange(y.size)
        loss, grad = softmax_with_cross_entropy(predictions, y)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        prev_layer_grad = grad
        for layer in reversed(self.layers):
            prev_layer_grad = layer.backward(prev_layer_grad)
            
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        l2_loss = 0
        for layer in self.layers:
            for param_key in layer.params():
                param = layer.params()[param_key]
                l2_loss_, grad = l2_regularization(param.value, self.reg)
                param.grad+=grad
                l2_loss+= l2_loss_

        loss += l2_loss
        return loss 

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        
        prev_layer_input = X
        for layer in self.layers:
            layer_output = layer.forward(prev_layer_input)
            prev_layer_input = layer_output

        pred = np.argmax(layer_output, axis=1)
        
        return pred

    def params(self):

        result = {}
        for layer in self.layers:
            for layer_param_key in layer.params():
                param = layer.params()[layer_param_key]
                result[layer.name+layer_param_key] =  param

        return result