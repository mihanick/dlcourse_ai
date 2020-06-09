import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        
        width, height, channels = input_shape
        pool_size = 4
        padding = 1
        filter_size = 3
        
        self.layers = []
        #self.layers.append(Flattener(name="Input"))
        self.layers.append(ConvolutionalLayer(
            in_channels=channels,
            out_channels=conv1_channels,
            filter_size = 3,
            padding = 1,
            name = "Conv1"
        ))
        self.layers.append(ReLULayer(name="Relu1"))
        self.layers.append(MaxPoolingLayer(pool_size = pool_size, stride = pool_size, name="MaxPool"))
        self.layers.append(ConvolutionalLayer(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            filter_size = filter_size,
            padding = padding,
            name = "Conv2"
        ))        
        self.layers.append(ReLULayer(name="Relu2"))
        self.layers.append(MaxPoolingLayer(pool_size = pool_size, stride = pool_size, name="MaxPool2"))
        self.layers.append(Flattener(name = "Flatten"))
        
        fc_input = conv2_channels*(width // (pool_size ** 2)) * (height // (pool_size **2) )
        
        self.layers.append(FullyConnectedLayer(n_input = fc_input, n_output = n_output_classes,  name = "FC"))
        

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

        loss, grad = softmax_with_cross_entropy(predictions, y)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        layer_grad = grad
        for layer in reversed(self.layers):
            layer_grad = layer.backward(layer_grad)
            
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

        prob = softmax(layer_output)    
        pred = np.argmax(prob, axis=1)
        
        return pred

    def params(self):

        result = {}
        for layer in self.layers:
            for layer_param_key in layer.params():
                param = layer.params()[layer_param_key]
                result[layer.name+layer_param_key] =  param

        return result
