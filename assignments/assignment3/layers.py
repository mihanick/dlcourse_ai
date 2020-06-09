import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.power(W,2))
    grad = W * 2 * reg_strength

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # print("softmax input:")
    # print(predictions)
    
    
    #result = np.zeros(predictions.shape)
    
    #batch_size = predictions.shape[0]
    #for i in range(batch_size):
    #    val = _predictions[i] -  np.max(_predictions[i])
    #    sigma = np.exp(val) / np.sum(np.exp(val))        
    #    result[i] = sigma
        
     # Нормализуем вход чтобы для больших чисел не было overflow
    _predictions = predictions.copy()
    
    if (predictions.ndim == 1):        
        _predictions -= _predictions.max()
        prob = np.exp(_predictions) / np.sum(np.exp(_predictions))
    else:        
        _predictions =  _predictions - np.max(_predictions, -1)[:,None]
        prob = np.exp(_predictions) / np.sum(np.exp(_predictions), -1)[:,None]

    return prob

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    #print ("croos_entropy_loss_input: ")
    #print('probs: ', probs)
    #print('target_index: ', target_index)
    
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        loss = np.mean(-np.log(probs[np.arange(probs.shape[0]), target_index]))
        
    if np.isinf(loss):
        return 0
    
    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    #print("input:")
    #print(predictions)
    #print( target_index)
    
    probs = softmax(predictions)
   
    loss = cross_entropy_loss(probs, target_index)
    
    grad =  probs.copy()
    if (grad.ndim == 1):
        batch_size = 1
    else:
        batch_size = predictions.shape[0]
    
    if (grad.ndim == 1):
        grad[target_index] -=1
    else:
        grad[np.arange(batch_size), target_index] -=1
    
    grad = grad / batch_size
    
    return loss, grad

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self, name = "relu"):
        self.X = None
        self.name = name
    
    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        result = X.copy()
        return np.where(result<0, 0, result)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops

        # print(self.X.shape)
        # print(d_out.shape)

        return d_out * (self.X > 0) 

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output, name = "linear"):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.name = name

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #the way I get matrix calc:
        #d_in/d_out = d_f (f=x*w+b)
        #дf/дb = 1T
        #дf/дw = xT
        #дf/дx = wT
        
        Bsize = (self.X.shape[0],1)
        dB = np.ones(Bsize).T
               
        self.B.grad = np.dot(dB, d_out)
        self.W.grad = np.dot(self.X.T, d_out)
        
        d_input = np.dot(d_out, self.W.value.T)
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, name = "Convolutional"):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        
        self.X = None
        self.name = name


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1  # stride is 1
        out_width = width - self.filter_size + 2 * self.padding + 1  # stride is 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        '''
        регион входа I размера `(batch_size, filter_size, filter_size, input_channels)`,  
        применяет к нему веса W `(filter_size, filter_size, input_channels, output_channels`
        и выдает `(batch_size, output_channels)`. 

        Если:  
        - вход преобразовать в I' `(batch_size, filter_size*filter_size*input_channels)`,  
        - веса в W' `(filter_size*filter_size*input_channels, output_channels)`,  
        то выход "пикселе" будет эквивалентен полносвязному слою со входом I' и весами W'.
        '''
        self.X = X.copy()
        
        #I dont get wy we should expand array with padding, if we have filter size?
        
        # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
        # https://www.reddit.com/r/learnpython/comments/1cnjok/numpy_trying_to_pad_an_array_with_zeros/
        x_padded = np.zeros((batch_size, width+2*self.padding, height+2*self.padding, channels))
        x_padded[:, self.padding:width+self.padding, self.padding:height+self.padding] = self.X
        #print(self.X.shape)
        #print(x_padded.shape)
        
        result = np.zeros((batch_size,  out_height , out_width, self.out_channels))
        
        # magic np flattening
        wl = self.W.value.reshape(-1, self.W.value.shape[-1])
        
        for y in range(out_height):
            for x in range(out_width):
                x_filter = x_padded[:,x:x+self.filter_size,y:y+self.filter_size,:]
                
                xl = x_filter.reshape(x_padded.shape[0],-1)
                
                result[:,x,y] = np.dot(xl, wl) + self.B.value
        
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        x_padded = np.zeros((batch_size, height + 2*self.padding, width + 2*self.padding, channels))
        x_padded[:,self.padding:height+self.padding,self.padding:width+self.padding] = self.X
        
        wl = self.W.value.reshape(-1,self.W.value.shape[-1])
        
        # print(self.X.shape)
        
        d_input = np.zeros_like(self.X)
        d_input_padded = np.zeros_like(x_padded)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                x_filter = x_padded[:,x:x+self.filter_size,y:y+self.filter_size]
                xl = x_filter.reshape(x_filter.shape[0],-1)
                
                d_out_filter = d_out[:,x,y]
                
                db = np.ones((batch_size, 1))
                self.B.grad += np.dot(db.T, d_out_filter).reshape(self.B.grad.shape)
                self.W.grad += np.dot(xl.T, d_out_filter).reshape(self.W.grad.shape)
                
                d_flat = np.dot(d_out_filter,wl.T)
                # print(d_out[:,x,y].shape)
                d_input_padded[:,x:x+self.filter_size,y:y+self.filter_size] += d_flat.reshape(
                    d_input_padded[:,x:x+self.filter_size,y:y+self.filter_size].shape
                )
                
        
        d_input = d_input_padded[:,self.padding:height+self.padding,self.padding:width+self.padding]
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride, name = "MaxPool"):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.Name = name

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        self.X = X
        
        out_height = 1 + ((height - self.pool_size) //self.stride)
        out_width = 1 + ((width - self.pool_size) //self.stride)
        
        result = np.zeros((batch_size, out_height, out_width, channels))
        
        for x in range(out_width):
            for y in range(out_height):
                
                window = self.X[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size]
                result[:,x,y] = np.max(window, axis=(1,2))
                
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        s = self.stride
        f = self.pool_size

        d_input = np.zeros_like(self.X)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(out_height):
                    for x in range(out_width):
                        X_window = self.X[b, y * s:np.minimum(y * s + f, height), x * s:np.minimum(x * s + f, width), c]
                        d_input_argmax = np.unravel_index(np.argmax(X_window, axis=None), X_window.shape)
                        d_input_pool = np.zeros_like(X_window)
                        d_input_pool[d_input_argmax] = d_out[b, y, x, c]
                        d_input[b, y * s:np.minimum(y * s + f, height), x * s:np.minimum(x * s + f, width), c] += d_input_pool

        return d_input
    
    # fucking my implementation doesn't compute correct gradient
    # tried, but not succeded
    def backward_(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _,out_width,out_height,out_channels = d_out.shape
        # print(out_width, out_height)
        d_input = np.zeros_like(self.X)
        
        # give all gradient to a max
        for x in range(out_width):
            for y in range(out_height):
                # print(x,y)
                window = self.X[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size]
                
                # https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
                def max_index(arr):
                    return np.unravel_index(arr.argmax(), arr.shape)
                
                # print("window", window)
                m = max_index(window) #maxcoord
                # print("m", m)
                _,xmax,ymax,_ = m
                #xmax+= x*self.stride
                #ymax+= y*self.stride
                
                d_input[:,xmax, ymax] += d_out[:,x,y]
        
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self, name = "Flattener"):
        self.X_shape = None
        self.Name = name
        
    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        self.X_shape = X.shape
        
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
