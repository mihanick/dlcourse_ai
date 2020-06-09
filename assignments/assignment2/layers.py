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
