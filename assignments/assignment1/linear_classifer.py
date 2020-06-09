import numpy as np


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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    predictions = np.dot(X, W)

    loss, loss_grad = softmax_with_cross_entropy(predictions, target_index)
    w_grad = np.dot(X.T, loss_grad)
    
    return loss, w_grad

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss = 0
            for idx, batch_indices in enumerate(batches_indices):
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                loss_lin, grad_lin = linear_softmax(batch_X, self.W, batch_y)
                loss_l2, grad_l2 = l2_regularization(self.W, reg)
                loss += loss_lin + loss_l2
                dW = grad_lin + grad_l2
                self.W -= learning_rate * dW
            loss /= len(batches_indices)
            loss_history.append(loss)
            # end
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        prob = softmax(predictions)
        y_pred = np.argmax(prob, -1)

        return y_pred
    
class LinearSoftmaxClassifier1():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            
            # Compute loss and gradients
            
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            
            loss = 0 
            for idx, batch_indexes in enumerate(batches_indices):
                batch_X = X[batch_indexes]
                batch_y = y[batch_indexes]
                loss_lin, grad_lin = linear_softmax(batch_X, self.W, batch_y)
                loss_l2, grad_l2 = l2_regularization(self.W, reg)
                loss += loss_lin + loss_l2
                dW = grad_lin + grad_l2
                self.W -= learning_rate * dW
                
            loss /= len(batches_indices)
            loss_history.append(loss)
            
            
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        
        predictions = np.dot(X, self.W)
        prob = softmax(predictions)
        y_pred = np.argmax(prob, -1)

        return y_pred



                
                                                          

            

                
