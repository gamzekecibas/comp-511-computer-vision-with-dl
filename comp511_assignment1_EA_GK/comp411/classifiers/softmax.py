import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def safelog(x):
        return(np.log(x + 1e-200))
    
    num_samples = X.shape[0]    
    num_classes = W.shape[1]
    
    # scores: shape (N, C)
    scores = np.dot(X, W)
    
    for sample in range(num_samples):
        # scaling the values for numeric stability, i.e. make highest number 0
        x = scores[sample] - np.max(scores[sample])
        
        # calculate loss
        softmax = np.exp(x) / np.sum(np.exp(x))
        loss = loss - np.log(softmax[y[sample]])
     
        # gradient operations
        for c in range(num_classes):
            dW[:,c] += X[sample] * softmax[c]
        dW[:,y[sample]] = dW[:,y[sample]] - X[sample]
    
    ## regularization part, 
    ## reg_l1 value changes the regularization method
    if reg_l1 == 0:
        ## just L2 regularization
        reg = reg_l2 * np.sum(W*W)
    else:
    ## When reg_l1 has a value, ElasticNet is used for regularization
        alpha, beta = 0.1, 0.1
        reg_1 = reg_l1 * np.sum(W)
        reg_2 = reg_l2 * np.sum(W*W)
        
        reg = alpha * reg_1 + 0.5 * beta * np.power(reg_2, 2)
        
    loss /= X.shape[0]
    loss += reg
    
    dW /= X.shape[0]
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def safelog(x):
        return(np.log(x + 1e-100))
    
    scores = np.dot(X, W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    
    softmax = scores[range(X.shape[0]), y] / np.sum(scores, axis = 1)
    
    ## regularization part, 
    ## reg_l1 value changes the regularization method
    if reg_l1 == 0:
        ## just L2 regularization
        reg = reg_l2 * np.sum(W*W)
    else:
    ## When reg_l1 has a value, ElasticNet is used for regularization
        alpha, beta = 0.5, 0.5
        reg_1 = reg_l1 * np.sum(W)
        reg_2 = reg_l2 * np.sum(W*W)
        
        reg = alpha * reg_1 + 0.5 * beta * np.power(reg_2, 2)
    
    loss = np.sum(-safelog(softmax))
    loss /= X.shape[0]
    loss += reg
    
    ## gradient step
    derivative = scores / np.sum(scores, axis = 1, keepdims = True)
    derivative[np.arange(X.shape[0]), y] -= 1
    
    dW = np.dot(X.T, derivative) / X.shape[0]
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
