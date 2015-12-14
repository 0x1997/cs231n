import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes, num_train = X.shape
  for i in xrange(num_train):
    fi = W.dot(X[:, i])
    fi -= np.max(fi)
    loss += -fi[y[i]]
    dW[y[i]] += -X[:, i]
    fi_exp = np.exp(fi)
    fi_exp_sum = fi_exp.sum()
    loss += np.log(fi_exp_sum)
    dW += np.outer(fi_exp / fi_exp_sum, X[:, i])
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes, num_train = X.shape
  train_indices = xrange(num_train)
  f = W.dot(X)
  f -= np.max(f, axis=0)
  fy = f[y, train_indices]
  f_exp = np.exp(f)
  f_exp_sum = f_exp.sum(axis=0)
  loss += np.mean(-fy + np.log(f_exp_sum))
  loss += 0.5 * reg * np.sum(W * W)
  f_exp /= f_exp_sum
  f_exp[y, train_indices] -= 1
  dW += f_exp.dot(X.T)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
