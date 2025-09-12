# This is the file where all the steps of generating a Neural Network are 
# generated in form of functions, for the consecutive deployment and reuse
#
# All the functions will be done from scratch with NumPy framework, trying
# to ensure Vectorization for optimized performance.

import numpy as np

# 1. Initialization of parameters (Weights - W and bias - b)
#    Weights will be initialized to random values in order to break simmety
#    and allow the neurons to learn different features of the data.

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):

        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        # check that weight and bias shape are correct
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# 2. Implement the linear part of a layer's forward propagation
#    Z[l] = W[l]xA[l-1] + b[l]

def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b   
    cache = (A, W, b)
    
    return Z, cache


# 3. Implement the activations functions

# 3.1 Sigmoid activation function - g(Z) = 1 / (1+e^-Z) 

def sigmoid(Z):
    """
    Arguments:
    Z -- the input of the activation function, also called pre-activation parameter

    Returns:
    A -- activation value
    cache -- a variable that contains "Z"; stored for computing the backward pass efficiently
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

# 3.2 ReLU activation function - g(Z) = max(0,Z)

def relu(Z):
    """
    Arguments:
    Z -- the input of the activation function, also called pre-activation parameter

    Returns:
    A -- activation value
    cache -- a variable that contains "Z"; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


# 4. Implement the forward propagation for the "Linear Activation" layer
#    Z[l] = W[l]xA[l-1] + b[l]
#    A[l] = g[l](Z[l]

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
        
    cache = (linear_cache, activation_cache)

    return A, cache


# 6. Implement the full forward propagation - in this Neural Net the hidden layers will activate with 
#    the "ReLU" function, and the final L layer with "Sigmoid", as it will be used for binary classification

def L_model_forward(X, parameters):
    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2     # number of layers in the neural network (params is composed of layer_dims * each pair of W/b)
    
    # Implement (L-1) layers: Linear -> ReLU
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):

        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)
        
    
    # Implement the L layer: Linear -> Sigmoid
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
    # YOUR CODE ENDS HERE
          
    return AL, caches