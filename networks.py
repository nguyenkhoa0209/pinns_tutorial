import os
import numpy as np
import time
from math import *
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class SimpleDense(tf.keras.layers.Layer):
    """
    Simple dense main class

    Parameters
    ----------
    units : int
             number of neurons in the Dense layer
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    """
    def __init__(self, units=100, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'):
        super(SimpleDense, self).__init__()
        self.units = units

        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        """
        :meta private:
        """
        if self.kernel_initializer == 'glorot_normal':
            w_init = tf.keras.initializers.GlorotNormal()
        else:
            w_init = self.kernel_initializer
        if self.bias_initializer == 'zeros':
            b_init = tf.zeros_initializer()
        else:
            b_init = self.bias_initializer
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float64'),
                             trainable=True)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float64'), trainable=True)

        super().build(input_shape)

    def call(self, inputs):
        """
        :meta private:
        """
        return self.activation(tf.matmul(inputs, self.w) + self.b)

class GAAF(tf.keras.layers.Layer):
    """
    GAAFs main class

    Parameters
    ----------
    units : int
             number of neurons in the Dense layer
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    n : int
        hyperparameter in GAAF
    """
    def __init__(self, units=100, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros', n=10, a=0.1):
        super(GAAF, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        #self.w_prev = w_prev
        #self.b_prev = b_prev
        #self.a_prev = a_prev
        self.n = n
        self.a = a

    def build(self, input_shape):
        """
        :meta private:
        """
        if self.kernel_initializer == 'glorot_normal':
            w_init = tf.keras.initializers.GlorotNormal()
        else:
            w_init = self.kernel_initializer
        if self.bias_initializer == 'zeros':
            b_init = tf.zeros_initializer()
        else:
            b_init = self.bias_initializer
        #a_init = tf.ones_initializer()

        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float64'),
                             trainable=True)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float64'), trainable=True)
        #self.a = tf.Variable(name="a",
        #                     initial_value=a_init(shape=(1,), dtype='float64'), trainable=True)

        super().build(input_shape)

    def call(self, inputs):
        """
        :meta private:
        """
        return self.activation(self.n * self.a * tf.matmul(inputs, self.w) + self.b)

class L_LAAF(tf.keras.layers.Layer):
    """
    L-LAAFs main class

    Parameters
    ----------
    units : int
             number of neurons in the Dense layer
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    n : int
        hyperparameter in L-LAAF
    """
    def __init__(self, units=100, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros', n=10):
        super(L_LAAF, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        #self.w_prev = w_prev
        #self.b_prev = b_prev
        #self.a_prev = a_prev
        self.n = n

    def build(self, input_shape):
        """
        :meta private:
        """
        if self.kernel_initializer == 'glorot_normal':
            w_init = tf.keras.initializers.GlorotNormal()
        else:
            w_init = self.kernel_initializer
        if self.bias_initializer == 'zeros':
            b_init = tf.zeros_initializer()
        else:
            b_init = self.bias_initializer
        #a_init = tf.ones_initializer()

        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float64'),
                             trainable=True)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float64'), trainable=True)
        #self.a = tf.Variable(name="a",
        #                     initial_value=a_init(shape=(1,), dtype='float64'), trainable=True)
        self.a = tf.Variable(name="a",
                             initial_value=0.1, dtype='float64' , trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """
        :meta private:
        """
        return self.activation(self.n * self.a * tf.matmul(inputs, self.w) + self.b)

class N_LAAF(tf.keras.layers.Layer):
    """
    N-LAAFs main class

    Parameters
    ----------
    units : int
             number of neurons in the Dense layer
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    n : int
        hyperparameter in N-LAAF
    """
    def __init__(self, units=100, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros', n=10):
        super(N_LAAF, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        #self.w_prev = w_prev
        #self.b_prev = b_prev
        #self.a_prev = a_prev
        self.n = n

    def build(self, input_shape):
        """
        :meta private:
        """
        if self.kernel_initializer == 'glorot_normal':
            w_init = tf.keras.initializers.GlorotNormal()
        else:
            w_init = self.kernel_initializer
        if self.bias_initializer == 'zeros':
            b_init = tf.zeros_initializer()
        else:
            b_init = self.bias_initializer
        a_init = tf.ones_initializer()

        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float64'),
                             trainable=True)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float64'), trainable=True)
        self.a = tf.Variable(name="a",
                             initial_value=tf.keras.initializers.Constant(0.1)(shape=(self.units,)), dtype='float64', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """
        :meta private:
        """
        return self.activation(self.n * self.a * tf.matmul(inputs, self.w) + self.b)


class Rowdy(tf.keras.layers.Layer):
    """
    Rowdy main class

    Parameters
    ----------
    units : int
             number of neurons in the Dense layer
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    n : int
        hyperparameter in Rowdy
    K : int
        hyperparameter in Rowdy
    """
    def __init__(self, units=100,activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros', n=10, K=1):
        super(Rowdy, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.n = n
        self.K = K
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        """
        :meta private:
        """
        if self.kernel_initializer == 'glorot_normal':
            w_init = tf.keras.initializers.GlorotNormal()
        else:
            w_init = self.kernel_initializer
        if self.bias_initializer == 'zeros':
            b_init = tf.zeros_initializer()
        else:
            b_init = self.bias_initializer

        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float64'),
                             trainable=True)
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float64'), trainable=True)

        a_init = tf.ones_initializer()

        self.a = tf.Variable(name="a",
                             initial_value=tf.keras.initializers.Constant(0.1)(shape=(1,)), dtype='float64', trainable=True)
        self.fai = tf.Variable(name="fai",
                               initial_value=tf.keras.initializers.Constant(0.1)(shape=(self.K * 2,)), dtype='float64', trainable=True)

        super().build(input_shape)

    def call(self, inputs):
        """
        :meta private:
        """
        out = self.activation(self.n * self.a * (tf.matmul(inputs, self.w) + self.b))
        for i in range(self.K):
            out += self.n * self.fai[2 * i + 0] * self.activation(
                self.n * (i + 1) * self.fai[2 * i + 1] * (tf.matmul(inputs, self.w) + self.b))
        return out

class FNN(tf.keras.Model):
    """
    Feedforward networks (FNN) main class

    Parameters
    ----------
    layers : list
             layers of the neural networks.
    activation :
                 activation function
    kernel_initializer :
                         initialization of the kernel
    bias_initializer :
                       initialization of the bias
    method : str
             method used in FNN (supported method: 'classic' (Dense layer), 'L-LAAF', 'N-LAAF', 'Rowdy')
    n : int
        hyperparameter if using L-LAAFs, N-LAAFs or Rowdy
    K : int
        hyperparameter if using Rowdy
    """
    def __init__(self, layers, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros', method='classic', n=10,K=1):
        super(FNN, self).__init__()
        self.layers_size = layers
        self.layers_array = []
        self.method = method
        for i in range(1, len(layers)-1):
            if self.method == 'classic':
                self.layers_array.append(
                    SimpleDense(layers[i], activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
            elif self.method == 'GAAF':
                if i==1:
                    self.layers_array.append(
                        GAAF(layers[i], activation=activation, kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer, n=n, a=tf.Variable(name="a", initial_value=0.1, dtype='float64' , trainable=True)))
                else:
                    self.layers_array.append(
                        GAAF(layers[i], activation=activation, kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer, n=n, a=self.layers_array[0].a))
            elif self.method == 'L_LAAF':
                self.layers_array.append(
                    L_LAAF(layers[i], activation=activation, kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer, n=n))
            elif self.method == 'N_LAAF':
                self.layers_array.append(
                    N_LAAF(layers[i], activation=activation, kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer, n=n))
            elif self.method == 'Rowdy':
                self.layers_array.append(
                    Rowdy(layers[i], activation=activation, kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer, n=n, K=K))
            else:
                raise Exception('The FNN architecture is not supported.')
        self.layers_array.append(tf.keras.layers.Dense(layers[-1], activation=None, kernel_initializer=kernel_initializer))

    def call(self, inputs):
        """
        :meta private:
        """
        y = inputs
        for layer in self.layers_array:
            y = layer(y)
        return y
