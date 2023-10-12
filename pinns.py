import os
import numpy as np
import time
from math import *
import tensorflow as tf
import scipy.optimize

tf.keras.backend.set_floatx('float64')

class PINNs:
    """
    PINNs main class

    Parameters
    ----------
    X_colloc : numpy.ndarray
               Collocation points.
    net_transform :
                    Function to transform the solution output so that it sastisfies automatically some conditions.
    net_pde_user :
                   PDE defined by user.
    loss_f :
             Loss for PDE residuals.
    networks :
             Architecture of the neural networks
    lr : float
         Learning rate for Adam optimizer.
    param_pde : numpy.ndarray
                Parameter of the PDE.
    type_problem : str
                   Type of the considered problems (default: 'forward', supported type: 'forward', 'inverse', 'ill-posed', 'generalization').
    type_formulation: str
                      Type of the formulation (default: 'strong', supported type: 'strong', 'weak').
    thres : float
            threshold to stop the training
    X_bc : numpy.ndarray
           Points for boundary conditions.
    u_bc : numpy.ndarray
           Solution on boundary conditions.
    net_bc :
             Equation for other boundary conditions
    X_init : numpy.ndarray
           Points for initial condition.
    u_init : numpy.ndarray
           Solution on initial condition.
    net_init :
             Equation for initial conditions
    X_other : numpy.ndarray
           Points for other conditions.
    u_other : numpy.ndarray
           Solution on other conditions.
    net_other :
             Equation for other conditions
    X_data : numpy.ndarray
           Points for supervised measurements.
    u_data : numpy.ndarray
           Solution at supervised points.
    X_test : numpy.ndarray
           Points for testing data.
    u_test : numpy.ndarray
           Solution at testing points.
    X_traction : numpy.ndarray
           Points for traction in weak formulation.
    w_pde : float
            Weights for the PDEs residual in the cost function.
    normalized_data : bool
             Normalized or not the supervised data/IC/BC
    model_init :
                 Initial or pre-trained model.
    slope_recovery : bool
                     Slope recovery is using L-LAAFs, N-LAAFs

    """
    def __init__(self, X_colloc, net_transform, net_pde_user, loss_f, layers, lr, param_pde=None,
                 type_problem='forward', type_formulation='strong', thres=None,
                 X_bc=None, u_bc=None, net_bc=None, X_init=None, u_init=None, net_init=None, X_data=None, u_data=None,
                 X_other=None, u_other=None, net_other=None,X_test=None, u_test=None, X_traction=None, w_pde=1 ,model_init=None):
        """
        Initialisation function of PINNs class
        """

        if X_bc is None:
            self.X_bc = None
            print("No data on the boundary")
            self.u_bc = 0
            self.nb_bc = 0
        else:
            self.X_bc = tf.convert_to_tensor(X_bc, dtype='float64')
            self.u_bc = tf.convert_to_tensor(u_bc, dtype='float64')
            self.nb_bc = self.X_bc.shape[0]
            if net_bc is None:
                self.net_bc = net_transform
            else:
                self.net_bc = net_bc

        if X_init is None:
            self.X_init = None
            print("No data at the initial instant")
            self.u_init = 0
            self.nb_init = 0
        else:
            self.X_init = tf.convert_to_tensor(X_init, dtype='float64')
            self.u_init = tf.convert_to_tensor(u_init, dtype='float64')
            self.nb_init = self.X_init.shape[0]
            if net_init is None:
                self.net_init = net_transform
            else:
                self.net_init = net_init

        if X_data is None:
            self.X_data = None
            print('No data inside the domain')
            self.u_data = 0
            self.nb_data = 0
        else:
            self.X_data = tf.convert_to_tensor(X_data, dtype='float64')
            self.u_data = tf.convert_to_tensor(u_data, dtype='float64')
            self.nb_data = self.X_data.shape[0]

        if X_other is None:
            self.X_other = None
            print("No other condition is provided")
            self.u_other = 0
            self.nb_other = 0
        else:
            self.X_other = tf.convert_to_tensor(X_other, dtype='float64')
            self.u_other = tf.convert_to_tensor(u_other, dtype='float64')
            self.nb_other = self.X_other.shape[0]
            if net_other is None:
                self.net_other = net_transform
            else:
                self.net_other = net_other

        if X_test is None:
            self.X_test = None
            print('No data for testing')
            self.u_test = 0
            self.nb_test = 0
        else:
            self.X_test = tf.convert_to_tensor(X_test, dtype='float64')
            self.u_test = tf.convert_to_tensor(u_test, dtype='float64')
            self.nb_test = self.X_test.shape[0]

        if X_traction is None:
            self.X_traction = None
            self.nb_traction = 0
        else:
            self.X_traction = tf.convert_to_tensor(X_traction, dtype='float64')
            self.nb_traction = self.X_traction.shape[0]

        self.X_colloc = tf.convert_to_tensor(X_colloc, dtype='float64')
        self.type_problem = type_problem
        self.type_formulation = type_formulation
        if self.type_formulation=='weak':
            print('Using weak formulation')
        if self.type_problem =='inverse':
            self.param_pde_array = np.array([])
            if param_pde is None:
                raise Exception('Must provide initial value for the PDE parameters')
            self.param_pde = tf.Variable(param_pde, dtype='float64')
            self.nb_param = self.param_pde.shape[0]
        else:
            if param_pde is not None:
                self.param_pde = tf.convert_to_tensor(param_pde, dtype='float64')
                self.nb_param = self.param_pde.shape[0]
            else:
                self.param_pde = None
                self.nb_param = 1

        self.nb_colloc = self.X_colloc.shape[0]
        self.net_pde_user = net_pde_user
        self.loss_f = loss_f
        self.w_pde = w_pde
        self.pde_weights = self.w_pde
        self.model_init = model_init

        self.layers = layers
        self.net_transform = net_transform
        if self.model_init is None:
            if self.type_formulation == 'weak':
                self.net_u = tf.keras.Sequential()
                self.net_u.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
                for i in range(1, len(self.layers) - 1):
                    self.net_u.add(
                        tf.keras.layers.Dense(self.layers[i], activation=tf.nn.tanh,
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.1),
                                              bias_initializer='zeros'))  # "glorot_normal"
                self.net_u.add(tf.keras.layers.Dense(self.layers[-1], activation=None,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,
                                                                                                           stddev=0.1)))
            else:
                self.net_u = tf.keras.Sequential()
                self.net_u.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
                for i in range(1, len(self.layers) - 1):
                    self.net_u.add(
                        tf.keras.layers.Dense(self.layers[i], activation=tf.nn.tanh,
                                              kernel_initializer="glorot_normal"))
                self.net_u.add(
                    tf.keras.layers.Dense(self.layers[-1], activation=None, kernel_initializer="glorot_normal"))
        else:
            self.net_u = tf.keras.Sequential()
            self.net_u.add(tf.keras.layers.InputLayer(input_shape=(self.model_init.layers[0].get_weights()[0].shape[0],)))
            for i in range(1, len(self.model_init.layers)):
                W = self.model_init.trainable_variables[2 * (i - 1) + 0].numpy()
                b = self.model_init.trainable_variables[2 * (i - 1) + 1].numpy()
                self.net_u.add(tf.keras.layers.Dense(self.model_init.layers[i].get_weights()[0].shape[0], activation=self.activation,
                                                     kernel_initializer=tf.constant_initializer(W),
                                                     bias_initializer=tf.constant_initializer(b)))
            W = self.model_init.trainable_variables[-2].numpy()
            b = self.model_init.trainable_variables[-1].numpy()
            self.net_u.add(tf.keras.layers.Dense(self.model_init.layers[-1].get_weights()[0].shape[0], activation=None,
                                                 kernel_initializer=tf.constant_initializer(W),
                                                 bias_initializer=tf.constant_initializer(b)))

        self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss_array = np.array([])
        self.test_array = np.array([])
        self.thres = thres
        self.epoch = 0

    def pinns_training_variables(self):
        """
        Define training parameters in the neural networks

        :meta private:
        """
        var = self.net_u.trainable_variables
        if self.type_problem == 'inverse':
            var.extend([self.param_pde])
        return var

    @tf.function
    def net_pde(self, X_f, model_nn, param_f=None, X_traction=None):
        """
        Call PDE function defined by users

        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float
        :param model_nn: neural networks
        :type model_nn:

        :return: PDEs residual vectors

        :meta private:

        """
        if self.type_problem=='inverse':
            if X_traction is None:
                f = self.net_pde_user(X_f, model_nn, param_f)
            else:
                f = self.net_pde_user(X_f, model_nn, X_traction, param_f)
        elif (self.type_problem=='forward')&(self.type_formulation=='weak'):
            if X_traction is None:
                f = self.net_pde_user(X_f, model_nn)
            else:
                f = self.net_pde_user(X_f, model_nn, X_traction)
        else:
            f = self.net_pde_user(X_f, model_nn)
        return f

    @tf.function
    def loss_pinns(self, X_f, param_f, model_nn, u_pred_bc, u_star_bc, u_pred_init, u_star_init, u_pred_data,
                   u_star_data, u_pred_other, u_star_other, X_traction, pde_weights):
        """
        Define the cost function

        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float
        :param model_nn: neural networks
        :type model_nn:
        :param u_pred_bc: prediction for the solution on the boundary
        :type u_pred_bc: numpy.ndarray
        :param u_star_bc: reference solution on the boundary
        :type u_star_bc: numpy.ndarray
        :param u_pred_init: prediction for the solution at initial instant
        :type u_pred_init: numpy.ndarray
        :param u_star_init: reference solution at initial instant
        :type u_star_init: numpy.ndarray
        :param u_pred_data: prediction for the observed measurements
        :type u_pred_data: numpy.ndarray
        :param u_star_data: reference solution for the observed measurements
        :type u_star_data: numpy.ndarray
        :param u_pred_other: prediction for the solution on other boundary
        :type u_pred_other: numpy.ndarray
        :param u_star_other: reference solution on other boundary
        :type u_star_other: numpy.ndarray
        :param pde_weights: weights for PDE residuals
        :type pde_weights: numpy.ndarray

        :return: loss value during the training

        :meta private:
        """
        #f_value = 0
        #if self.nb_colloc > 0:
        #    f = self.net_pde(X_f, param_f, model_nn)
        #    num_pde = len(f)
        #    for i in range(num_pde):
        #        f_value += tf.reduce_mean(tf.square(f[i]))
        loss_obs = 0.0
        loss_bc = 0.0
        loss_init = 0.0
        loss_data = 0.0
        loss_other = 0.0
        loss_f = 0.0
        if self.nb_colloc > 0:
            f = self.net_pde(X_f, model_nn,param_f, X_traction)
        else:
            f = 0.0
        if self.type_problem!='generalization':
            loss_f += self.loss_f(f)

            if self.nb_bc > 0:
                for i in range(u_star_bc.shape[1]):
                    if not tf.math.is_nan(u_star_bc[0, i:(i+1)]):
                        loss_bc += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))
                        loss_obs += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i+1)] - u_star_bc[:, i:(i+1)]))
                    else:
                        loss_bc +=  tf.convert_to_tensor(0, dtype='float64')
                        loss_obs +=  tf.convert_to_tensor(0, dtype='float64')

            if self.nb_init > 0:
                for i in range(u_star_init.shape[1]):
                    if not tf.math.is_nan(u_star_init[0, i:(i+1)]):
                        loss_init += tf.reduce_mean(tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]))
                        loss_obs += tf.reduce_mean(tf.square(u_pred_init[:, i:(i+1)] - u_star_init[:, i:(i+1)]))
                    else:
                        loss_init +=  tf.convert_to_tensor(0, dtype='float64')
                        loss_obs +=  tf.convert_to_tensor(0, dtype='float64')
            if self.nb_data > 0:
                for i in range(u_star_data.shape[1]):
                    if not tf.math.is_nan(u_star_data[0, i:(i+1)]):
                        loss_data += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]))
                        loss_obs += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]))
                    else:
                        loss_data +=  tf.convert_to_tensor(0, dtype='float64')
                        loss_obs +=  tf.convert_to_tensor(0, dtype='float64')
            if self.nb_other >0:
                for i in range(u_star_other.shape[1]):
                    if not tf.math.is_nan(u_star_other[0, i:(i+1)]):
                        loss_other += tf.reduce_mean(tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]))
                        loss_obs += tf.reduce_mean(tf.square(u_pred_other[:, i:(i+1)] - u_star_other[:, i:(i+1)]))
                    else:
                        loss_other +=  tf.convert_to_tensor(0, dtype='float64')
                        loss_obs +=  tf.convert_to_tensor(0, dtype='float64') 
        else:
            for i_param in range(self.nb_param):
                if self.nb_bc > 0:
                    size_bc = int(u_star_bc.shape[0] / self.nb_param)
                    for i in range(u_star_bc.shape[1]):
                        if not tf.math.is_nan(u_star_bc[size_bc * i_param:size_bc * (i_param + 1), i:(i+1)][0]):
                            loss_obs += tf.reduce_mean(tf.square(
                                u_pred_bc[size_bc * i_param:size_bc * (i_param + 1), i:(i+1)] - u_star_bc[size_bc * i_param:size_bc * (
                                            i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.convert_to_tensor(0, dtype='float64')

                if self.nb_init > 0:
                    size_init = int(u_star_init.shape[0] / self.nb_param)
                    for i in range(u_star_init.shape[1]):
                        if not tf.math.is_nan(u_star_init[size_init * i_param:size_init * (i_param + 1), i:(i + 1)][0]):
                            if type(self.w_pde) == str:
                                loss_obs += tf.reduce_mean(tf.square((
                                    u_pred_init[size_init * i_param:size_init * (i_param + 1), i:(i + 1)] - u_star_init[
                                                                                                            size_init * i_param:size_init * (
                                                                                                                    i_param + 1),
                                                                                                            i:(i + 1)])*self.init_weights))
                            else:
                                loss_obs += tf.reduce_mean(tf.square(
                                    u_pred_init[size_init * i_param:size_init * (i_param + 1), i:(i+1)] - u_star_init[size_init * i_param:size_init * (
                                                i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.convert_to_tensor(0, dtype='float64')

                if self.nb_data > 0:
                    size_data = int(u_star_data.shape[0] / self.nb_param)
                    for i in range(u_star_data.shape[1]):
                        if not tf.math.is_nan(u_star_data[size_data * i_param:size_data * (i_param + 1), i:(i + 1)][0]):
                            loss_obs += tf.reduce_mean(tf.square(
                                u_pred_data[size_data * i_param:size_data * (i_param + 1), i:(i+1)] - u_star_data[size_data * i_param:size_data * (
                                            i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.convert_to_tensor(0, dtype='float64')

                if self.nb_other > 0:
                    size_other = int(u_star_other.shape[0] / self.nb_param)
                    for i in range(u_star_data.shape[1]):
                        if not tf.math.is_nan(u_star_other[size_other * i_param:size_other * (i_param + 1), i:(i + 1)][0]):
                            loss_obs += tf.reduce_mean(tf.square(
                                u_pred_other[size_other * i_param:size_other * (i_param + 1), i:(i+1)] - u_star_other[size_other * i_param:size_other * (
                                            i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.convert_to_tensor(0, dtype='float64')

                index_i_param = tf.where(X_f[:, -1] == param_f[i_param])
                index_i_param = tf.reshape(index_i_param, [-1])
                f_i = tf.gather(f, index_i_param)

                loss_f += self.loss_f(f_i)

        loss = loss_obs + loss_f*pde_weights

        return loss, loss_bc, loss_init, loss_data, loss_other, loss_f

    @tf.function
    def test_pde(self, X_sup_test, u_sup_test, model_test):
        """
        Define testing function

        :param X_sup_test: testing points
        :type X_sup_test: numpy.ndarray
        :param u_sup_test: reference solution on testing points
        :type u_sup_test: numpy.ndarray
        :param model_test: neural networks
        :type model_test:

        :return: error in testing data set
        :meta private:
        """
        u_pred_test = self.net_transform(X_sup_test, model_test)
        return tf.reduce_mean(
            tf.square(u_pred_test - u_sup_test)) / tf.reduce_mean(tf.square(u_sup_test))

    @tf.function
    def get_grad(self, X_f, param_f):
        """
        Calculate the gradients of the cost function w.r.t. training variables

        :param X_f: collocation points
        :type X_f: numpy.ndarray
        :param param_f: parameter of the PDE
        :type param_f: float

        :return: gradients
        :meta private:
        """
        with tf.GradientTape(persistent=True) as tape:
            if self.nb_bc > 0:
                if self.type_problem == 'inverse':
                    u_pred_bc = self.net_bc(self.X_bc, self.net_u, param_f)
                else:
                    u_pred_bc = self.net_bc(self.X_bc, self.net_u)
            else:
                u_pred_bc = 0

            if self.nb_init > 0:
                if self.type_problem == 'inverse':
                    u_pred_init = self.net_init(self.X_init, self.net_u, param_f)
                else:
                    u_pred_init = self.net_init(self.X_init, self.net_u)
            else:
                u_pred_init = 0

            if self.nb_data > 0:
                u_pred_data = self.net_transform(self.X_data, self.net_u)
            else:
                u_pred_data = 0

            if self.nb_other > 0:
                if self.type_problem == 'inverse':
                    u_pred_other = self.net_other(self.X_other, self.net_u, param_f)
                else:
                    u_pred_other = self.net_other(self.X_other, self.net_u)
            else:
                u_pred_other = 0

            loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f = self.loss_pinns(X_f, param_f, self.net_u, u_pred_bc, self.u_bc, u_pred_init, self.u_init,
                                         u_pred_data, self.u_data, u_pred_other, self.u_other, self.X_traction, self.pde_weights)

            grads = tape.gradient(loss_value, self.pinns_training_variables())

        return loss_value,loss_bc, loss_init, loss_data, loss_other, loss_f, grads

    def train(self, max_epochs_adam=0, max_epochs_lbfgs=0, print_per_epochs=1000):
        """
        Train the neural networks

        :param max_epochs_adam: Maximum number of epochs for Adam optimizer
        :type max_epochs_adam: int
        :param max_epochs_lbfgs: Maximum number of epochs for LBFGS optimizer
        :type max_epochs_lbfgs: int
        :param print_per_epochs: Print the loss after a certain of epochs.
        :type print_per_epochs: int

        """
        @tf.function
        def train_step(X_f, param_f):

            loss_value_, loss_bc_, loss_init_, loss_data_, loss_other_, loss_f_, grads = self.get_grad(X_f, param_f)
            self.tf_optimizer.apply_gradients(
                zip(grads, self.pinns_training_variables()))

            return loss_value_, loss_bc_, loss_init_, loss_data_, loss_other_, loss_f_

        for epoch in range(max_epochs_adam):
            loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f = train_step(self.X_colloc, self.param_pde)
            if self.epoch % print_per_epochs == 0:
                print('Loss pinns at epoch %d (Adam):' % self.epoch, loss_value.numpy())
            self.loss_array = np.append(self.loss_array, loss_value.numpy())
            if self.type_problem == 'inverse':
                self.param_pde_array = np.append(self.param_pde_array, self.param_pde.numpy())

            if self.X_test is not None:
                if epoch % 1000 == 0:
                    if self.nb_param==1:
                        res_test = self.test_pde(self.X_test, self.u_test, self.net_u)
                        self.test_array = np.append(self.test_array, res_test.numpy())
                        if res_test.numpy() < self.thres**2:
                            break
                    else:
                        res_test_array = np.array([])
                        for i_param in range(self.nb_param):
                            size_test = int(self.u_test.shape[0] / self.nb_param)
                            res_test = self.test_pde(self.X_test[size_test * i_param:size_test * (i_param + 1)],
                                                     self.u_test[size_test * i_param:size_test * (i_param + 1)], self.net_u)
                            res_test_array = np.append(res_test_array, res_test.numpy())
                        if np.mean(res_test_array) < self.thres**2:
                            break
            self.epoch += 1

        def callback(x=None):
            if self.type_problem == 'inverse':
                self.param_pde_array = np.append(self.param_pde_array, self.param_pde.numpy())
            if self.epoch % print_per_epochs == 0:
                print('Loss pinns at epoch %d (L-BFGS):' % self.epoch, self.current_loss)
            self.epoch += 1

        def optimizer_lbfgs(X_f, param_f, method='L-BFGS-B', **kwargs):
            """
            Optimizer LBFGS to minimize the loss

            :param X_f: Collocation points
            :type X_f: numpy.ndarray
            :param param_f: PDE parameters
            :type param_f: numpy.ndarray

            :meta private:
            """

            def get_weight():
                list_weight = []
                for variable in self.pinns_training_variables():
                    list_weight.extend(variable.numpy().flatten())
                list_weight = tf.convert_to_tensor(list_weight)
                return list_weight

            def set_weight(list_weight):
                index = 0
                for variable in self.pinns_training_variables():
                    if len(variable.shape) == 2:
                        len_weights = variable.shape[0] * variable.shape[1]
                        new_variable = tf.reshape(list_weight[index:index + len_weights],
                                                  (variable.shape[0], variable.shape[1]))
                        index += len_weights
                    elif len(variable.shape) == 1:
                        len_biases = variable.shape[0]
                        new_variable = list_weight[index:index + len_biases]
                        index += len_biases
                    else:
                        new_variable = list_weight[index]
                        index += 1
                    variable.assign(tf.cast(new_variable, 'float64'))

            def get_loss_and_grad(w):
                set_weight(w)
                loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grad = self.get_grad(X_f, param_f)
                self.loss_array = np.append(self.loss_array, loss_value.numpy())
                loss = loss_value.numpy().astype(np.float64)
                self.current_loss = loss

                grad_flat = []
                for g in grad:
                    grad_flat.extend(g.numpy().flatten())
                grad_flat = np.array(grad_flat, dtype=np.float64)
                return loss, grad_flat

            return scipy.optimize.minimize(fun=get_loss_and_grad,
                                           x0=get_weight(),
                                           jac=True,
                                           method=method, callback=callback, **kwargs)

        if max_epochs_lbfgs > 0:
            if max_epochs_adam == 0:
                draft = self.net_u(self.X_colloc)

            optimizer_lbfgs(self.X_colloc, self.param_pde,
                            method='L-BFGS-B',
                            options={'maxiter': max_epochs_lbfgs,
                                     'maxfun': max_epochs_lbfgs,
                                     'maxcor': 100,
                                     'maxls': 100,
                                     'ftol': 0,
                                     'gtol': 1.0 * np.finfo(float).eps})