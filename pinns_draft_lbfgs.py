import os
import numpy as np
import time
from math import *
import tensorflow as tf
import tensorflow_probability as tfp
from adapt_sampling import FBOAL,RAD,RARD,Evo,Dynamic
from networks import FNN
from utils import jacobian, compute_J, compute_K, compute_Ki
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
    def __init__(self, X_colloc, net_transform, net_pde_user, loss_f, networks, lr, param_pde=None,
                 type_problem='forward', type_formulation='strong', thres=None,
                 X_bc=None, u_bc=None, net_bc=None, X_init=None, u_init=None, net_init=None, X_data=None, u_data=None,
                 X_other=None, u_other=None, net_other=None,X_test=None, u_test=None, X_traction=None, w_pde=1,w_init=1, w_bc=1,w_other=1,w_data=1, normalize_data = False, period_w_pde=1, slope_recovery=False,model_init=None):
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
            self.w_bc = tf.convert_to_tensor(w_bc, dtype='float64')
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
            self.w_init = tf.convert_to_tensor(w_init, dtype='float64')
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
            self.w_data = tf.convert_to_tensor(w_data, dtype='float64')
            self.nb_data = self.X_data.shape[0]

        if X_other is None:
            self.X_other = None
            print("No other condition is provided")
            self.u_other = 0
            self.nb_other = 0
        else:
            self.X_other = tf.convert_to_tensor(X_other, dtype='float64')
            self.u_other = tf.convert_to_tensor(u_other, dtype='float64')
            self.w_other = tf.convert_to_tensor(w_other, dtype='float64')
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
        self.normalize_data = normalize_data
        self.period_w_pde = period_w_pde
        if type(self.w_pde)==str:
            if self.w_pde=='SAPINNs':
                pde_weights_init = tf.ones_initializer()
                self.pde_weights = tf.Variable(
                                     initial_value=pde_weights_init(shape=(self.X_colloc.shape[0],1), dtype='float64'), trainable=True)
                if self.nb_init > 0:
                    init_weights_init = tf.ones_initializer()
                    self.init_weights = tf.Variable(
                        initial_value=init_weights_init(shape=(self.X_init.shape[0], 1), dtype='float64'),
                        trainable=True)
                if self.nb_data > 0:
                    data_weights_init = tf.ones_initializer()
                    self.data_weights = tf.Variable(
                        initial_value=data_weights_init(shape=(self.X_data.shape[0], 1), dtype='float64'),
                        trainable=True)
                if self.nb_other > 0:
                    other_weights_init = tf.ones_initializer()
                    self.other_weights = tf.Variable(
                        initial_value=other_weights_init(shape=(self.X_other.shape[0], 1), dtype='float64'),
                        trainable=True)
            elif self.w_pde=='LRA':
                self.pde_weights = 1.0
                self.alpha = 0.1
                if self.nb_bc>0:
                    self.bc_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_init>0:
                    self.init_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_data>0:
                    self.data_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_other>0:
                    self.other_weights = tf.Variable(1.0, dtype='float64')
            elif self.w_pde=='NTK':
                self.pde_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_bc > 0:
                    self.bc_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_init > 0:
                    self.init_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_data > 0:
                    self.data_weights = tf.Variable(1.0, dtype='float64')
                if self.nb_other > 0:
                    self.other_weights = tf.Variable(1.0, dtype='float64')
        else:
            self.pde_weights = self.w_pde

        self.slope_recovery = slope_recovery
        self.model_init = model_init

        self.net_transform = net_transform
        if self.model_init is None :
            self.net_u = networks
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
        if self.net_u.method != 'classic':
            print('WARNING: the LBFGS optimizer is not supported in this networks architecture.')

        self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.tf_optimizer_col = tf.keras.optimizers.Adam(learning_rate=lr)
        self.tf_optimizer_init = tf.keras.optimizers.Adam(learning_rate=lr)
        self.tf_optimizer_data = tf.keras.optimizers.Adam(learning_rate=lr)
        self.tf_optimizer_other = tf.keras.optimizers.Adam(learning_rate=lr)

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
        :param param_f: parameters of the PDE
        :type param_f: numpy.ndarray
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
        :type param_f: numpy.ndarray
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
                        if (self.w_pde == 'LRA') or (self.w_pde == 'NTK'):
                            loss_bc += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))*self.bc_weights
                            loss_obs += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))*self.bc_weights
                        else:
                            if self.normalize_data&(tf.reduce_mean(tf.square(u_star_bc[:, i:(i + 1)])) > 0):
                                loss_bc += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_bc[:, i:(i + 1)]))
                                loss_obs += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_bc[:, i:(i + 1)]))
                            else:
                                loss_bc += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)])*self.w_bc)
                                loss_obs += tf.reduce_mean(tf.square(u_pred_bc[:, i:(i+1)] - u_star_bc[:, i:(i+1)])*self.w_bc)
                    else:
                        #a = 1
                        loss_bc += tf.convert_to_tensor(0, dtype='float64')# tf.reduce_mean(tf.square(u_star_bc[:, i:(i + 1)] - u_star_bc[:, i:(i + 1)]))
                        loss_obs += tf.convert_to_tensor(0, dtype='float64')#tf.reduce_mean(tf.square(u_star_bc[:, i:(i+1)] - u_star_bc[:, i:(i+1)]))

            if self.nb_init > 0:
                for i in range(u_star_init.shape[1]):
                    if not tf.math.is_nan(u_star_init[0, i:(i+1)]):
                        if self.w_pde == 'SAPINNs':
                            loss_init += tf.reduce_mean(
                                tf.square((u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]) * self.init_weights))
                            loss_obs += tf.reduce_mean(tf.square((u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)])*self.init_weights))
                        elif (self.w_pde == 'LRA') or (self.w_pde == 'NTK'):
                            loss_init += tf.reduce_mean(
                                tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)])) * self.init_weights
                            loss_obs += tf.reduce_mean(tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]))* self.init_weights
                        else:
                            if self.normalize_data&(tf.reduce_mean(tf.square(u_star_init[:, i:(i + 1)])) > 0):
                                loss_init += tf.reduce_mean(tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_init[:, i:(i + 1)]))
                                loss_obs += tf.reduce_mean(tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_init[:, i:(i + 1)]))
                            else:
                                loss_init += tf.reduce_mean(tf.square(u_pred_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)])*self.w_init)
                                loss_obs += tf.reduce_mean(tf.square(u_pred_init[:, i:(i+1)] - u_star_init[:, i:(i+1)])*self.w_init)
                    else:
                        #a = 1
                        loss_init += tf.convert_to_tensor(0, dtype='float64')# tf.reduce_mean(tf.square(u_star_init[:, i:(i + 1)] - u_star_init[:, i:(i + 1)]))
                        loss_obs += tf.convert_to_tensor(0, dtype='float64')# tf.reduce_mean(tf.square(u_star_init[:, i:(i+1)] - u_star_init[:, i:(i+1)]))
            if self.nb_data > 0:
                for i in range(u_star_data.shape[1]):
                    if not tf.math.is_nan(u_star_data[0, i:(i+1)]):
                        if self.w_pde == 'SAPINNs':
                            loss_data += tf.reduce_mean(
                                tf.square((u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]) * self.data_weights))
                            loss_obs += tf.reduce_mean(tf.square((u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]) * self.data_weights))
                        elif (self.w_pde == 'LRA') or (self.w_pde == 'NTK'):
                            loss_data += tf.reduce_mean(
                                tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)])) * self.data_weights
                            loss_obs += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]))*self.data_weights
                        else:
                            if self.normalize_data&(tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)])) > 0):
                                loss_data += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)]))
                                loss_obs += tf.reduce_mean(tf.square(u_pred_data[:, i:(i+1)] - u_star_data[:, i:(i+1)]))/tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)]))
                            else:
                                loss_data += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)])*self.w_data)
                                loss_obs += tf.reduce_mean(tf.square(u_pred_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)])*self.w_data)
                    else:
                        #a = 1
                        loss_data += tf.convert_to_tensor(0, dtype='float64')# tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)] - u_star_data[:, i:(i + 1)]))
                        loss_obs += tf.convert_to_tensor(0, dtype='float64')#tf.reduce_mean(tf.square(u_star_data[:, i:(i+1)] - u_star_data[:, i:(i+1)]))
            if self.nb_other >0:
                for i in range(u_star_other.shape[1]):
                    if not tf.math.is_nan(u_star_other[0, i:(i+1)]):
                        if self.w_pde == 'SAPINNs':
                            loss_other += tf.reduce_mean(tf.square(
                                (u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]) * self.data_weights))
                            loss_obs += tf.reduce_mean(tf.square((u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]) * self.data_weights))
                        elif (self.w_pde == 'LRA') or (self.w_pde == 'NTK'):
                            loss_other += tf.reduce_mean(
                                tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)])) * self.other_weights
                            loss_obs += tf.reduce_mean(tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]))*self.other_weights
                        else:
                            if self.normalize_data&(tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)])) > 0):
                                loss_other += tf.reduce_mean(tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)]))
                                loss_obs += tf.reduce_mean(tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]))/tf.reduce_mean(tf.square(u_star_data[:, i:(i + 1)]))
                            else:
                                loss_other += tf.reduce_mean(tf.square(u_pred_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)])*self.w_other)
                                loss_obs += tf.reduce_mean(tf.square(u_pred_other[:, i:(i+1)] - u_star_other[:, i:(i+1)])*self.w_other)
                    else:
                        #a = 1
                        loss_other += tf.convert_to_tensor(0, dtype='float64')#tf.reduce_mean(tf.square(u_star_other[:, i:(i + 1)] - u_star_other[:, i:(i + 1)]))
                        loss_obs += tf.convert_to_tensor(0, dtype='float64')#tf.reduce_mean(tf.square(u_star_other[:, i:(i+1)] - u_star_other[:, i:(i+1)]))
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
                            loss_obs += tf.reduce_mean(tf.square(
                                u_star_bc[size_bc * i_param:size_bc * (i_param + 1), i:(i+1)] - u_star_bc[size_bc * i_param:size_bc * (
                                            i_param + 1), i:(i+1)]))

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
                            loss_obs += tf.reduce_mean(tf.square(
                                    u_star_init[size_init * i_param:size_init * (i_param + 1), i:(i+1)] - u_star_init[size_init * i_param:size_init * (
                                                i_param + 1), i:(i+1)]))

                if self.nb_data > 0:
                    size_data = int(u_star_data.shape[0] / self.nb_param)
                    for i in range(u_star_data.shape[1]):
                        if not tf.math.is_nan(u_star_data[size_data * i_param:size_data * (i_param + 1), i:(i + 1)][0]):
                            loss_obs += tf.reduce_mean(tf.square(
                                u_pred_data[size_data * i_param:size_data * (i_param + 1), i:(i+1)] - u_star_data[size_data * i_param:size_data * (
                                            i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.reduce_mean(tf.square(
                                u_star_data[size_data * i_param:size_data * (i_param + 1), i:(i+1)] - u_star_data[size_data * i_param:size_data * (
                                            i_param + 1), i:(i+1)]))

                if self.nb_other > 0:
                    size_other = int(u_star_other.shape[0] / self.nb_param)
                    for i in range(u_star_data.shape[1]):
                        if not tf.math.is_nan(u_star_other[size_other * i_param:size_other * (i_param + 1), i:(i + 1)][0]):
                            loss_obs += tf.reduce_mean(tf.square(
                                u_pred_other[size_other * i_param:size_other * (i_param + 1), i:(i+1)] - u_star_other[size_other * i_param:size_other * (
                                            i_param + 1), i:(i+1)]))
                        else:
                            loss_obs += tf.reduce_mean(tf.square(
                                u_star_other[size_other * i_param:size_other * (i_param + 1), i:(i+1)] - u_star_other[size_other * i_param:size_other * (
                                            i_param + 1), i:(i+1)]))

                index_i_param = tf.where(X_f[:, -1] == param_f[i_param])
                index_i_param = tf.reshape(index_i_param, [-1])
                f_i = tf.gather(f, index_i_param)

                loss_f += self.loss_f(f_i)
        if self.w_pde=='SAPINNs':
            loss = loss_obs + tf.reduce_mean(tf.square(f*self.pde_weights))
        elif self.w_pde=='NTK':
            loss = loss_obs + tf.reduce_mean(tf.square(f))*self.pde_weights
        else:
            loss = loss_obs + loss_f*pde_weights
        if self.slope_recovery:
            sum_exp = 0
            a_loss = self.net_u.trainable_variables[2::3]
            for i in range(len(self.net_u.layers)-1):
                sum_exp += tf.exp(tf.reduce_mean(a_loss[i]))
            loss += 1.0 / (tf.reduce_mean(sum_exp))
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
        :type param_f: numpy.ndarray

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
            if self.w_pde=='SAPINNs':
                if (self.epoch + 1) % self.period_w_pde == 0:
                    grads_col = tape.gradient(loss_value, self.pde_weights)
                    if self.nb_init > 0:
                        grads_init = tape.gradient(loss_value, self.init_weights)
                    else:
                        grads_init = 1
                    if self.nb_data > 0:
                        grads_data = tape.gradient(loss_value, self.data_weights)
                    else:
                        grads_data = 1
                    if self.nb_other > 0:
                        grads_other = tape.gradient(loss_value, self.other_weights)
                    else:
                        grads_other = 1
            elif self.w_pde == 'LRA':
                if (self.epoch + 1) % self.period_w_pde == 0:
                    grads_res = []
                    grads_bc = []
                    grads_init = []
                    grads_data = []
                    grads_other = []

                    for i in range(len(self.net_u.trainable_variables[::2]) - 1):
                        grads_res.append(tape.gradient(loss_f, self.net_u.trainable_variables[::2][i]))
                        if self.nb_bc > 0:
                            grads_bc.append(tape.gradient(loss_bc, self.net_u.trainable_variables[::2][i]))
                        if self.nb_init > 0:
                            grads_init.append(tape.gradient(loss_init, self.net_u.trainable_variables[::2][i]))
                        if self.nb_data > 0:
                            grads_data.append(tape.gradient(loss_data, self.net_u.trainable_variables[::2][i]))
                        if self.nb_other > 0:
                            grads_other.append(tape.gradient(loss_other, self.net_u.trainable_variables[::2][i]))

                    max_grads_res_list = []
                    mean_grads_bc_list = []
                    mean_grads_init_list = []
                    mean_grads_data_list = []
                    mean_grads_other_list = []

                    for i in range(len(self.net_u.trainable_variables[::2]) - 1):
                        max_grads_res_list.append(tf.reduce_max(tf.abs(grads_res[i])))
                        if self.nb_bc > 0:
                            mean_grads_bc_list.append(tf.reduce_mean(tf.abs(grads_bc[i])))
                        if self.nb_init > 0:
                            mean_grads_init_list.append(tf.reduce_mean(tf.abs(grads_init[i])))
                        if self.nb_data > 0:
                            mean_grads_data_list.append(tf.reduce_mean(tf.abs(grads_data[i])))
                        if self.nb_other > 0:
                            mean_grads_other_list.append(tf.reduce_mean(tf.abs(grads_other[i])))

                    max_grads_res = tf.reduce_max(tf.stack(max_grads_res_list))
                    if self.nb_bc > 0:
                        mean_grads_bc = tf.reduce_mean(tf.stack(mean_grads_bc_list))
                    if self.nb_init > 0:
                        mean_grads_init = tf.reduce_mean(tf.stack(mean_grads_init_list))
                    if self.nb_data > 0:
                        mean_grads_data = tf.reduce_mean(tf.stack(mean_grads_data_list))
                    if self.nb_other > 0:
                        mean_grads_other = tf.reduce_mean(tf.stack(mean_grads_other_list))

                    if self.nb_bc > 0:
                        bc_weights = max_grads_res / mean_grads_bc
                        self.bc_weights.assign(self.bc_weights * (1 - self.alpha) + self.alpha * bc_weights)
                    if self.nb_init > 0:
                        init_weights = max_grads_res / mean_grads_init
                        self.init_weights.assign(self.init_weights * (1 - self.alpha) + self.alpha * init_weights)
                    if self.nb_data > 0:
                        data_weights = max_grads_res / mean_grads_data
                        self.data_weights.assign(self.data_weights * (1 - self.alpha) + self.alpha * data_weights)
                    if self.nb_other > 0:
                        other_weights = max_grads_res / mean_grads_other
                        self.other_weights.assign(self.other_weights * (1 - self.alpha) + self.alpha * other_weights)
            elif self.w_pde=='NTK':
                if (self.epoch+1) % self.period_w_pde == 0:
                    J_r = compute_J(X_f, self.net_pde, self.net_u)
                    K_rr = compute_Ki(X_f, J_r)
                    trace_K_rr = tf.linalg.trace(K_rr)
                    if self.nb_bc > 0:
                        J_bc = compute_Jbc(self.X_bc, self.net_bc, self.net_u)
                        K_bc = compute_Ki(self.X_bc, J_bc)
                        trace_K_bc = tf.linalg.trace(K_bc)
                    else:
                        trace_K_bc = 0
                    if self.nb_init > 0:
                        J_init = compute_Jinit(self.X_init, self.net_init, self.net_u)
                        K_init = compute_Ki(self.X_init, J_init)
                        trace_K_init = tf.linalg.trace(K_init)
                    else:
                        trace_K_init = 0
                    if self.nb_data > 0:
                        J_data = compute_Jdata(self.X_data, self.net_transform, self.net_u)
                        K_data = compute_Ki(self.X_data, J_data)
                        trace_K_data = tf.linalg.trace(K_data)
                    else:
                        trace_K_data = 0
                    if self.nb_other > 0:
                        J_other = compute_Jother(self.X_other, self.net_other, self.net_u)
                        K_other = compute_Ki(self.X_other, J_other)
                        trace_K_other = tf.linalg.trace(K_other)
                    else:
                        trace_K_other = 0
                    trace_all = trace_K_rr + trace_K_bc + trace_K_init + trace_K_data + trace_K_other

                    self.pde_weights.assign(trace_all/trace_K_rr)
                    if self.nb_bc > 0:
                        self.bc_weights.assign(trace_all/trace_K_bc)
                    if self.nb_init > 0:
                        self.init_weights.assign(trace_all/trace_K_init)
                    if self.nb_data > 0:
                        self.data_weights.assign(trace_all/trace_K_data)
                    if self.nb_other > 0:
                        self.other_weights.assign(trace_all/trace_K_other)

        if self.w_pde == 'SAPINNs':
                return loss_value,loss_bc, loss_init, loss_data, loss_other, loss_f, grads, grads_col, grads_init, grads_data, grads_other
        else:
            return loss_value,loss_bc, loss_init, loss_data, loss_other, loss_f, grads

    def resampling(self, method='FBOAL', X_domain=None,
                 m_FBOAL=None, square_side_FBOAL=None, k_RAD=None, c_RAD=None, k_RARD=None, c_RARD=None, m_RARD=None):
        """
        Re-sampling strategy for collocation points

        :param method: Adaptive strategy (supported choices: 'FBOAL', 'RARD', 'RAD')
        :type method: str
        :param X_domain: Points inside domain to generate new collocation points for adaptive resampling strategy.
        :type X_domain: numpy.ndarray
        :param m_FBOAL: Number of added and removed points when choosing FBOAL strategy.
        :type m_FBOAL: int
        :param square_side_FBOAL: Side of sub-domains (squares). when choosing FBOAL strategy.
        :type square_side_FBOAL: int
        :param k_RAD: Hyper-parameter of RAD to define collocation points distribution.
        :type k_RAD: int
        :param c_RAD: Hyper-parameter of RAD to define collocation points distribution.
        :type c_RAD: int
        :param k_RARD: Hyper-parameter of RARD to define collocation points distribution.
        :type k_RARD: int
        :param c_RARD: Hyper-parameter of RARD to define collocation points distribution.
        :type c_RARD: int
        :param m_RARD: Number of added and removed points when choosing RARD strategy..
        :type m_RARD: int
        """
        if method == 'FBOAL':
            if m_FBOAL == None or square_side_FBOAL == None:
                raise TypeError("Must provide m_FBOAL and square_side_FBOAL")
            FBOAL_resampling = FBOAL(X_domain, m_FBOAL, square_side_FBOAL, self.X_colloc,
                                     self.param_pde, self.net_u, self.net_pde)
            self.X_colloc = FBOAL_resampling.resampling()
        elif method == 'RAD':
            if k_RAD == None or c_RAD == None:
                raise TypeError("Must provide k_RAD and c_RAD")
            RAD_resampling = RAD(X_domain, k_RAD, c_RAD, self.X_colloc, self.param_pde,
                                 self.net_u, self.net_pde)
            self.X_colloc = RAD_resampling.resampling()
        elif method == 'RARD':
            if k_RARD == None or c_RARD == None or m_RARD == None:
                raise TypeError("Must provide k_RARD, c_RARD and m_RARD")
            RARD_resampling = RARD(X_domain, k_RARD, c_RARD, m_RARD, self.X_colloc,
                                   self.param_pde, self.net_u, self.net_pde)
            self.X_colloc = RARD_resampling.resampling()
        elif method == 'Evo':
            Evo_resampling = Evo(X_domain,self.X_colloc, self.param_pde,
                                 self.net_u, self.net_pde)
            self.X_colloc = Evo_resampling.resampling()
        elif method == 'Dynamic':
            Dynamic_resampling = Dynamic(X_domain,self.X_colloc, self.param_pde,
                                 self.net_u, self.net_pde)
            self.X_colloc = Dynamic_resampling.resampling()
        else:
            print(
                'The adaptive sampling strategy is not supported, collocation points are fixed during the training')

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
            if self.w_pde=='SAPINNs':
                loss_value_,loss_bc_, loss_init_, loss_data_, loss_other_, loss_f_, grads, grads_col, grads_init, grads_data, grads_other = self.get_grad(X_f, param_f)
                self.tf_optimizer.apply_gradients(
                    zip(grads, self.pinns_training_variables()))
                self.tf_optimizer_col.apply_gradients(zip([-grads_col], [self.pde_weights]))
                if self.nb_init > 0:
                    self.tf_optimizer_init.apply_gradients(zip([-grads_init], [self.init_weights]))
                if self.nb_data > 0:
                    self.tf_optimizer_data.apply_gradients(zip([-grads_data], [self.data_weights]))
                if self.nb_other > 0:
                    self.tf_optimizer_other.apply_gradients(zip([-grads_other], [self.other_weights]))
            else:
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
                        new_variable = tf.reshape(list_weight[index:index + len_weights], (variable.shape[0], variable.shape[1]))
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
                if self.w_pde=='SAPINNs':
                    loss_value, loss_bc, loss_init, loss_data, loss_other, loss_f, grad, grads_col, grads_init, grads_data, grads_other = self.get_grad(X_f, param_f)
                else:
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

        if max_epochs_lbfgs>0:
            if self.net_u.method=='classic':
                if max_epochs_adam==0:
                    draft = self.net_u(self.X_colloc)

                optimizer_lbfgs(self.X_colloc, self.param_pde,
                                              method='L-BFGS-B',
                                              options={'maxiter': max_epochs_lbfgs,
                                                       'maxfun': max_epochs_lbfgs,
                                                       'maxcor': 100,
                                                       'maxls': 100,
                                                       'ftol': 0,
                                                       'gtol': 1.0 * np.finfo(float).eps})