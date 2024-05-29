import os
import numpy as np
import time
from math import *
import tensorflow as tf
import shapely
from shapely.geometry import Point, Polygon

class FBOAL:
    """
    FBOAL main class

    Parameters
    ----------
    X_domain : numpy.ndarray
               Points inside domain to generate new collocation points for adaptive resampling strategy.
    m: int
       Number of added and removed points.
    square_side: float
                 Side of sub-domains (squares).
    X_f: numpy.ndarray
         Initial collocation points.
    param_pde: float
               Parameter of the PDE.
    model_nn:
               Neural networks.
    f_user:
            PDE defined by users.

    Returns
    -------
    Instantiate a FBOAL caller.

    """
    def __init__(self, X_domain, m, square_side, X_f, param_pde, model_nn, f_user):
        """
        Initialisation function of FBOAL
        """
        self.m = m
        self.square_side = square_side
        self.X_f = X_f
        self.param_pde = param_pde
        self.model_nn = model_nn
        self.f_user = f_user
        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float64')

    def define_rec(self, X_verify_rec):
        """
        Define the sub-domains and the training collocation points lying in each sub-domain

        :param X_verify_rec: Training collocation points
        :type X_verify_rec: numpy.ndarray

        :return: Array that includes index of collocation points in each sub-domain

        :meta private:

        """
        x_rec = np.arange(np.min(X_verify_rec[:, 0]), np.max(X_verify_rec[:, 0]), self.square_side)
        y_rec = np.arange(np.min(X_verify_rec[:, 1]), np.max(X_verify_rec[:, 1]), self.square_side)
        index_in_array = []
        for i in range(x_rec.shape[0]):
            for j in range(y_rec.shape[0]):
                poly = Polygon([(x_rec[i] - 10 ** (-4), y_rec[j] - 10 ** (-4)),
                                (x_rec[i] - 10 ** (-4), y_rec[j] + self.square_side + 10 ** (-4)),
                                (x_rec[i] + self.square_side + 10 ** (-4), y_rec[j] + self.square_side + 10 ** (-4)),
                                (x_rec[i] + self.square_side + 10 ** (-4), y_rec[j] - 10 ** (-4))])
                contains = np.vectorize(lambda p: poly.contains(Point(p)), signature='(n)->()')
                index_in = np.where(contains(X_verify_rec) * 1 == 1)[0]
                index_in_array.append(index_in)
        return index_in_array

    def setdiff2d_bc(self, arr1, arr2):
        """
        Define the points in array 1 that are not in array 2

        :param arr1: Array 1
        :type arr1: numpy.ndarray
        :param arr2: Array 2
        :type arr2: numpy.ndarray

        :return: Points in array 1 that are not in array 2

        :meta private:

        """
        arr1_np = arr1.numpy()
        arr2_np = arr2.numpy()
        idx = (arr1_np[:, None] != arr2_np).any(-1).all(1)
        return tf.convert_to_tensor(arr1_np[idx], dtype='float64')

    def resampling(self):
        """
        Sampling (collocation) points

        Returns
        -------
        X_f: numpy.ndarray
             Collocation points after resampling
        """
        f_test = self.f_user(self.X_domain, self.model_nn, self.param_pde)  # .numpy()#self.net_burgers(X_colloc_rand_test)
        index_in_add_array = self.define_rec(self.X_domain.numpy())  # self.define_rec(X_colloc_rand_test.numpy())

        X_colloc_rec_max = []
        f_rec_max = []
        for i in range(np.shape(index_in_add_array)[0]):
            X_test_rec_i = tf.gather(self.X_domain, index_in_add_array[i])
            f_i_rec = tf.gather(f_test, index_in_add_array[i])
            pde_res = tf.reshape(tf.abs(f_i_rec), (X_test_rec_i.shape[0],))
            if self.param_pde is None:
                index_max = tf.math.top_k(pde_res, k=1).indices
            else:
                index_max = tf.math.top_k(pde_res, k=1*self.param_pde.shape[0]).indices
            X_colloc_rec_i_max = tf.gather(X_test_rec_i, index_max)
            f_i_rec_max = tf.gather(f_i_rec, index_max)
            X_colloc_rec_max.append(X_colloc_rec_i_max)
            f_rec_max.append(f_i_rec_max)

        X_colloc_rec_max_concat = tf.concat((X_colloc_rec_max), axis=0)
        f_rec_max_concat = tf.concat((f_rec_max), axis=0)
        pde_res_concat = tf.reshape(tf.abs(f_rec_max_concat), (X_colloc_rec_max_concat.shape[0],))
        index_max_final = tf.math.top_k((pde_res_concat), k=self.m).indices
        X_colloc_rec_max_final = tf.gather(X_colloc_rec_max_concat, index_max_final)

        f_colloc = self.f_user(self.X_f, self.model_nn, self.param_pde)  # self.net_burgers(self.X_colloc)
        index_in_rm_array = self.define_rec(self.X_f.numpy())
        X_colloc_rec_min = []
        f_rec_min = []
        for i in range(np.shape(index_in_rm_array)[0]):
            X_colloc_rec_i = tf.gather(self.X_f, index_in_rm_array[i])
            f_i_rec = tf.gather(f_colloc, index_in_rm_array[i])
            pde_res = tf.reshape(tf.abs(f_i_rec), (X_colloc_rec_i.shape[0],))
            if tf.shape(pde_res)[0] < 1:
                continue
            else:
                if self.param_pde is None:
                    index_min = tf.math.top_k(-pde_res, k=1).indices
                else:
                    index_min = tf.math.top_k(-pde_res, k=1*self.param_pde.shape[0]).indices
                X_colloc_rec_i_min = tf.gather(X_colloc_rec_i, index_min)
                f_i_rec_min = tf.gather(f_i_rec, index_min)
                X_colloc_rec_min.append(X_colloc_rec_i_min)
                f_rec_min.append(f_i_rec_min)

        X_colloc_rec_min_concat = tf.concat((X_colloc_rec_min), axis=0)
        f_rec_min_concat = tf.concat((f_rec_min), axis=0)
        pde_res_concat = tf.reshape(tf.abs(f_rec_min_concat), (X_colloc_rec_min_concat.shape[0],))
        index_min_final = tf.math.top_k(-pde_res_concat, k=self.m).indices
        X_colloc_rec_min_final = tf.gather(X_colloc_rec_min_concat, index_min_final)

        self.X_f = self.setdiff2d_bc(self.X_f, X_colloc_rec_min_final)
        self.X_f = tf.concat([self.X_f, X_colloc_rec_max_final], axis=0)
        return self.X_f

class RAD:
    """
    RAD main class

    Parameters
    ----------
    X_domain : numpy.ndarray
               Points inside domain to generate new collocation points for adaptive resampling strategy.
    k: float
       Hyper-parameter of RAD to define collocation points distribution.
    c: float
       Hyper-parameter of RAD to define collocation points distribution.
    X_f: numpy.ndarray
         Initial collocation points
    param_pde: numpy.ndarray
               Parameters of the PDE
    model_nn:
              neural networks
    f_user:
            PDE defined by users

    """
    def __init__(self, X_domain, k, c, X_f, param_pde, model_nn, f_user):
        """
        Initialisation function of RAD
        """
        self.k = k
        self.c = c
        self.X_f = X_f
        self.param_pde = param_pde
        self.model_nn = model_nn
        self.f_user = f_user
        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float64')

    def resampling(self):
        """
        Sampling (collocation) points

        Returns
        -------
        X_f: numpy.ndarray
             Collocation points after resampling
        """
        Y = np.abs(self.f_user(self.X_domain, self.model_nn, self.param_pde).numpy())
        err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=self.X_domain.shape[0], size=self.X_f.shape[0], replace=False, p=err_eq_normalized)
        X_selected = tf.gather(self.X_domain, X_ids)
        return X_selected

class RARD:
    """
    RARD main class

    Parameters
    ----------
    X_domain : numpy.ndarray
               Points inside domain to generate new collocation points for adaptive resampling strategy.
    k: float
       Hyper-parameter of RARD to define collocation points distribution.
    c: float
       Hyper-parameter of RARD to define collocation points distribution.
    m: int
       Number of added points.
    X_f: numpy.ndarray
         Initial collocation points
    param_pde: numpy.ndarray
               Parameters of the PDE
    model_nn:
              neural networks
    f_user:
            PDE defined by users
    """
    def __init__(self, X_domain, k, c, m, X_f, param_pde, model_nn, f_user):
        """
        Initialisation function of RARD
        """
        self.k = k
        self.c = c
        self.m = m
        self.X_f = X_f
        self.param_pde = param_pde
        self.model_nn = model_nn
        self.f_user = f_user
        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float64')

    def resampling(self):
        """
        Sampling (collocation) points

        Returns
        -------
        X_f: numpy.ndarray
             Collocation points after resampling
        """
        Y = np.abs(self.f_user(self.X_domain, self.model_nn, self.param_pde).numpy())
        err_eq = np.power(Y, self.k) / np.power(Y, self.k).mean() + self.c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=self.X_domain.shape[0], size=self.m, replace=False, p=err_eq_normalized)
        X_selected = tf.gather(self.X_domain, X_ids)
        return tf.concat([self.X_f, X_selected], axis=0)

class Evo:
    """
    Evo main class

    Parameters
    ----------
    X_domain : numpy.ndarray
               Points inside domain to generate new collocation points for adaptive resampling strategy.
    X_f: numpy.ndarray
         Initial collocation points
    param_pde: numpy.ndarray
               Parameters of the PDE
    model_nn:
              neural networks
    f_user:
            PDE defined by users

    """
    def __init__(self, X_domain, X_f, param_pde, model_nn, f_user):
        """
        Initialisation function of Evo
        """

        self.X_f = X_f
        self.param_pde = param_pde
        self.model_nn = model_nn
        self.f_user = f_user
        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float64')

    def resampling(self):
        """
        Sampling (collocation) points

        Returns
        -------
        X_f: numpy.ndarray
             Collocation points after resampling
        """
        Y = np.abs(self.f_user(self.X_f, self.model_nn, self.param_pde).numpy())
        thres = np.mean(Y)
        index_chosen = np.where(Y>thres)[0]
        X_chosen = tf.gather(self.X_f, index_chosen) #self.X_f[index_chosen]
        index_new = np.random.choice(self.X_domain.shape[0], int(self.X_f.shape[0]- X_chosen.shape[0]), replace=False)
        X_new = tf.gather(self.X_domain, index_new)
        return tf.concat([X_chosen, X_new], axis=0)

class Dynamic:
    """
    Dynamic main class

    Parameters
    ----------
    X_domain : numpy.ndarray
               Points inside domain to generate new collocation points for adaptive resampling strategy.
    X_f: numpy.ndarray
         Initial collocation points
    param_pde: numpy.ndarray
               Parameters of the PDE
    model_nn:
              neural networks
    f_user:
            PDE defined by users

    """
    def __init__(self, X_domain, X_f, param_pde, model_nn, f_user):
        """
        Initialisation function of Dynamical resampling
        """

        self.X_f = X_f
        self.param_pde = param_pde
        self.model_nn = model_nn
        self.f_user = f_user
        self.X_domain = tf.convert_to_tensor(X_domain, dtype='float64')

    def resampling(self):
        """
        Sampling (collocation) points

        Returns
        -------
        X_f: numpy.ndarray
             Collocation points after resampling
        """
        index_new = np.random.choice(self.X_domain.shape[0], self.X_f.shape[0], replace=False)
        X_new = tf.gather(self.X_domain, index_new)
        return X_new