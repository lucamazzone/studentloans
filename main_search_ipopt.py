############################################################################################
############################# MAIN CODE FOR STUDENT LOANS PROJECT ##########################
#############################    MODEL WITH SEARCH ON THE JOB     ##########################
############################################################################################

import numpy as np
#import sys
#import os
#import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import numpy.matlib
from scipy.optimize import fsolve
#from scipy.optimize import brentq
#from scipy.interpolate import griddata
#from scipy.interpolate import CubicSpline
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C
import warnings
#from numpy import linalg as LA
from numba import jit
import random
import pyipopt

from params import *
from declare_arrays import *


############################################################################################
############################################################################################

############################ DECLARATIONS : GENERAL USE FUNCTIONS ############################

def m_fun(theta):
    return 1 - np.exp(-eta * theta)

def m_funprime(theta):
    return eta * np.exp(-eta * theta)

def q_fun(theta):
    if (theta==0.0):
        return 1.0
    else:
        return (1 - np.exp(-eta * theta))/theta

def q_funprime(theta):
    if (theta==0.0):
        return 1.0
    else:
        return -np.exp(-eta * theta)*(np.exp(eta * theta)-eta*theta-1)/theta**2
    # -(1 / (theta ** 2)) + ((1+eta*theta) * np.exp(-eta * theta))/(theta**2)

def u(c):
    if nu == 1:
        return np.log(c)
    else:
        return (c ** (1 - nu)) / (1 - nu)

def uprime_inv(c):
    return c ** (-1 / nu)

def u_prime(c):
    return c ** (-nu)

def f( A, y, x):
    return A * (y**alpha  + g_fun(y, x)**alpha )**(1/alpha) #A * (y**alpha)*g_fun(y, x)**(1-alpha)

def f_prime(A, y, x):  # rename as f_y
    return  y**(alpha-1)*A*(y**alpha  + g_fun(y, x)**alpha )**(1/alpha-1)  #A * (alpha * y ** (alpha - 1)*g_fun(y, x)**(1-alpha) +
            #    (1-alpha)*g_fun_prime(y, x)*(y**alpha)*g_fun(y, x)**(-alpha))

def k_fun(y):
    return kappa*(y**gamma)/gamma

def k_funprime(y):
    return kappa*(y**(gamma-1))

def c_u(a, a_prime):
    return b + R * a - a_prime

def c_e(w, a, a_prime):
    return w + R * a - a_prime

def wage(y, theta, x, t):
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    return f(A, y, x) - k_fun(y)/(cosa1*q_fun(theta)) -  (max(y - g_fun(y,x),0))*5


def E_fun(U, E, a, a_prime, w,theta,E_tilde): #
    return u(c_e(w, a, a_prime)) + beta*(1-pi*m_fun(theta))*((1-lamb)*E + lamb*U) + beta*pi*m_fun(theta)*E_tilde


def E_funprime(E_prime, a, a_prime, w):
    return u_prime(c_e(w, a, a_prime)) + beta*(1-lamb)*E_prime   #E_prime function of a_prime

def U_fun(U, E, a, a_prime, theta):
    return u(c_u(a, a_prime)) + beta*(m_fun(theta)*E + (1-m_fun(theta))*U)


def l_fun(dis):
    if np.abs(dis):
        dis = dis/np.abs(dis)
    return 1/(1+np.exp(-2*100*dis))

def l_fun_prime(dis):
    if np.abs(dis):
        dis = dis/np.abs(dis)
    return 2*100*np.exp(2*100*dis)/(1+np.exp(2*100*dis))**2

def g_fun(p, x):
    y = p*adj
    return x + phi*(y-x)*l_fun(y-x)

def g_fun_prime(p, x):
    y = p*adj
    return phi*l_fun(y-x) + phi*(y-x)*l_fun_prime(y-x)

############################ DECLARATIONS : UNEMPLOYED FUNCTIONS ############################

def eval_Uf(xx, user_data= state):
    assert len(xx) == nvar
    a = state[0]
    x = state[1]
    t = state[2]

    a_prime = xx[0]
    theta = xx[1]
    y = xx[2]

    w_wage = wage(y, theta, x, t)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)), 0)

    return -u(c_u(a, a_prime)) - beta * (
    m_fun(theta) * E_fun(UU, EE, a_prime, a_emp_int, w_wage, theta, EE_tilde)
    + (1 - m_fun(theta)) * U_fun(UU, EE, a_prime, a_int, theta))

def eval_grad_Uf(x, user_data = state):
    assert len(x) == nvar
    # finite differences
    A = np.empty((nvar * ncon))
    xAdj = x
    Fx1 = eval_Uf(xAdj,state)
    h = 1e-4
    for ii in range(nvar):
        xAdj = x
        xAdj[ii] = xAdj[ii] + h
        Fx2 = eval_Uf(xAdj,state)
        A[ii] = (Fx2 - Fx1) / h
        xAdj[ii] = xAdj[ii] - h
    return A #grad_f

def eval_g(x, user_data= None):
    assert len(x) == nvar
    return np.array([0.0])


def eval_jac_g(x, flag, user_data = None):
    if flag:
        return (np.array([0, 0, 0]),
            np.array([0, 1, 2]))

    else:
        assert len(x) == 3
        return np.array([
        0.0,
        0.0,
        0.0
        ])


def foccs(thetas, params):
    a = params[0]
    x = params[1]
    t = params[2]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y = thetas[2]
    w_wage = wage(y, theta, g_fun(y,x), t)
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    ##
    cu = c_u(a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)

    ##

    foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut))
    foc2 = m_funprime(theta) * ( E_fun(UU, EE, a_prime, a_emp_int, w_wage,theta_tilde,E_prime_tilde)
    - U_fun(UU, EE, a_prime, a_int, theta) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)

    return [foc1.item(), foc2.item(), foc3]


def foccs_constrained(thetas, params):
    a = params[0]
    x = params[1]
    t = params[2]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y = thetas[2]
    vincolo = thetas[3]
    w_wage = wage(y, theta, g_fun(y,x), t)
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    theta_tilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))

    ##
    cu = c_u(a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)

    ##

    foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) +\
        np.amax([0, +vincolo])
    foc2 = m_funprime(theta) * ( E_fun(UU, EE, a_prime, a_emp_int, w_wage,theta_tilde,E_prime_tilde)
    - U_fun(UU, EE, a_prime, a_int, theta) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)
    foc4 = np.amax([0, -vincolo]) - a_prime

    return [foc1.item(), foc2.item(), foc3, foc4]  #np.array().ravel()


def unemp_solver(x0, state):
    nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_Uf, eval_grad_Uf, eval_g, eval_jac_g)
    nlp.num_option('tol', 1e-8)
    nlp.int_option('print_level', 1)
    nlp.str_option('limited_memory_update_type', 'bfgs')
    # nlp.str_option('linear_solver', 'mumps')

    kk, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    nlp.close()

    return [kk[0],kk[1],kk[2], status]

############################ DECLARATIONS : EMPLOYED FUNCTIONS ############################




def eval_Ef(xx, user_data= e_state):
    assert len(xx) == nvar
    a = e_state[0]
    x = e_state[1]
    w = e_state[2]
    y = e_state[3]
    t = e_state[4]

    a_prime = xx[0]
    theta = xx[1]
    y_tilde = xx[2]

    w_wage = wage(y_tilde, theta, x, t)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
    a_emp_tilde =  np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,
                                                                g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)), 0)
    theta_int = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x)]), return_std=False))
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x)]), return_std=False))

    return  -u(c_e(w, a, a_prime)) - \
           beta * ( (1-pi*m_fun(theta))*((1-lamb)*E_fun(UU, EE, a_prime, a_emp_int, w, theta_int, EE_tilde) +
                                             lamb * U_fun(UU, EE, a_prime, a_int, theta_int)) +
                                pi*m_fun(theta)*E_fun(UU, EE, a_prime, a_emp_tilde, w_wage, theta_tilde, EE_tilde))


def eval_grad_Ef(x, user_data = e_state):
    assert len(x) == nvar
    # finite differences
    A = np.empty((nvar * ncon))
    xAdj = x
    Fx1 = eval_Ef(xAdj,e_state)
    h = 1e-4
    for ii in range(nvar):
        xAdj = x
        xAdj[ii] = xAdj[ii] + h
        Fx2 = eval_Ef(xAdj,e_state)
        A[ii] = (Fx2 - Fx1) / h
        xAdj[ii] = xAdj[ii] - h
    return A #grad_f


def emp_solver(x0, e_state):
    nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_Ef, eval_grad_Ef, eval_g, eval_jac_g)
    nlp.num_option('tol', 1e-8)
    nlp.int_option('print_level', 2)
    nlp.str_option('limited_memory_update_type', 'bfgs')
    nlp.str_option('linear_solver', 'mumps')

    kk, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    nlp.close()

    return [kk[0],kk[1],kk[2], obj, status]


def foc_empl(thetas,other):
    a = other[0] #np.asscalar()
    x = other[1] # np.asscalar()
    w = other[2] #np.asscalar()
    y = other[3] #np.asscalar()
    t = other[4]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y_tilde = thetas[2]
    w_wage = wage(y_tilde, theta, g_fun(y_tilde, x), t)

    a_prime_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w, y]), return_std=False))
    a_prime_tilde = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde,x), w_wage, y_tilde]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde,x), w_wage, y_tilde]), return_std=False))
    theta_int = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w, y]), return_std=False))


    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_int,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_int,g_fun(y,x), w, y ]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_tilde,g_fun(y_tilde,x), w_wage, y_tilde ]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime_tilde,g_fun(y_tilde,x), w_wage,y_tilde ]), return_std=False))


    ce = c_e(w, a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w, a_prime, a_prime_int)
    cetilde = c_e(w_wage, a_prime, a_prime_tilde)

    foc1 = -u_prime(ce) + (1-pi*m_fun(theta))*beta * R * ((1 - lamb) * u_prime(cet)+
                                      (lamb) * u_prime(cut)) + beta*R*pi*m_fun(theta)*u_prime(cetilde)
    foc2 = beta*pi*m_funprime(theta) * ( E_fun(UU, EE_tilde, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde)
    -lamb*U_fun(UU, EE, a_prime, a_int, theta_int) - (1-lamb)*E_fun(UU, EE, a_prime, a_prime_int, w, theta_int, EE_tilde)) + \
     theta*pi*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y_tilde, x)*q_fun(theta)*cosa1 - k_funprime(y_tilde)

    return [foc1.item(), foc2.item(), foc3]



############################################################################################
############################################################################################

############################ DECLARATIONS : UNEMPLOYED PROBLEM ############################

nvar = 3
ncon = 1
x_L = np.ones((nvar)) * 0.0  #, dtype=float_
x_U = np.ones((nvar)) * 20.0 # , dtype=float_

g_L = np.array([0.0])
g_U = np.array([0.0])
nnzj = 3
nnzh = 3

############################ SETUP : VALUE / POLICY FUNCTIONS  ############################

E_func = {}
a_func =  {}
U_func =  {}
Eprime_func =  {}
a_emp_func =  {}
y_func = {}
theta_func = {}
y_emp_func = {}
theta_emp_func = {}

for i in range(1,life+1):
        E_func[i] = 'E_func_'+str(i)
        U_func[i] =  'U_func_'+str(i)
        Eprime_func[i] = 'Eprime_func_'+str(i)
        a_func[i] =  'a_func_'+str(i)
        a_emp_func[i] = 'a_emp_func_'+str(i)
        y_func[i] = 'y_func_' + str(i)
        theta_func[i] = 'theta_func_' + str(i)
        y_emp_func[i] = 'y_emp_func_' + str(i)
        theta_emp_func[i] = 'theta_emp_func_' + str(i)


status_collect = np.ones((rsize, life))
status_collect_emp = np.ones((rsize, life))

kernel = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e2))
kernel_bis = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e3)) + WhiteKernel(noise_level=1e-3,
                                                                      noise_level_bounds=(1e-10, 1e+1))
kernel_E = 1.0 * RBF(7.0, (1e-3, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e+1))


controls_unemp = np.where(status_collect[:, 0] < 1.0)
controls_emp = np.where( status_collect_emp[:,0] < 1.0 )
joined =  controls_unemp + controls_emp
joint = [x for xs in joined for x in xs]


for t in range(1,life+1):

    print("THIS IS AGE", life-t)

    U_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    a_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    E_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=5)
    Eprime_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=10)
    a_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    y_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    theta_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    y_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    theta_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)

    if any(joint) and (t>1):  # [0]
        print("status collector is not empty:")
        print('we have to change sizes')
        unempl_grid_old_b = np.delete(unempl_grid, controls_unemp, axis=0)
        empl_grid_old_b = np.delete(empl_grid, controls_emp, axis=0)

        a_star_old_b = np.delete(a_star_old, controls_unemp, axis=0)
        wage_star_old_b = np.delete(wage_star_old, controls_unemp, axis=0)
        theta_star_old_b = np.delete(theta_star_old, controls_unemp, axis=0)
        y_star_old_b = np.delete(y_star_old, controls_unemp, axis=0)
        a_star_emp_old_b = np.delete(a_star_emp_old, controls_emp, axis=0)
        wage_star_emp_old_b = np.delete(wage_star_emp_old, controls_emp, axis=0)
        theta_star_emp_old_b = np.delete(theta_star_emp_old, controls_emp, axis=0)
        y_star_emp_old_b = np.delete(y_star_emp_old, controls_emp, axis=0)
        U_old_b = np.delete(U_old, controls_unemp, axis=0)
        E_old_b = np.delete(E_old, controls_emp, axis=0)
        E_prime_old_b = np.delete(E_prime_old, controls_emp, axis=0)
        rsize_old = unempl_grid_old_b.shape[0]
        print('new unemp row size is', rsize_old)
        rsize_old = empl_grid_old_b.shape[0]
        print('new emp row size is', rsize_old)
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # if you get useless warnings print "ignore"
            U_func[t].fit(unempl_grid_old_b, U_old_b)
            a_func[t].fit(unempl_grid_old_b, a_star_old_b)
            y_func[t].fit(unempl_grid_old_b, y_star_old_b)
            theta_func[t].fit(unempl_grid_old_b, theta_star_old_b)
            E_func[t].fit(empl_grid_old_b, E_old_b)
            a_emp_func[t].fit(empl_grid_old_b, a_star_emp_old_b)
            Eprime_func[t].fit(empl_grid_old_b, E_prime_old_b)
            y_emp_func[t].fit(empl_grid_old_b, y_star_emp_old_b)
            theta_emp_func[t].fit(empl_grid_old_b, theta_star_emp_old_b)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # if you get useless warnings print "ignore"
            U_func[t].fit(unempl_grid, U_old)
            a_func[t].fit(unempl_grid, a_star_old)
            y_func[t].fit(unempl_grid, y_star_old)
            theta_func[t].fit(unempl_grid, theta_star_old)
            E_func[t].fit(empl_grid, E_old)
            a_emp_func[t].fit(empl_grid, a_star_emp_old)
            Eprime_func[t].fit(empl_grid, E_prime_old)
            y_emp_func[t].fit(empl_grid, y_star_emp_old)
            theta_emp_func[t].fit(empl_grid, theta_star_emp_old)
        print('rsize does not change')
        rsize_old = unempl_grid.shape[0]
        print('row size is', rsize_old)


    if (t>1):
        for i, aval in enumerate(a_grid):
            a_u_try[i] = max(np.asscalar(a_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False)),
                             0)
            a_e_try[i] = np.asscalar(
                a_emp_func[t].predict(
                    np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                    return_std=False))
            U_u_try[i] = np.asscalar(U_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
            E_e_try[i] = E_func[t].predict(
                np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                return_std=False)
            y_try[i] = np.asscalar(y_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
            theta_try[i] = np.asscalar(theta_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
            theta_emp_try[i] = max(np.asscalar(theta_emp_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2,
                                                                                    (low_w + high_w) / 2,
                                                                                    (low_y + high_y) / 2]),
                                                                     return_std=False)), 0)


        if (max(a_e_try)<1.0):
            a_emp_func[t] = a_emp_func[t-1]


        '''
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
        plt.scatter(unempl_grid_old[:, 0], a_star_old,s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(a_grid, U_u_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
        plt.scatter(unempl_grid_old[:,0], U_old)
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(a_grid, y_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
        plt.scatter(unempl_grid_old[:,0], y_star_old)
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(a_grid, theta_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
        plt.scatter(unempl_grid_old[:,0], theta_star_old)
        plt.show()
        '''

    for i,jj in enumerate(unempl_grid):
        print("VALUE", i, "IS", jj)
        a_prime = np.asscalar(a_func[t].predict(np.atleast_2d([jj[0], jj[1]]), return_std=False))
        a = jj[0]
        x = jj[1]
        state = [a, x, t]
        xx =  np.array([a/2, 2.0, 0.8])
        try:
            solsol = root(foccs, xx, state, method='hybr')
        except:
            pass
        print("unconstrained root success?", solsol.success)
        try:
            solconstr = root(foccs_constrained, [max(0, solsol.x[0]), solsol.x[1], solsol.x[2], 0], state,method='hybr')
        except:
            pass
        print("constrained root success?", solconstr.success)
        norm_distance = np.linalg.norm(solconstr.x[0:1] - solsol.x[0:1])
        if (norm_distance > 1e-5):
                if (solconstr.success == False) and (solsol.success == False):
                    status_collect[i, t - 1] = 0
                    #print("IPOPT SOLUTION")
                    #ipopt_sol = unemp_solver(xx, state)
                    #a_star_new[i, t - 1] = ipopt_sol[0]
                    #theta_star_new[i, t - 1] = ipopt_sol[1]
                    #y_star_new[i, t - 1] = ipopt_sol[2]
                    #if (ipopt_sol[3] < 0) or (ipopt_sol[1]==0):
                    #    status_collect[i, t - 1] = 0
                    #else:
                    #    print("ottimo")
                    #    print("the right solution is")
                    #    print(ipopt_sol[0:2])
                    try:
                        solconstr = root(foccs_constrained, [xx[0], xx[1], xx[2], 0], state,
                                     method='anderson')
                    except:
                        pass
                    if (solconstr.success == False):
                        status_collect[i, t - 1] = 0
                    else:
                        print("the right solution is")
                        print(solconstr.x)
                else:
                    print("the right solution is solconstr")
                    print(solconstr.x)
                    if (solsol.success==False) or (solconstr.x[0]>solsol.x[0]):
                        a_star_new[i, t - 1] = solconstr.x[0]
                        theta_star_new[i, t - 1] = solconstr.x[1]
                        y_star_new[i, t - 1] = solconstr.x[2]
                    else:
                        a_star_new[i, t - 1] = solsol.x[0]
                        theta_star_new[i, t - 1] = solsol.x[1]
                        y_star_new[i, t - 1] = solsol.x[2]


        else:
            print("the right solution is solsol")
            print(solsol.x)
            theta_star_new[i, t - 1] = solsol.x[1]
            y_star_new[i, t - 1] = solsol.x[2]
            a_star_new[i, t - 1] = solsol.x[0]




        if (status_collect[i, t - 1] == 0):
            U_new[i, t - 1] =  -eval_Uf([max(0,a),1.7,0.9], state)
        else:
            U_new[i,t-1] = -eval_Uf([max(0,a_star_new[i, t - 1]),theta_star_new[i, t - 1],y_star_new[i, t - 1]], state)


    controls_unemp = np.where(status_collect[:, t - 1] < 1.0)
    '''
    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid[:, 0], a_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)

    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid[:, 0], theta_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)

    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid[:, 0], y_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)

    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid[:, 0], U_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
    '''

    for i, g in enumerate(empl_grid):
        print("value", i, "is", g)
        if (t==1):
            xx = np.array([g[0], 1.0, 0.9])
        else:
            xx = np.array([g[0], theta_star_emp_new[i, t - 2], y_star_emp_new[i, t - 2]])
        e_state = [g[0], g[1], g[2], g[3], t]
        #solsol = root(foc_empl, xx, e_state, method='hybr')
        #print("the FOC solution is")
        #print(solsol.x)
        '''
        ipopt_emp_sol = emp_solver(xx, e_state)
        print("the IPOPT solution is", ipopt_emp_sol)
        a_star_emp_new[i, t - 1] = ipopt_emp_sol[0]
        theta_star_emp_new[i, t - 1] = ipopt_emp_sol[1]
        y_star_emp_new[i, t - 1] = ipopt_emp_sol[2]
        wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1],  g[1], t)
        E_new[i, t - 1] = -ipopt_emp_sol[3]
        '''
        try:
            solsol = root(foc_empl, xx, e_state, method='hybr') # or hybr?
        except:
            pass
        print("the IPOPT solution is", solsol.x)
        a_star_emp_new[i, t - 1] = solsol.x[0]
        theta_star_emp_new[i, t - 1] = solsol.x[1]
        y_star_emp_new[i, t - 1] = solsol.x[2]
        wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1],  g[1], t)
        E_new[i, t - 1] = -eval_Ef([max(0,a_star_emp_new[i, t - 1]),theta_star_emp_new[i, t - 1],
                                    y_star_emp_new[i, t - 1]], e_state)
        if (t==1):
            E_prime_new[i, t - 1] = E_funprime(0.0, g[0], a_star_emp_new[i,t-1], g[2])
        else:
            E_prime_new[i, t - 1] = E_funprime(E_prime_new[i, t - 2], g[0], a_star_emp_new[i, t - 1], g[2])
        if (solsol.success == False) or (solsol.x[1]>5.0) or (solsol.x[0]>8.0) :
            status_collect_emp[i, t - 1] = 0


    controls_emp = np.where( status_collect_emp[:,t-1] < 1.0 )
    joined =  controls_unemp + controls_emp
    joint = [x for xs in joined for x in xs]
    print('joint',joint)


    '''
    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, a_e_try, 'r', lw=3, zorder=9, label='employed (interpolation)')
    plt.scatter(empl_grid[:, 0], a_star_emp_new[:, t - 1], s=50, zorder=10, edgecolors=(0, 0, 0))

    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, theta_emp_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(empl_grid[:, 2], theta_star_emp_new[:, t - 1])

    fig, ax = plt.subplots(figsize=(9, 5))
    #ax.plot(a_grid, theta_emp_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(empl_grid[:, 2], E_new[:, t - 1])
    plt.show()
    '''
################## STORING BEFORE INTERPOLATING ##################################


    E_old = E_new[:, t - 1]
    E_prime_old = E_prime_new[:, t - 1]
    U_old = U_new[:, t - 1]
    a_star_old = a_star_new[:, t - 1]
    theta_star_old = theta_star_new[:, t - 1]
    y_star_old = y_star_new[:, t - 1]
    wage_star_old = wage_star_new[:, t - 1]
    a_star_emp_old = a_star_emp_new[:, t - 1]
    theta_star_emp_old = theta_star_emp_new[:, t - 1]
    y_star_emp_old = y_star_emp_new[:, t - 1]
    wage_star_emp_old = wage_star_emp_new[:, t - 1]

    unempl_grid_old = unempl_grid
    empl_grid_old = empl_grid

    status_collect = np.ones((rsize, life))


############# SIMULATION ##################

## initialization

wage_sim[0,:] = 0.0
y_sim[0,:] = 0.0
emp_status_sim[0,:] = 0

a_sim[0,:] = np.linspace(low_a+1.5, 1.5*(low_a+1.5), num=n_workers)
hum_k_sim[0,:] = np.random.uniform(low=1.2*low_x, high=0.8*high_x, size=(1, n_workers))

emp_matrix = np.random.uniform(low=0, high=1, size=(n_workers, life))
sep_matrix = np.random.uniform(low=0, high=1, size=(n_workers*2, life))




for ii in range(n_workers):
    for t in range(1,life):
        outcome_emp = emp_matrix[ii,t]
        outcome_sep = sep_matrix[ii,t]

        print(t)
        print('random', outcome_emp)
        print('random sep', outcome_sep)

        if emp_status_sim[t-1,ii]==1:
            print('employed problem')
            a_sim[t, ii] = R*max(a_emp_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii],wage_sim[t-1,ii], y_sim[t-1,ii]]),return_std=False),0)
            theta_sim[t, ii] = max(np.asscalar(theta_emp_func[life - t].predict(np.atleast_2d([a_sim[t - 1, ii],hum_k_sim[t - 1, ii],
                                                        wage_sim[t - 1, ii],y_sim[t - 1, ii]]),return_std=False)),0)

            jobchange = m_fun(theta_sim[t, ii] )*pi

            print('prob of job change', jobchange)

            if outcome_sep < lamb:
                emp_status_sim[t, ii] = 0
                y_sim[t, ii] = y_sim[t - 1, ii]
                wage_sim[t, ii] = wage_sim[t - 1, ii]
                job_to_job_sim[t,ii] = 0
            else:
                emp_status_sim[t, ii] = 1
                if jobchange > outcome_emp:
                    y_sim[t, ii] = np.asscalar(y_emp_func[life-t].predict(
                        np.atleast_2d([a_sim[t - 1, ii], hum_k_sim[t - 1, ii], wage_sim[t - 1, ii], y_sim[t - 1, ii]]),
                        return_std=False))
                    wage_sim[t, ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t - 1, ii], life - t)
                    job_to_job_sim[t,ii] = 1
                else:
                    y_sim[t, ii] = y_sim[t - 1, ii]
                    wage_sim[t, ii] = wage_sim[t - 1, ii]
                    job_to_job_sim[t, ii] = 0
            hum_k_sim[t, ii] = g_fun(y_sim[t, ii], hum_k_sim[t - 1, ii])
        else:
            print('unemployed problem')
            a_sim[t,ii] = R*max(np.asscalar(a_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii],hum_k_sim[t-1,ii]]), return_std=False)),0)
            y_sim[t, ii] = np.asscalar(y_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii]]), return_std=False))
            theta_sim[t, ii] = np.asscalar(theta_func[life-t].predict(np.atleast_2d([a_sim[t - 1, ii],
                                                                                    hum_k_sim[t-1,ii]]), return_std=False))
            wage_sim[t,ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t-1,ii], life-t)
            if (wage_sim[t,ii]<0):
                wage_sim[t, ii] = wage_sim[t - 1, ii]
            hum_k_sim[t, ii] =  hum_k_sim[t - 1, ii]
            job_to_job_sim[t, ii] = 0
            if outcome_emp < m_fun(theta_sim[t, ii]):
                emp_status_sim[t, ii] = 1
            else:
                emp_status_sim[t, ii] = 0

        print('probability for unemployed agent',ii,m_fun(theta_sim[t, ii]))
        print('human capital for agent', ii, hum_k_sim[t, ii])
        print('emp status for agent',ii, emp_status_sim[t,ii])
        print('matched firm for agent',ii, y_sim[t, ii])
        print('a_sim for agent',ii, a_sim[t,ii])
        print('wage for agent',ii,wage_sim[t,ii])





fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,life+1), a_sim[:,190], 'r', lw=3, zorder=9, label='agent 0')
ax.plot(range(1,life+1), a_sim[:,291], 'b', lw=3, zorder=9, label='agent 1')
ax.plot(range(1,life+1), a_sim[:,392], 'g', lw=3, zorder=9, label='agent 2')
ax.plot(range(1,life+1), a_sim[:,493], 'y', lw=3, zorder=9, label='agent 3')
ax.plot(range(1,life+1), a_sim[:,594], 'c', lw=3, zorder=9, label='agent 4')
ax.plot(range(1,life+1), a_sim[:,695], 'm', lw=3, zorder=9, label='agent 5')
ax.plot(range(1,life+1), a_sim[:,796], '--r', lw=3, zorder=9, label='agent 6')
ax.legend()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,life+1), hum_k_sim[:,190], 'r', lw=3, zorder=9, label='agent 0')
ax.plot(range(1,life+1), hum_k_sim[:,291], 'b', lw=3, zorder=9, label='agent 1')
ax.plot(range(1,life+1), hum_k_sim[:,392], 'g', lw=3, zorder=9, label='agent 2')
ax.plot(range(1,life+1), hum_k_sim[:,493], 'y', lw=3, zorder=9, label='agent 3')
ax.plot(range(1,life+1), hum_k_sim[:,594], 'c', lw=3, zorder=9, label='agent 4')
ax.plot(range(1,life+1), hum_k_sim[:,695], 'm', lw=3, zorder=9, label='agent 5')
ax.plot(range(1,life+1), hum_k_sim[:,796], '--r', lw=3, zorder=9, label='agent 6')
ax.legend()


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,life+1), emp_status_sim[:,190], 'r', lw=3, zorder=9, label='agent 0')
ax.plot(range(1,life+1), emp_status_sim[:,291], 'b', lw=3, zorder=9, label='agent 1')
ax.plot(range(1,life+1), emp_status_sim[:,392], 'g', lw=3, zorder=9, label='agent 2')
ax.plot(range(1,life+1), emp_status_sim[:,493], 'y', lw=3, zorder=9, label='agent 3')
ax.plot(range(1,life+1), emp_status_sim[:,594], 'c', lw=3, zorder=9, label='agent 4')
ax.plot(range(1,life+1), emp_status_sim[:,695], 'm', lw=3, zorder=9, label='agent 5')
ax.plot(range(1,life+1), emp_status_sim[:,796], '--r', lw=3, zorder=9, label='agent 6')
ax.legend()


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,life+1), wage_sim[:,190], 'r', lw=3, zorder=9, label='agent 0')
ax.plot(range(1,life+1), wage_sim[:,291], 'b', lw=3, zorder=9, label='agent 1')
ax.plot(range(1,life+1), wage_sim[:,392], 'g', lw=3, zorder=9, label='agent 2')
ax.plot(range(1,life+1), wage_sim[:,493], 'y', lw=3, zorder=9, label='agent 3')
ax.plot(range(1,life+1), wage_sim[:,594], 'c', lw=3, zorder=9, label='agent 4')
ax.plot(range(1,life+1), wage_sim[:,695], 'm', lw=3, zorder=9, label='agent 5')
ax.plot(range(1,life+1), wage_sim[:,796], '--r', lw=3, zorder=9, label='agent 6')
ax.legend()

plt.show()


#### statistics ####

mean_wage = np.empty((life,1))
mean_assets = np.empty_like(mean_wage)
unemployment_rate = np.empty_like(mean_wage)
transition_rate = np.empty_like(mean_wage)
for t in range(1,life):
    mean_wage[t] = np.mean(wage_sim[t,:])
    mean_assets[t] = np.mean(a_sim[t,:])
    unemployment_rate[t] = 1 - np.mean(emp_status_sim[t,:])
    transition_rate[t] = np.mean(job_to_job_sim[t,:])/(1-unemployment_rate[t])


for i in range(1,life):
    print("period", i, "and wage", mean_wage[i], "a", mean_assets[i], "U", unemployment_rate[i], "EE", transition_rate[i] )


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),mean_wage[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),mean_assets[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),unemployment_rate[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),transition_rate[1:life], 'r', lw=3, zorder=9, label='agent 0')



plt.show()




for t in range(1,life):
    a_tent = np.asscalar(a_emp_func[life-t].predict(np.atleast_2d([a_grid[7], x_grid[7], 7.7, 1.1]),return_std=False))
    theta_tent = np.asscalar(theta_emp_func[life - t].predict(np.atleast_2d([a_grid[7], x_grid[7], 7.7, 1.1]),return_std=False))
    y_tent = np.asscalar(y_emp_func[life - t].predict(np.atleast_2d([a_grid[7], x_grid[7], 7.7, 1.1]),return_std=False))
    wage_tent = wage(y_tent, theta_tent, x_grid[7], life-t)

    print("age", t, "a_policy is", a_tent, "theta_policy is", theta_tent, "y_policy is", y_tent,"implied wage is", wage_tent)

    a_tent = np.asscalar(a_emp_func[life-t].predict(np.atleast_2d([a_grid[4], x_grid[7], 7.7, 1.1]),return_std=False))
    theta_tent = np.asscalar(theta_emp_func[life - t].predict(np.atleast_2d([a_grid[4], x_grid[7], 7.7, 1.1]),return_std=False))
    y_tent = np.asscalar(y_emp_func[life - t].predict(np.atleast_2d([a_grid[4], x_grid[7], 7.7, 1.1]),return_std=False))
    wage_tent = wage(y_tent, theta_tent, x_grid[7], life - t)
    print("now lower a")
    print("age", t, "a_policy is", a_tent, "theta_policy is", theta_tent, "y_policy is", y_tent,"implied wage is", wage_tent)

    wage_tent = wage(y_tent, theta_tent, x_grid[7], life - t)
    wage_tent_2 = wage(0.75*y_tent, theta_tent, x_grid[7], life - t)

    print("compare wages with high and low y", wage_tent,wage_tent_2)



