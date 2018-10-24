############################################################################################
############################# MAIN CODE FOR STUDENT LOANS PROJECT ##########################
#############################    MODEL WITH ONTHEJOB-S + mrisk    ##########################
############################################################################################

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
    return  y**(alpha-1)*A*(y**alpha  + g_fun(y, x)**alpha )**(1/alpha-1)  +\
            g_fun_prime(y,x)*g_fun(y, x)**(alpha-1)*A*(y**alpha  + g_fun(y, x)**alpha )**(1/alpha-1)  #A * (alpha * y ** (alpha - 1)*g_fun(y, x)**(1-alpha) +
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
    return f(A, y, x) - k_fun(y)/(cosa1*q_fun(theta)) -  (max(adj*y - g_fun(y,x),0))*1


def E_fun(U, E, EU, a, a_prime, w,theta, E_tilde, t_f): #
    return u(c_e(w, a, a_prime)) + beta*(1-pi*m_fun(theta))*((1-lamb)*E + lamb*U) + \
           beta*pi*m_fun(theta)*(t_f*EU + (1-t_f)*E_tilde )


def E_funprime(E_prime, a, a_prime, theta, w):
    return u_prime(c_e(w, a, a_prime)) + beta*(1-lamb)*(1-pi*m_fun(theta))*E_prime   #E_prime function of a_prime

def U_fun(U, E, EU, a, a_prime, theta,t_f):
    return u(c_u(a, a_prime)) + beta*(m_fun(theta)*(t_f*EU + (1-t_f)*E) + (1-m_fun(theta))*U)

def EU_fun(U, E, EU, a, a_prime, theta,t_f,w):
    return u(c_e(w,a, a_prime)) + beta*(m_fun(theta)*(t_f*EU + (1-t_f)*E) + (1-m_fun(theta))*U)

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

def t_fun(p, x):
    y = p*adj
    if (y<x):
        return 0
    else:
        return ((y-x)/y)   #*l_fun(y-x)

def t_fun_prime(p, x):
    y = p*adj
    return (x/y**2)*l_fun(y-x)


############################ DECLARATIONS : UNEMPLOYED FUNCTIONS ############################

def eval_Uf(xx, user_data= state):
    assert len(xx) == nvar
    a = state[0]
    x = state[1]
    t = state[2]

    a_prime = xx[0]
    theta = xx[1]
    y = xx[2]

    w_wage = wage(y, theta, g_fun(y, x), t)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    theta_etilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    t_f = t_fun(y,x)
    t_f_1 = t_fun(y_int,x)
    t_fe_1 = t_fun(y_emp_int,x)
    t_feu_1 = t_fun(y_eu_int,x)


    return -u(c_u(a, a_prime)) - beta * (
    m_fun(theta) *( t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) +
                            (1-t_f)*E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1) )+
    (1 - m_fun(theta)) * U_fun(UU, EE, EU, a_prime, a_int, theta_tilde, t_f_1))

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

def foccs_risk(thetas, params):
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
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    theta_etilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    ey_state = [a_prime, g_fun(x,y), w_wage, y, t]
    ey_xx = [a_emp_int,theta_tilde,y_emp_int]
    if t==1:
        Ey = 0.0
    else:
        Ey = eval_grad_E_y(ey_xx,ey_state )
    ##
    cu = c_u(a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)
    ceut = c_e(w_wage, a_prime, a_eump_int)
    t_f = t_fun(y,x)
    t_fprime = t_fun_prime(y,x)
    t_f_1 = t_fun(y_int,x)
    t_fe_1 = t_fun(y_emp_int,x)
    t_feu_1 = t_fun(y_eu_int,x)
    ##

    foc1 = -u_prime(cu) + beta * R * (m_fun(theta)* (t_fun(y, x)*u_prime(ceut) +
                                                     (1-t_fun(y, x))*u_prime(cet)) + (1 - m_fun(theta)) * u_prime(cut))
    foc2 = m_funprime(theta) * ( t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) +
                                 (1-t_f)*E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)
        - U_fun(UU, EE, EU, a_prime, a_int, theta_tilde, t_f_1) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, theta_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = t_fprime*(EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
                     E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)) \
           + (1-t_fun(y,x))*Ey \
           + (t_f*u_prime(ceut) +(1-t_f)*E_funprime(E_prime, a_prime, a_emp_int, theta_etilde, w_wage))*(f_prime(A, y, x)
                                                                    - k_funprime(y) / (q_fun(theta)*cosa1) - (1-phi)*l_fun(adj*y-x))


    return [foc1.item(), foc2.item(), foc3]


def foccs_risk_constrained(thetas, params):
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
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    theta_etilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    ey_state = [a_prime, g_fun(x,y), w_wage, y, t]
    ey_xx = [a_emp_int,theta_tilde,y_emp_int]
    if t==1:
        Ey = 0.0
    else:
        Ey = eval_grad_E_y(ey_xx,ey_state )
    ##
    cu = c_u(a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)
    ceut = c_e(w_wage, a_prime, a_eump_int)
    t_f = t_fun(y,x)
    t_fprime = t_fun_prime(y,x)
    t_f_1 = t_fun(y_int,x)
    t_fe_1 = t_fun(y_emp_int,x)
    t_feu_1 = t_fun(y_eu_int,x)
    ##

    foc1 = -u_prime(cu) + beta * R * (m_fun(theta)* (t_fun(y, x)*u_prime(ceut) +
                                            (1-t_fun(y, x))*u_prime(cet)) + (1 - m_fun(theta)) * u_prime(cut)) \
           + np.amax([0, +vincolo])
    foc2 = m_funprime(theta) * ( t_fun(y, x)*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) +
                                 (1-t_fun(y, x))*E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)
        - U_fun(UU, EE, EU, a_prime, a_int, theta_tilde, t_f_1) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, theta_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = t_fprime*(EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
                     E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)) \
           +  (1-t_fun(y,x))*Ey + (t_f*u_prime(ceut) +
            (1-t_f)*E_funprime(E_prime, a_prime, a_emp_int, theta_etilde, w_wage))*(f_prime(A, y, x)  -(1-phi)*l_fun(adj*y-x)
                                                                                    - k_funprime(y) / (q_fun(theta)*cosa1))
    foc4 = np.amax([0, -vincolo]) - a_prime

    return [foc1.item(), foc2.item(), foc3, foc4]


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
    t = int(e_state[4])

    a_prime = xx[0]
    theta = xx[1]
    y_tilde = xx[2]


    w_wage = wage(y_tilde, theta, g_fun(y_tilde, x), t)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    a_emp_tilde =  np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,
                                                                g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)), 0)
    theta_tilde = np.asscalar(
        theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    theta_int = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(
        y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))

    t_f = t_fun(y_tilde, x)
    t_f_1 = t_fun(y_int, x)
    t_fe_1 = t_fun(y_emp_int, x)
    t_feu_1 = t_fun(y_eu_int, x)


    return  -u(c_e(w, a, a_prime)) - \
           beta * ( (1-pi*m_fun(theta))*((1-lamb)*E_fun(UU, EE, EU, a_prime, a_emp_int, w, theta_int, EE_tilde, t_fe_1) +
                                             lamb * U_fun(UU, EE, EU, a_prime, a_int, theta_int,t_f_1 )) +
                    pi*m_fun(theta)*(t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage)  +
                    (1-t_f)*E_fun(UU, EE, EU, a_prime, a_emp_tilde, w_wage, theta_tilde, EE_tilde, t_fe_1)))



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

def eval_grad_E_y(x, user_data = state):
    assert len(x) == nvar
    # finite differences
    A = np.empty((nvar * ncon))
    xAdj = x
    Fx1 = eval_Ef(xAdj,state)
    h = 1e-4
    for ii in range(nvar):
        xAdj = x
        xAdj[ii] = xAdj[ii] + h
        Fx2 = eval_Ef(xAdj,state)
        A[ii] = (Fx2 - Fx1) / h
        xAdj[ii] = xAdj[ii] - h
    return A[2] #grad_f



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
     theta*pi*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, theta_int, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y_tilde, x)*q_fun(theta)*cosa1 - k_funprime(y_tilde)

    return [foc1.item(), foc2.item(), foc3]


def foc_empl_constrained(thetas,other):
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
    vincolo = thetas[3]
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
     theta*pi*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta)) + np.amax([0,+vincolo])
    foc3 = f_prime(A, y_tilde, x)*q_fun(theta)*cosa1 - k_funprime(y_tilde)
    foc4 = np.amax([0,-vincolo]) - theta

    return [foc1.item(), foc2.item(), foc3, foc4]



def foc_empl_risk_constrained(thetass, other):
        a = other[0]  # np.asscalar()
        x = other[1]  # np.asscalar()
        w = other[2]  # np.asscalar()
        y = other[3]  # np.asscalar()
        t = other[4]
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
        a_prime = thetass[0]
        theta = thetass[1]
        y_tilde = thetass[2]
        vincolo = thetass[3]
        w_wage = wage(y_tilde, theta, g_fun(y_tilde, x), t)

        a_prime_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
        a_prime_tilde = np.asscalar(
            a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
        a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)), 0)
        a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
        theta_tilde = np.asscalar(
            theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
        theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
        theta_int = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
        y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
        y_emp_int = np.asscalar(
            y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
        y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))

        t_f = t_fun(y_tilde, x)
        t_fprime = t_fun_prime(y_tilde, x)
        t_f_1 = t_fun(y_int, x)
        t_fe_1 = t_fun(y_emp_int, x)
        t_feu_1 = t_fun(y_eu_int, x)

        UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_int, x]), return_std=False))
        EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_int, g_fun(y, x), w, y]), return_std=False))
        EE_tilde = np.asscalar(
            E_func[t].predict(np.atleast_2d([a_prime_tilde, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
        E_prime_tilde = np.asscalar(
            Eprime_func[t].predict(np.atleast_2d([a_prime_tilde, g_fun(y_tilde, x), w_wage, y_tilde]),
                                   return_std=False))
        EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
        ey_state = [a_prime, g_fun(x,y_tilde), w_wage, y_tilde, t]
        ey_xx = [a_prime_tilde, theta_tilde, y_emp_int]
        if t == 1:
            Ey = 0.0
        else:
            Ey = eval_grad_E_y(ey_xx, ey_state)

        ce = c_e(w, a, a_prime)
        cut = c_u(a_prime, a_int)
        cet = c_e(w, a_prime, a_prime_int)
        cetilde = c_e(w_wage, a_prime, a_prime_tilde)
        ceutilde = c_e(w_wage, a_prime, a_eump_int)

        foc1 = -u_prime(ce) + (1 - pi * m_fun(theta)) * beta * R * ((1 - lamb) * u_prime(cet) +
                                                                (lamb) * u_prime(cut)) + beta * R * pi * m_fun(theta) *\
                                                                        ((1-t_f)*u_prime(cetilde)+t_f*u_prime(ceutilde))
        foc2 =  m_funprime(theta) * ((1-t_f)*E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde,t_fe_1)
                                     +  t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
                lamb * U_fun(UU, EE, EU, a_prime, a_int, theta_int, t_f_1) -
                                     (1 - lamb) * E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde, t_fe_1)) + \
               theta  * ((1-t_f)*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, theta_tilde, w_wage) +
                         t_f*u_prime(ceutilde))* q_funprime(theta) * k_fun(y) / (cosa1 * q_fun(theta))  + np.amax([0,+vincolo])
        foc3 = t_fprime*( EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
                E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde,t_fe_1) ) + (1-t_f)*Ey + \
          (t_f*u_prime(ceutilde) + (1-t_f)*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, theta_tilde, w_wage))*\
          (f_prime(A, y_tilde, x)  -(1-phi)*l_fun(adj*y_tilde-x) - k_funprime(y_tilde)/(q_fun(theta) * cosa1))
        foc4 = np.amax([0, -vincolo]) - theta

        return [foc1.item(), foc2.item(), foc3, foc4]


def foc_empl_risk(thetass, other):
    a = other[0]  # np.asscalar()
    x = other[1]  # np.asscalar()
    w = other[2]  # np.asscalar()
    y = other[3]  # np.asscalar()
    t = other[4]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetass[0]
    theta = thetass[1]
    y_tilde = thetass[2]
    w_wage = wage(y_tilde, theta, g_fun(y_tilde, x), t)

    a_prime_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
    a_prime_tilde = np.asscalar(
        a_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)), 0)
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    theta_tilde = np.asscalar(
        theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    theta_int = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(
        y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))

    t_f = t_fun(y_tilde, x)
    t_fprime = t_fun_prime(y_tilde, x)
    t_f_1 = t_fun(y_int, x)
    t_fe_1 = t_fun(y_emp_int, x)
    t_feu_1 = t_fun(y_eu_int, x)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_int, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_int, g_fun(y, x), w, y]), return_std=False))
    EE_tilde = np.asscalar(
        E_func[t].predict(np.atleast_2d([a_prime_tilde, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    E_prime_tilde = np.asscalar(
        Eprime_func[t].predict(np.atleast_2d([a_prime_tilde, g_fun(y_tilde, x), w_wage, y_tilde]),
                               return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    ey_state = [a_prime, g_fun(x, y_tilde), w_wage, y_tilde, t]
    ey_xx = [a_prime_tilde, theta_tilde, y_emp_int]
    if t == 1:
        Ey = 0.0
    else:
        Ey = eval_grad_E_y(ey_xx, ey_state)

    ce = c_e(w, a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w, a_prime, a_prime_int)
    cetilde = c_e(w_wage, a_prime, a_prime_tilde)
    ceutilde = c_e(w_wage, a_prime, a_eump_int)

    foc1 = -u_prime(ce) + (1 - pi * m_fun(theta)) * beta * R * ((1 - lamb) * u_prime(cet) +
                                                                (lamb) * u_prime(cut)) + beta * R * pi * m_fun(theta) * \
                                                                                         ((1 - t_f) * u_prime(
                                                                                             cetilde) + t_f * u_prime(
                                                                                             ceutilde))
    foc2 = m_funprime(theta) * (
    (1 - t_f) * E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde, t_fe_1)
    + t_f * EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
    lamb * U_fun(UU, EE, EU, a_prime, a_int, theta_int, t_f_1) -
    (1 - lamb) * E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde, t_fe_1)) + \
           theta * ((1 - t_f) * E_funprime(E_prime_tilde, a_prime, a_prime_tilde, theta_tilde,w_wage) +
                    t_f * u_prime(ceutilde)) * q_funprime(theta) * k_fun(y) / (cosa1 * q_fun(theta))
    foc3 = t_fprime * (EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
            E_fun(UU, EE_tilde, EU, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde, t_fe_1)) + (1 - t_f) * Ey + \
           (t_f * u_prime(ceutilde) + (1 - t_f) * E_funprime(E_prime_tilde, a_prime, a_prime_tilde, theta_tilde,w_wage)) * \
           (f_prime(A, y_tilde, x) -(1-phi)*l_fun(adj*y_tilde-x) - k_funprime(y_tilde) / (q_fun(theta) * cosa1))

    return [foc1.item(), foc2.item(), foc3]


############################################################################################

def eval_EUf(xx, user_data= state):
    assert len(xx) == nvar
    a = state[0]
    x = state[1]
    w = state[2]
    t = state[3]

    a_prime = xx[0]
    theta = xx[1]
    y = xx[2]

    w_wage = wage(y, theta, g_fun(y, x), t)

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    theta_etilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    t_f = t_fun(y,x)
    t_f_1 = t_fun(y_int,x)
    t_fe_1 = t_fun(y_emp_int,x)
    t_feu_1 = t_fun(y_eu_int,x)


    return -u(c_e(w, a, a_prime)) - beta * (
    m_fun(theta) *( t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) +
                            (1-t_f)*E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1) )+
    (1 - m_fun(theta)) * U_fun(UU, EE, EU, a_prime, a_int, theta_tilde, t_f_1))


def foccs_eu_risk(thetas, params):
    a = params[0]
    x = params[1]
    w = params[2]
    t = params[3]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y = thetas[2]
    w_wage = wage(y, theta, g_fun(y,x), t)
    a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    a_eump_int = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
    theta_tilde = np.asscalar(theta_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    theta_etilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    theta_eu_tilde = np.asscalar(theta_eump_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    y_int = np.asscalar(y_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
    y_emp_int = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y,x), w_wage, y]), return_std=False))
    y_eu_int = np.asscalar(y_eu_func[t].predict(np.atleast_2d([a_prime, x, w_wage]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    EU = np.asscalar(EU_func[t].predict(np.atleast_2d([a_prime,x, w_wage ]), return_std=False))
    E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    ey_state = [a_prime, g_fun(x,y), w_wage, y, t]
    ey_xx = [a_emp_int,theta_tilde,y_emp_int]
    if t==1:
        Ey = 0.0
    else:
        Ey = eval_grad_E_y(ey_xx,ey_state )
    ##
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)
    ceut = c_e(w_wage, a_prime, a_eump_int)
    c_eu = c_e(w,a,a_prime)
    t_f = t_fun(y,x)
    t_fprime = t_fun_prime(y,x)
    t_f_1 = t_fun(y_int,x)
    t_fe_1 = t_fun(y_emp_int,x)
    t_feu_1 = t_fun(y_eu_int,x)
    ##

    foc1 = -u_prime(c_eu) + beta * R * (m_fun(theta)* (t_fun(y, x)*u_prime(ceut) +
                                                     (1-t_fun(y, x))*u_prime(cet)) + (1 - m_fun(theta)) * u_prime(cut))
    foc2 = m_funprime(theta) * ( t_f*EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) +
                                 (1-t_f)*E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)
        - U_fun(UU, EE, EU, a_prime, a_int, theta_tilde, t_f_1) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, theta_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = t_fprime*(EU_fun(UU, EE, EU, a_prime, a_eump_int, theta_eu_tilde, t_feu_1, w_wage) -
                     E_fun(UU, EE, EU, a_prime, a_emp_int, w_wage, theta_etilde, EE_tilde, t_fe_1)) \
           + (1-t_fun(y,x))*Ey \
           + (t_f*u_prime(ceut) +(1-t_f)*E_funprime(E_prime, a_prime, a_emp_int, theta_etilde, w_wage))*\
             (f_prime(A, y, x)-(1-phi)*l_fun(adj*y-x)- k_funprime(y) / (q_fun(theta)*cosa1))


    return [foc1.item(), foc2.item(), foc3]





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
EU_func = {}
U_func =  {}
Eprime_func =  {}
a_func =  {}
a_emp_func =  {}
a_eump_func =  {}
y_func = {}
y_emp_func = {}
y_eu_func = {}
theta_func = {}
theta_emp_func = {}
theta_eump_func = {}

theta_wage_func = {}


for i in range(1,life+1):
    E_func[i] = 'E_func_'+str(i)
    EU_func[i] = 'EU_func_' + str(i)
    U_func[i] =  'U_func_'+str(i)
    Eprime_func[i] = 'Eprime_func_'+str(i)
    a_func[i] =  'a_func_'+str(i)
    a_emp_func[i] = 'a_emp_func_'+str(i)
    a_eump_func[i] = 'a_eump_func_'+str(i)
    y_eu_func[i] = 'y_eu_func_' + str(i)
    y_func[i] = 'y_func_' + str(i)
    y_emp_func[i] = 'y_emp_func_' + str(i)
    theta_emp_func[i] = 'theta_emp_func_' + str(i)
    theta_func[i] = 'theta_func_' + str(i)
    theta_eump_func[i] = 'theta_eump_func_' + str(i)

    theta_wage_func[i] = 'theta_wage_func' + str(i)

kernel = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e2))
kernel_bis = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e3)) + WhiteKernel(noise_level=1e-3,
                                                                      noise_level_bounds=(1e-10, 1e+1))
kernel_E = 1.0 * RBF(7.0, (1e-3, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e+1))


controls_unemp = np.where(status_collect[:, 0] < 1.0)
controls_emp = np.where( status_collect_emp[:,0] < 1.0 )
joined =  controls_unemp + controls_emp
joint = [x for xs in joined for x in xs]

wage_grid[:,0] = unempl_grid[:,1]
wage_grid[:,1] = y_star_old[:,0]

for t in range(1,life+1):

    print("THIS IS AGE", life-t)
    E_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=5)
    U_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    EU_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=5)
    Eprime_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=10)
    a_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    a_eump_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    a_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    y_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    y_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    y_eu_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    theta_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
    theta_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
    theta_eump_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)

    theta_wage_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)

    if any(joint) and (t>1):  # [0]
        print("status collector is not empty:")
        print('we have to change sizes')
        unempl_grid_old_b = np.delete(unempl_grid, controls_unemp, axis=0)
        empl_grid_old_b = np.delete(empl_grid, controls_emp, axis=0)
        eu_grid_old_b = np.delete(eu_grid, controls_eump, axis=0)

        wage_grid_b = np.delete(wage_grid, controls_unemp, axis=0)

        a_star_old_b = np.delete(a_star_old, controls_unemp, axis=0)
        a_star_emp_old_b = np.delete(a_star_emp_old, controls_emp, axis=0)
        a_star_eump_old_b = np.delete(a_star_eump_old, controls_eump, axis=0)

        y_star_old_b = np.delete(y_star_old, controls_unemp, axis=0)
        y_star_emp_old_b = np.delete(y_star_emp_old, controls_emp, axis=0)
        y_star_eump_old_b = np.delete(y_star_eump_old, controls_eump, axis=0)

        theta_star_old_b = np.delete(theta_star_old, controls_unemp, axis=0)
        theta_star_emp_old_b = np.delete(theta_star_emp_old, controls_emp, axis=0)
        theta_star_eump_old_b = np.delete(theta_star_eump_old, controls_eump, axis=0)

        wage_star_old_b = np.delete(wage_star_old, controls_unemp, axis=0)
        wage_star_emp_old_b = np.delete(wage_star_emp_old, controls_emp, axis=0)

        U_old_b = np.delete(U_old, controls_unemp, axis=0)
        E_old_b = np.delete(E_old, controls_emp, axis=0)
        E_prime_old_b = np.delete(E_prime_old, controls_emp, axis=0)
        EU_old_b = np.delete(EU_old, controls_eump, axis=0)

        rsize_old = eu_grid_old_b.shape[0]
        print('new eu row size is', rsize_old)
        rsize_old = unempl_grid_old_b.shape[0]
        print('new unemp row size is', rsize_old)
        rsize_old = empl_grid_old_b.shape[0]
        print('new emp row size is', rsize_old)
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # if you get useless warnings print "ignore"
            a_func[t].fit(unempl_grid_old_b, a_star_old_b)
            a_emp_func[t].fit(empl_grid_old_b, a_star_emp_old_b)
            a_eump_func[t].fit(eu_grid_old_b, a_star_eump_old_b)

            y_func[t].fit(unempl_grid_old_b, y_star_old_b)
            y_emp_func[t].fit(empl_grid_old_b, y_star_emp_old_b)
            y_eu_func[t].fit(eu_grid_old_b, y_star_eump_old_b)

            theta_func[t].fit(unempl_grid_old_b, theta_star_old_b)
            theta_emp_func[t].fit(empl_grid_old_b, theta_star_emp_old_b)
            theta_eump_func[t].fit(eu_grid_old_b, theta_star_eump_old_b)

            theta_wage_func[t].fit(wage_grid_b, theta_star_old_b)

            E_func[t].fit(empl_grid_old_b, E_old_b)
            U_func[t].fit(unempl_grid_old_b, U_old_b)
            EU_func[t].fit(eu_grid_old_b, EU_old_b)
            Eprime_func[t].fit(empl_grid_old_b, E_prime_old_b)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # if you get useless warnings print "ignore"
            a_func[t].fit(unempl_grid, a_star_old)
            a_emp_func[t].fit(empl_grid, a_star_emp_old)
            a_eump_func[t].fit(eu_grid, a_star_eump_old)

            theta_func[t].fit(unempl_grid, theta_star_old)
            theta_emp_func[t].fit(empl_grid, theta_star_emp_old)
            theta_eump_func[t].fit(eu_grid, theta_star_eump_old)

            theta_wage_func[t].fit(wage_grid, theta_star_old)

            y_func[t].fit(unempl_grid, y_star_old)
            y_emp_func[t].fit(empl_grid, y_star_emp_old)
            y_eu_func[t].fit(eu_grid, y_star_eump_old)

            U_func[t].fit(unempl_grid, U_old)
            E_func[t].fit(empl_grid, E_old)
            EU_func[t].fit(eu_grid, EU_old)
            Eprime_func[t].fit(empl_grid, E_prime_old)
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

        for j, xval in enumerate(x_grid):
            theta_wage_try[j] = max(np.asscalar(theta_wage_func[t].predict(np.atleast_2d([xval,(low_y + high_y) / 2]),
                                                                         return_std=False)), 0)



        if (max(a_e_try)<1.0):
            print("***** BIG FUCK UP *****")
            print("***** BIG FUCK UP *****")
            print("***** BIG FUCK UP *****")
            print("***** BIG FUCK UP *****")


        if plot_stuff==1:
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

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x_grid, theta_wage_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
            plt.scatter(unempl_grid_old[:,1], theta_star_old)
            plt.show()


    '''
    a_prime = np.asscalar(a_func[t].predict(np.atleast_2d([unempl_grid[0,0], unempl_grid[0,1]]), return_std=False))
    a = unempl_grid[0,0]
    x = unempl_grid[0,1]
    state = [a, x, t]
    xx =  np.array([a_prime, 1.0, 0.6])
    print(foccs_risk(xx,state)) # wage(0.8,0.0,x,t)
    solsol = root(foccs_risk, xx, state, method='hybr')
    print(solsol)
    '''

    #a_prime = np.asscalar(a_func[t].predict(np.atleast_2d([unempl_grid[0,0], unempl_grid[0,1]]), return_std=False))
    #a = eu_grid[0,0]
    #x = eu_grid[0,1]
    #w = eu_grid[0,2]
    #state = [a, x, w, t]
    #xx = np.array([a/2.5, 2.4, 0.7])
    #print(state)
    #print(foccs_eu_risk(xx,state))
    #solsol = root(foccs_eu_risk, xx, state, method='hybr')
    #print(solsol)

    for i, jj in enumerate(eu_grid):
        print("VALUE", i, "IS", jj)
        a = jj[0]
        x = jj[1]
        w = jj[2]
        a_prime = np.asscalar(a_eump_func[t].predict(np.atleast_2d([a,x,w]), return_std=False))
        state = [a, x, w, t]
        xx = np.array([a, 3.0, 0.7])
        try:
            sol_sol = root(foccs_eu_risk, xx, state, method='hybr')
        except:
            pass
        print( "unconstrained root success?", sol_sol.success )
        print(sol_sol.x)
        if (sol_sol.success == True and sol_sol.x[0] < 10.0 and sol_sol.x[1] < 6.0):
            print("the right solution is solsol")
            #print(sol_sol.x)
            a_star_eump_new[i, t - 1] = sol_sol.x[0]
            theta_star_eump_new[i, t - 1] = sol_sol.x[1]
            y_star_eump_new[i, t - 1] = sol_sol.x[2]
        else:
            if (t > 7) and (status_collect[i, t - 2] == 1):
                print("used past solution because of too many issues")
                a_star_eump_new[i, t - 1] = a_star_eump_new[i, t - 2]
                theta_star_eump_new[i, t - 1] = theta_star_eump_new[i, t - 2]
                y_star_eump_new[i, t - 1] = y_star_eump_new[i, t - 2]
                wage_star_new[i, t - 1] = wage(y_star_eump_new[i, t - 1], theta_star_eump_new[i, t - 1], g[1], t)
            else:
                try:
                    sol_sol = root(foccs_eu_risk, xx, state, method='anderson')
                except:
                    pass
                if (sol_sol.success == True and sol_sol.x[0] < 10.0 and sol_sol.x[1] < 6.0):
                    a_star_eump_new[i, t - 1] = sol_sol.x[0]
                    theta_star_eump_new[i, t - 1] = sol_sol.x[1]
                    y_star_eump_new[i, t - 1] = sol_sol.x[2]
                else:
                    print("no solution works")
                    status_collect_eump[i, t - 1] = 0

        if (status_collect_eump[i, t - 1] == 0):
            EU_new[i, t - 1] =  -eval_EUf([max(0,a),1.7,0.9], state)
        else:
            EU_new[i,t-1] = -eval_EUf([max(0,a_star_eump_new[i, t - 1]),
                                      theta_star_eump_new[i, t - 1],y_star_eump_new[i, t - 1]], state)

        print(EU_new[i,t-1])

    controls_eump = np.where(status_collect_eump[:, t - 1] < 1.0)

    if plot_stuff==1:
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(eu_grid[:, 0]+eu_grid[:, 2],
                    a_star_eump_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('a_eump_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(eu_grid[:, 0]+eu_grid[:, 2],
                    theta_star_eump_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('theta_eump_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(eu_grid[:, 0]+eu_grid[:, 2],
                    y_star_eump_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('y_eump_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(eu_grid[:, 0]+eu_grid[:, 2],
                    EU_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('EU_star_new')

        plt.show()



    for i,jj in enumerate(unempl_grid):
        print("VALUE", i, "IS", jj)
        a = jj[0]
        x = jj[1]
        a_prime = max(np.asscalar(a_func[t].predict(np.atleast_2d([a, x]), return_std=False)), 0)
        state = [a, x, t]
        xx =  np.array([a/2, 1.5, 1.0])
        try:
            solsol = root(foccs_risk, xx, state, method='hybr')
        except:
            pass
        print("unconstrained root success?", solsol.success)
        try:
            solconstr = root(foccs_risk_constrained, [solsol.x[0], 2.5, 1.25, 0], state,method='hybr')
        except:
            pass
        print("constrained root success?", solconstr.success)
        norm_distance = np.linalg.norm(solconstr.x[0:1] - solsol.x[0:1])
        if (norm_distance > 1e-5) or ((solconstr.success == False) and (solsol.success == False)):
                if (solconstr.success == False) and (solsol.success == False):
                    #status_collect[i, t - 1] = 0
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
                        solconstr = root(foccs_risk_constrained, [xx[0], xx[1], xx[2], 0], state,
                                     method='anderson')
                    except:
                        pass
                    if (solconstr.success == False):
                        status_collect[i, t - 1] = 0
                    else:
                        print("the right solution is")
                        print(solconstr.x)
                        a_star_new[i, t - 1] = solconstr.x[0]
                        theta_star_new[i, t - 1] = solconstr.x[1]
                        y_star_new[i, t - 1] = solconstr.x[2]
                else:
                    if (solconstr.success == True and solsol.x[0]<0 and solsol.x[0]>-5.5 and solsol.x[0]<10.0):
                        print("the right solution is solconstr")
                        print(solconstr.x)
                        a_star_new[i, t - 1] = min(solsol.x[0],0.0)
                        theta_star_new[i, t - 1] = solconstr.x[1]
                        y_star_new[i, t - 1] = solconstr.x[2]
                    else:
                        if (solsol.success==True and solsol.x[0]<10.0 and solsol.x[1]<6.0):
                            print("the right solution is solsol")
                            print(solsol.x)
                            a_star_new[i, t - 1] = solsol.x[0]
                            theta_star_new[i, t - 1] = solsol.x[1]
                            y_star_new[i, t - 1] = solsol.x[2]
                        else:
                            if (t > 7) and (status_collect[i, t - 2] == 1):
                                print("used past solution because of too many issues")
                                a_star_new[i, t - 1] = a_star_new[i, t - 2]
                                theta_star_new[i, t - 1] = theta_star_new[i, t - 2]
                                y_star_new[i, t - 1] = y_star_new[i, t - 2]
                                wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x,t)
                            else:
                                print("no solution works")
                                status_collect[i, t - 1] = 0



        else:
            print("the right solution is solsol")
            print(solsol.x)
            theta_star_new[i, t - 1] = solsol.x[1]
            y_star_new[i, t - 1] = solsol.x[2]
            a_star_new[i, t - 1] = solsol.x[0]


        if (status_collect[i, t - 1] == 0):
            U_new[i, t - 1] =  -eval_Uf([max(0,a),1.7,0.9], state)
            print("U_new",U_new[i,t-1] )
        else:
            U_new[i,t-1] = -eval_Uf([max(0,a_star_new[i, t - 1]),theta_star_new[i, t - 1],y_star_new[i, t - 1]], state)
            print("U_new",U_new[i,t-1] )


    controls_unemp = np.where(status_collect[:, t - 1] < 1.0)

    if plot_stuff==1:
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(unempl_grid[:, 0], a_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('a_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(unempl_grid[:, 0], theta_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('theta_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(unempl_grid[:, 0], y_star_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('y_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(unempl_grid[:, 0], U_new[:,t-1],s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
        plt.title('U_star_new')

        plt.show()


    #x_x = np.array([empl_grid[0,0], 1.0, 0.9])
    #e_state = [empl_grid[0,0], empl_grid[0,1], empl_grid[0,2], empl_grid[0,3], t]
    #print(e_state)
    #solsol = root(foc_empl_risk, xx, e_state, method='hybr')
    #print(solsol)
    

    for i, g in enumerate(empl_grid):
        print("value", i, "is", g)
        if (t==1):
            xx = np.array([g[0], 1.0, 0.9])
        else:
            xx = np.array([g[0], theta_star_emp_new[i, t - 2], y_star_emp_new[i, t - 2]])
        e_state = [g[0], g[1], g[2], g[3], t]
        try:
            solsol = root(foc_empl_risk, xx, e_state, method='hybr') # or hybr?
            solconsol = solsol
        except:
            pass
        print("the IPOPT solution is", solsol.x)
        #norm_distance = np.linalg.norm(solconsol.x[0:1] - solsol.x[0:1])
        if (solsol.x[1]<0) or (solsol.success == False):
            xxx = [solsol.x[0], max(solsol.x[0], 0.0), solsol.x[2], 0.2]
            try:
                solconsol = root(foc_empl_risk_constrained, xxx, e_state, method='hybr')
            except:
                pass
            if (solconsol.success == False):
                try:
                    solconsol = root(foc_empl_risk_constrained, xxx, e_state, method='anderson')
                except:
                    pass
            print("the constrained solution is", solconsol.x)
            if (solconstr.success == False):
                if (t > 6) and (status_collect_emp[i, t - 2] == 1):
                    print("used past solution")
                    a_star_emp_new[i, t - 1] = a_star_emp_new[i, t - 2]
                    theta_star_emp_new[i, t - 1] = theta_star_emp_new[i, t - 2]
                    y_star_emp_new[i, t - 1] = y_star_emp_new[i, t - 2]
                    wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], g[1], t)
                else:
                    print("no solution")
                    status_collect_emp[i, t - 1] = 0
            elif (solconstr.success == True):
                print("used constrained solution")
                a_star_emp_new[i, t - 1] = solconsol.x[0]
                theta_star_emp_new[i, t - 1] = solconsol.x[1]
                y_star_emp_new[i, t - 1] = solconsol.x[2]
                wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], g[1], t)
        else:
            print("used unconstrained solution")
            a_star_emp_new[i, t - 1] = solsol.x[0]
            theta_star_emp_new[i, t - 1] = solsol.x[1]
            y_star_emp_new[i, t - 1] = solsol.x[2]
            wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1],  g[1], t)


        E_new[i, t - 1] = -eval_Ef([max(0, a_star_emp_new[i, t - 1]), theta_star_emp_new[i, t - 1],
                                    y_star_emp_new[i, t - 1]], e_state)
        if (t == 1):
            E_prime_new[i, t - 1] = E_funprime(0.0, g[0], a_star_emp_new[i, t - 1], theta_star_emp_new[i, t - 1] , g[2])
        else:
            E_prime_new[i, t - 1] = E_funprime(E_prime_new[i, t - 2], g[0], a_star_emp_new[i, t - 1],
                                               theta_star_emp_new[i, t - 1], g[2])

        #print("E_y is",eval_grad_E_y([a_star_emp_new[i, t - 1],theta_star_emp_new[i, t - 1],y_star_emp_new[i, t - 1]], e_state))

        if (theta_star_emp_new[i, t - 1]>5.0) or (theta_star_emp_new[i, t - 1]< -0.001)\
                or (a_star_emp_new[i, t - 1]>8.0 or a_star_emp_new[i, t - 1]<0) :
            if (t>6) and (status_collect_emp[i, t - 2]==1):
                print("used past solution because of too many issues")
                a_star_emp_new[i, t - 1] = a_star_emp_new[i, t - 2]
                theta_star_emp_new[i, t - 1] = theta_star_emp_new[i, t - 2]
                y_star_emp_new[i, t - 1] = y_star_emp_new[i, t - 2]
                wage_star_emp_new[i, t - 1] = wage(y_star_emp_new[i, t - 1], theta_star_emp_new[i, t - 1], g[1], t)
                E_new[i, t - 1] = -eval_Ef([max(0, a_star_emp_new[i, t - 1]), theta_star_emp_new[i, t - 1],
                                            y_star_emp_new[i, t - 1]], e_state)
                if (t == 1):
                    E_prime_new[i, t - 1] = E_funprime(0.0, g[0], a_star_emp_new[i, t - 1],
                                                       theta_star_emp_new[i, t - 1], g[2])
                else:
                    E_prime_new[i, t - 1] = E_funprime(E_prime_new[i, t - 2], g[0], a_star_emp_new[i, t - 1],
                                                       theta_star_emp_new[i, t - 1], g[2])
            else:
                print("we are here")
                status_collect_emp[i, t - 1] = 0


    controls_emp = np.where( status_collect_emp[:,t-1] < 1.0 )
    joined =  controls_unemp + controls_emp + controls_eump
    joint = [x for xs in joined for x in xs]
    print('joint',joint)

    if plot_stuff==1:
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(empl_grid[:, 0], a_star_emp_new[:, t - 1], s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.title('a_star_emp_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(empl_grid[:, 2], theta_star_emp_new[:, t - 1])
        plt.title('thetaa_star_new')

        fig, ax = plt.subplots(figsize=(9, 5))
        plt.scatter(empl_grid[:, 2], E_new[:, t - 1])
        plt.title('E_star_new')
        plt.show()


################## STORING BEFORE INTERPOLATING ##################################


    E_prime_old = E_prime_new[:, t - 1]
    U_old = U_new[:, t - 1]
    E_old = E_new[:, t - 1]
    EU_old = EU_new[:, t - 1]

    a_star_old = a_star_new[:, t - 1]
    a_star_emp_old = a_star_emp_new[:, t - 1]
    a_star_eump_old = a_star_eump_new[:, t - 1]

    theta_star_old = theta_star_new[:, t - 1]
    theta_star_emp_old = theta_star_emp_new[:, t - 1]
    theta_star_eump_old = theta_star_eump_new[:, t - 1]

    y_star_old = y_star_new[:, t - 1]
    y_star_emp_old = y_star_emp_new[:, t - 1]
    y_star_eump_old = y_star_eump_new[:, t - 1]

    wage_star_old = wage_star_new[:, t - 1]
    wage_star_emp_old = wage_star_emp_new[:, t - 1]

    unempl_grid_old = unempl_grid
    empl_grid_old = empl_grid
    eu_grid_old = eu_grid

    #status_collect = np.ones((rsize, life))




t = 15
#theta = 2.0

y, x = np.mgrid[0.7:1.8:50j,  2.8:3.8:50j]

www = np.empty((50,50))


for i in range(50):
    for j in range(50):
        t_theta = max(np.asscalar(theta_wage_func[1].predict(np.atleast_2d([x[i,j],y[i,j]]),
                                                                         return_std=False)), 0)
        www[i,j] = wage(y[i,j],t_theta,x[i,j],t)

dy, dx = np.gradient(www)

print(dy)

skip = (slice(None, None, 3), slice(None, None, 3))

fig, ax = plt.subplots()
im = ax.imshow(www, extent=[x.min(), x.max(), y.min(), y.max()], cmap = 'RdBu')
ax.quiver(x[skip], y[skip], dx[skip], dy[skip])

fig.colorbar(im)
ax.set(aspect=1, title='Wage')
plt.xlabel('human capital')
plt.ylabel('firm productivity')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()



############# SIMULATION ##################

## initialization

wage_sim[0,:] = 0.0
y_sim[0,:] = 0.0
emp_status_sim[0,:] = 0

a_sim[0,:] = np.linspace(low_a+1.5, 0.8*(high_a), num=n_workers)
hum_k_sim[0,:] = np.random.uniform(low=1.2*low_x, high=0.8*high_x, size=(1, n_workers))

emp_matrix = np.random.uniform(low=0, high=1, size=(n_workers, life))
sep_matrix = np.random.uniform(low=0, high=1, size=(n_workers*2, life))
eu_matrix = np.random.uniform(low=0, high=1, size=(n_workers*2, life))


for ii in range(n_workers):
    for t in range(1,life):
        outcome_emp = emp_matrix[ii,t]
        outcome_sep = sep_matrix[ii,t]
        outcome_eu = eu_matrix[ii,t]

        print(t)
        print('random', outcome_emp)
        print('random sep', outcome_sep)
        print('random eu', outcome_eu)

        if emp_status_sim[t-1,ii]==1:
            print('employed problem')
            a_sim[t, ii] = R*max(a_emp_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii],wage_sim[t-1,ii], y_sim[t-1,ii]]),return_std=False),0)
            theta_sim[t, ii] = max(np.asscalar(theta_emp_func[life - t].predict(np.atleast_2d([a_sim[t - 1, ii],hum_k_sim[t - 1, ii],
                                                        wage_sim[t - 1, ii],y_sim[t - 1, ii]]),return_std=False)),0)

            jobchange = m_fun(theta_sim[t, ii] )*pi

            print('prob of job change', jobchange)
            hum_k_sim[t, ii] = g_fun(y_sim[t, ii-1], hum_k_sim[t - 1, ii])
            if outcome_sep < lamb:
                emp_status_sim[t, ii] = 0
                y_sim[t, ii] = y_sim[t - 1, ii]
                wage_sim[t, ii] = wage_sim[t - 1, ii]
                job_to_job_sim[t,ii] = 0
            else:
                if jobchange > outcome_emp:
                    y_sim[t, ii] = np.asscalar(y_emp_func[life-t].predict(
                        np.atleast_2d([a_sim[t - 1, ii], hum_k_sim[t - 1, ii], wage_sim[t - 1, ii], y_sim[t - 1, ii]]),
                        return_std=False))
                    wage_sim[t, ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t, ii], life - t)
                    job_to_job_sim[t,ii] = 1
                    print('prob of bad match', t_fun(y_sim[t, ii], hum_k_sim[t, ii]))
                    if outcome_eu < t_fun(y_sim[t, ii],hum_k_sim[t,ii]):
                        emp_status_sim[t, ii] = 2
                    else:
                        emp_status_sim[t, ii] = 1
                else:
                    y_sim[t, ii] = y_sim[t - 1, ii]
                    wage_sim[t, ii] = wage_sim[t - 1, ii]
                    job_to_job_sim[t, ii] = 0
                    emp_status_sim[t, ii] = 1

        elif emp_status_sim[t-1,ii]==0:
            print('unemployed problem')
            a_sim[t,ii] = R*max(np.asscalar(a_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii],hum_k_sim[t-1,ii]]), return_std=False)),0)
            y_sim[t, ii] = np.asscalar(y_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii]]), return_std=False))
            theta_sim[t, ii] = np.asscalar(theta_func[life-t].predict(np.atleast_2d([a_sim[t - 1, ii],
                                                                                    hum_k_sim[t-1,ii]]), return_std=False))
            wage_sim[t,ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t-1,ii], life-t)
            if (wage_sim[t,ii]<0):
                wage_sim[t, ii] = wage_sim[t - 1, ii]

            print('prob of job', m_fun(theta_sim[t, ii]))
            hum_k_sim[t, ii] =  hum_k_sim[t - 1, ii]
            job_to_job_sim[t, ii] = 0
            if outcome_emp < m_fun(theta_sim[t, ii]):
                print('prob of bad match', t_fun(y_sim[t, ii],hum_k_sim[t,ii]))
                if outcome_eu < t_fun(y_sim[t, ii],hum_k_sim[t,ii]):
                    emp_status_sim[t, ii] = 2
                else:
                    emp_status_sim[t, ii] = 1
            else:
                emp_status_sim[t, ii] = 0
                
        elif emp_status_sim[t-1,ii]==2:
            print('eu problem')
            a_sim[t,ii] = R*max(np.asscalar(a_eump_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii],hum_k_sim[t-1,ii], wage_sim[t,ii-1]]), return_std=False)),0)
            y_sim[t, ii] = np.asscalar(y_eu_func[life-t].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii],wage_sim[t,ii-1]]), return_std=False))
            theta_sim[t, ii] = np.asscalar(theta_eump_func[life-t].predict(np.atleast_2d([a_sim[t - 1, ii],
                                                                                    hum_k_sim[t-1,ii],wage_sim[t,ii-1]]), return_std=False))
            wage_sim[t,ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t-1,ii], life-t)

            hum_k_sim[t, ii] =  hum_k_sim[t - 1, ii]
            job_to_job_sim[t, ii] = 0
            if outcome_emp < m_fun(theta_sim[t, ii]):
                print('prob of bad match', t_fun(y_sim[t, ii], hum_k_sim[t, ii]))
                if outcome_eu < t_fun(y_sim[t, ii],hum_k_sim[t,ii]):
                    emp_status_sim[t, ii] = 2
                else:
                    emp_status_sim[t, ii] = 1
            else:
                emp_status_sim[t, ii] = 0


             

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

for t in range(1,life):
    for ii in range(n_workers):
        if emp_status_sim[t,ii]>1:
            emp_status_sim[t, ii] = 0
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


for i in range(0,life,2):
  print(i)
  j = int(i/2)
  mean_annual_wage[j] = (mean_wage[i+1] + mean_wage[i]) / 2
  print(mean_annual_wage[j],mean_wage[i+1] , mean_wage[i] )
  mean_annual_assets[j] = (mean_assets[i+1] + mean_assets[i]) / 2
  print(mean_annual_assets[j],mean_assets[i+1] , mean_assets[i] )
  transition_annual_rate[j] = (transition_rate[i +1] + transition_rate[i]) / 2

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,years),mean_annual_wage[1:years], 'r', lw=3, zorder=9, label='agent 0')
plt.title('mean wage')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,years),mean_annual_assets[1:years], 'r', lw=3, zorder=9, label='agent 0')
plt.title('mean assets')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,years),transition_annual_rate[1:years], 'r', lw=3, zorder=9, label='agent 0')
plt.title('mean EE rate')

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


