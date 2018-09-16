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


#from stuff import classifier
from params import *
from declare_arrays import *

############################################################################################
############################################################################################

def m_fun(theta):
    return 1 - np.exp(-eta * theta)

def m_funprime(theta):
    return eta * np.exp(-eta * theta)

def q_fun(theta):
    return (1 - np.exp(-eta * theta))/theta

def q_funprime(theta):
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
    return f(A, y, x) - k_fun(y)/(cosa1*q_fun(theta)) - (max(y - g_fun(y,x),0))*2


def E_fun(U, E, a, a_prime, w,theta,E_tilde): #
    return u(c_e(w, a, a_prime)) + beta*((1-lamb)*E + lamb*U) + beta*pi*m_fun(theta)*E_tilde


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
    y = p*2.5
    return x + phi*(y-x)*l_fun(y-x)

def g_fun_prime(p, x):
    y = p*2.5
    return phi*l_fun(y-x) + phi*(y-x)*l_fun_prime(y-x)


def foc_empl(thetas,other):
    a = np.asscalar(other[0])
    x = np.asscalar(other[1])
    w = np.asscalar(other[2])
    y = np.asscalar(other[3])
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

    UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_int,x]), return_std=False))
    EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_int,g_fun(y,x), w, y ]), return_std=False))
    #E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w,y ]), return_std=False))
    EE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_tilde,g_fun(y_tilde,x), w_wage, y_tilde ]), return_std=False))
    E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime_tilde,g_fun(y_tilde,x), w_wage,y_tilde ]), return_std=False))
    #yy_tilde = np.asscalar(y_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y_tilde, x), w_wage, y_tilde]), return_std=False))
    #ww_wage = wage(yy_tilde, theta_tilde, g_fun(yy_tilde,x), t)
    #EEE_tilde = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_tilde,g_fun(yy_tilde,x), ww_wage, yy_tilde ]), return_std=False))


    ce = c_e(w, a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w, a_prime, a_prime_int)
    cetilde = c_e(w_wage, a_prime, a_prime_tilde)

    foc1 = -u_prime(ce) + (1-pi*m_fun(theta))*beta * R * ((1 - lamb) * u_prime(cet)+
                                      (lamb) * u_prime(cut)) + beta*R*pi*m_fun(theta)*u_prime(cetilde)
    foc2 = pi*m_funprime(theta) * ( E_fun(UU, EE_tilde, a_prime, a_prime_tilde, w_wage, theta_tilde, EE_tilde)
    -lamb*U_fun(UU, EE, a_prime, a_int, theta) - (1-lamb)*E_fun(UU, EE, a_prime, a_prime_int, w, theta_tilde, EE_tilde)) + \
     theta*E_funprime(E_prime_tilde, a_prime, a_prime_tilde, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y_tilde, x)*q_fun(theta)*cosa1 - k_funprime(y_tilde)

    return [foc1.item(), foc2.item(), foc3]


def foccs(thetas, params):
    a = params[0]
    x = params[1]
    kkt = params[2]
    t = params[3]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y = thetas[2]
    #vincolo = thetas[3]
    w_wage = wage(y, theta, g_fun(y,x), t)
    if (interp_strategy == 'gpr'):
        a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        if (kkt==0):
            a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
        else:
            a_int = 0.0
        theta_tilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
        E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
        UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
        EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
    elif (interp_strategy == 'standard'):
        a_emp_int = a_emp_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y )
        a_int = a_func(a_prime, x)
        UU = U_func(a_prime, x)
        EE = E_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y)
        E_prime = Eprime_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y)
    ##
    cu = c_u(a, a_prime)
    cut = c_u(a_prime, a_int)
    cet = c_e(w_wage, a_prime, a_emp_int)

    ##

    foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) #+\
        #np.amax([0, +vincolo])
    foc2 = m_funprime(theta) * ( E_fun(UU, EE, a_prime, a_emp_int, w_wage,theta_tilde,E_prime_tilde)
    - U_fun(UU, EE, a_prime, a_int, theta) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)
    #foc4 = np.amax([0, -vincolo]) - a_prime

    return [foc1.item(), foc2.item(), foc3]


def foccs_constrained(thetas, params):
    a = params[0]
    x = params[1]
    kkt = params[2]
    t = params[3]
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    a_prime = thetas[0]
    theta = thetas[1]
    y = thetas[2]
    vincolo = thetas[3]
    w_wage = wage(y, theta, g_fun(y,x), t)
    if (interp_strategy == 'gpr'):
        if (kkt==0):
            a_int = max(np.asscalar(a_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
        else:
            a_int = 0.0
        a_emp_int = np.asscalar(a_emp_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        theta_tilde = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w_wage, y]), return_std=False))
        E_prime_tilde = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_emp_int, g_fun(y, x), w_wage, y]),
                                   return_std=False))
        UU = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime,x]), return_std=False))
        EE = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        E_prime = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        #theta_emp_try[i] = np.asscalar(theta_emp_func[t].predict(np.atleast_2d([a,y,w,y,return_std=False))
    elif (interp_strategy == 'standard'):
        a_emp_int = a_emp_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y )
        a_int = a_func(a_prime, x)
        UU = U_func(a_prime, x)
        EE = E_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y)
        E_prime = Eprime_func(np.asscalar(a_prime),g_fun(y,x), w_wage,y)

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


def interpolator():
    return 2
############################################################################################
############################################################################################



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

for t in range(1,life+1):
    print('*****', 'iteration number', t, '******')

    print('this is age', str(life - t))
    if (interp_strategy == 'gpr'):
        # Instanciate a Gaussian Process model
        kernel = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e2))
        kernel_bis = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e3)) + WhiteKernel(noise_level=1e-3,
                                                                              noise_level_bounds=(1e-10, 1e+1))
        kernel_E = 1.0 * RBF(7.0, (1e-3, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e+1))


        U_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        a_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        E_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=5)
        Eprime_func[t] = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=10)
        a_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        y_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        theta_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        y_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)
        theta_emp_func[t] = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=5)



        with warnings.catch_warnings():
            warnings.simplefilter("always")  # if you get useless warnings print "ignore"
            U_func[t].fit(unempl_grid_old, U_old)
            a_func[t].fit(unempl_grid_old, a_star_old)
            y_func[t].fit(unempl_grid_old, y_star_old)
            theta_func[t].fit(unempl_grid_old, theta_star_old)
            if t>1:
                E_func[t].fit(empl_grid_b_old, E_old_b)
                a_emp_func[t].fit(empl_grid_b_old, a_star_emp_old_b)
                Eprime_func[t].fit(empl_grid_b_old, E_prime_old_b)
                y_emp_func[t].fit(empl_grid_b_old, y_star_emp_old)
                theta_emp_func[t].fit(empl_grid_b_old, theta_star_emp_old)
            else:
                E_func[t].fit(empl_grid, E_old)
                a_emp_func[t].fit(empl_grid, a_star_emp_old)
                Eprime_func[t].fit(empl_grid, E_prime_old)
                y_emp_func[t].fit(empl_grid, y_star_emp_old)
                theta_emp_func[t].fit(empl_grid, theta_star_emp_old)



######################## UNEMPLOYED PROBLEM ##########################################

    for i,xx in enumerate(unempl_grid):
        print("value", i, "is", xx)
        a_prime = np.asscalar(a_func[t].predict(np.atleast_2d([xx[0], xx[1]]), return_std=False))
        thetas = [xx[0]*0.55, 1.0, 0.7] #[xx[0]*0.55, 1.0, 0.7, 0]
        #kkt = constraint[i,t-1]
        try:
            solsol = root(foccs, thetas, [xx[0],xx[1],0,t], method='hybr')
        except:
            pass
        try:
            solconstr = root(foccs_constrained, [max(0,solsol.x[0]), solsol.x[1], solsol.x[2],0], [xx[0],xx[1],0,t], method='hybr')
        except:
            pass
        x_p = xx[1]


        if solsol.success == False:
            print('***')
            print('error at point', i)
            print("but solution is", solsol.x[0],solsol.x[1],solsol.x[2])
            print('***')
            a_pred = max(np.asscalar(a_func[t].predict(np.atleast_2d([xx[0],xx[1]]), return_std=False)), 0)
            try:
                solsol = root(foccs, [a_pred, 1.4, 0.7], [xx[0], xx[1], 0, t-1], method='hybr') # [xx[0]*0.5, 0.99, 0.7, 0]
            except:
                pass
            try:
                solconstr = root(foccs_constrained, [max(0, a_pred), solsol.x[1], solsol.x[2], 0],
                                [xx[0], xx[1], 0, t], method='hybr')
            except:
                pass
            x_p = xx[1]

            if solsol.success == False:
                print('***')
                print('again an error at point', i)
                print("with solution is", solsol.x[0], solsol.x[1], solsol.x[2])
                print('***')
                status_collect[i, t - 1] = 0
            else:
                print("ok solution is", solsol.x[0], solsol.x[1], solsol.x[2])
                print("constrained solution is", solconstr.x[0], solconstr.x[1], solconstr.x[2], solconstr.x[3])
                status_collect[i, t - 1] = 1
        else:
            print("ok solution is", solsol.x[0],solsol.x[1],solsol.x[2])
            print("constrained solution is", solconstr.x[0],solconstr.x[1],solconstr.x[2],solconstr.x[3])
            status_collect[i,t-1] = 1


        a_star_new[i, t - 1] = solsol.x[0]

        if solconstr.x[3] > 0 or solsol.x[0]<0 :
            constraint[i, t - 1] = 1
            theta_star_new[i, t - 1] = solconstr.x[1]
            y_star_new[i, t - 1] = solconstr.x[2]
            wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)
        elif solconstr.x[3] < 0 and solsol.x[0]<solconstr.x[0] and  solsol.x[1]>solconstr.x[1]:
            a_star_new[i, t - 1] = solconstr.x[0]
            constraint[i, t - 1] = 0
            theta_star_new[i, t - 1] = solconstr.x[1]
            y_star_new[i, t - 1] = solconstr.x[2]
            wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)
        else:
            constraint[i, t - 1] = 0
            theta_star_new[i, t - 1] = solsol.x[1]
            y_star_new[i, t - 1] = solsol.x[2]
            wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)

        print('and wage is', wage_star_new[i, t - 1])
        if solsol.x[1] > 10.0 or a_star_new[i, t - 1]< -10.0:
            solsol.success = False
            status_collect[i, t - 1] = 0
        elif solconstr.x[1] > 5.5:  #and solconstr.x[3] > 0
            solconstr.success = False
            status_collect[i, t - 1] = 0

    controls_unemp = np.where(status_collect[:, t - 1] < 1.0)


    #empl_grid_b[:, 2] = wage_star_new[:, t - 1]  # [:,0]
    #empl_grid_b[:, 3] = y_star_new[:, t - 1]  # [:,0]
    empl_grid_b[:, 2] = empl_grid[:,2]
    empl_grid_b[:, 3] = empl_grid[:, 3]
    empl_grid_b_old = empl_grid_b
    #empl_grid_old = empl_grid
    unempl_grid_old = unempl_grid



##################### EMPLOYED PROBLEM ##########################################
# need old and new constraint vector
    '''
    for i,g in enumerate(empl_grid):
        #print("value", i, "is", g)
        if (interp_strategy == 'gpr'):
            a_prime = np.asscalar(a_emp_func.predict(np.atleast_2d([g[0],g[1],g[2],g[3]]), return_std=False))
        elif (interp_strategy == 'standard'):
            a_prime = a_emp_func(g[0], g[1], g[2], g[3])
        otherr  = [g[0],g[1],g[2],g[3],constraint[i,t-1]]
        sol = root(foc_empl, g[0], otherr, method='hybr') # or hybr?
        a_star_emp_new[i,t-1] = sol.x
        if sol.success == False:
            print('***')
            print('error in employed problem at', g)
            print("but solution is", sol.x)
            print('***')
            status_collect[i,t-1] = 0
        else:
            status_collect[i, t - 1] = 1
            #print("ok solution is", sol.x)
    '''

    for i,g in enumerate(empl_grid_b):
        print("value", i, "is", g)
        x_p = g[1]
        if (interp_strategy == 'gpr'):
            a_prime = np.asscalar(a_emp_func[t].predict(np.atleast_2d([g[0],g[1],g[2],g[3]]), return_std=False))
        elif (interp_strategy == 'standard'):
            a_prime = a_emp_func(g[0], g[1], g[2], g[3])
        otherr  = [g[0],g[1],g[2],g[3],t]
        try:
            sol = root(foc_empl, [g[2],0.7,1.1], otherr, method='hybr') # or hybr?
        except:
            pass
        if sol.success == False:
            print('***')
            print('error in employed b problem at', g)
            print("but solution is", sol.x)
            print('***')
            try:
                sol = root(foc_empl, [0.5*g[2], 0.6, 1.0], otherr, method='hybr')  # or hybr?
            except:
                pass
            if sol.success == False:
                print('***')
                print('again error in employed b problem at', g)
                print("but solution is", sol.x)
                print('***')
                status_collect[i, t - 1] = 0
            else:
                status_collect[i, t - 1] = 1
                print("ok solution is", sol.x)
                a_star_emp_new_b[i, t - 1] = sol.x[0]
                theta_star_emp_new[i, t - 1] = sol.x[1]
                y_star_emp_new[i, t - 1] = sol.x[2]
                wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)
                if sol.x[0] == a_star_emp_new_b[i-1, t - 1]:
                    status_collect[i,t-1] = 0
        else:
            status_collect[i, t - 1] = 1
            print("ok solution is", sol.x)
            a_star_emp_new_b[i, t - 1] = sol.x[0]
            theta_star_emp_new[i, t - 1] = sol.x[1]
            y_star_emp_new[i, t - 1] = sol.x[2]
            wage_star_emp_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)

            if sol.x[0] == a_star_emp_new_b[i - 1, t - 1] or a_star_emp_new_b[i - 1, t - 1]<0:
                status_collect[i, t - 1] = 0

    controls_emp = np.where( status_collect[:,t-1] < 1.0 )
    joined =  controls_unemp + controls_emp
    joint = [x for xs in joined for x in xs]
    print('joint',joint)



##################### VALUE FUNCTIONS ##########################################


    for i,k in enumerate(empl_grid_b):
        a_prime_emp = a_star_emp_new_b[i,t-1]
        a_prime = a_star_new[i,t-1]
        w = np.asscalar(wage_star_new[i, t - 1])
        y = np.asscalar(y_star_new[i, t - 1])
        y_tilde = y_star_emp_new[i, t - 1]
        w_tilde = wage_star_emp_new[i, t - 1]
        a = k[0]
        x = k[1]
        ww = k[2]
        yy = k[3]
        UU_nw = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime, x]), return_std=False))
        UE_nw = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime, g_fun(y, x), w, y]), return_std=False))
        EU_nw = np.asscalar(U_func[t].predict(np.atleast_2d([a_prime_emp, x]), return_std=False))
        EE_nw = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_emp, g_fun(yy, x), ww, yy]), return_std=False))
        EEtilde_nw = np.asscalar(E_func[t].predict(np.atleast_2d([a_prime_emp, g_fun(y_tilde, x), w_tilde, y_tilde]), return_std=False))
        E_prime_nw = np.asscalar(Eprime_func[t].predict(np.atleast_2d([a_prime_emp,g_fun(yy,x), ww,yy ]), return_std=False))
        if t>1:
            if (constraint[i, t - 1] == 0):
                U_new[i,t-1] = U_fun(UU_nw,UE_nw, a, a_star_new[i,t-1],theta_star_new[i, t - 1])
                E_new_b[i,t-1] = E_fun(EU_nw,EE_nw, a, a_star_emp_new_b[i,t-1], ww, max(theta_star_emp_new[i, t - 1],0), EEtilde_nw)
                E_prime_new_b[i,t-1] = E_funprime(E_prime_nw, a, a_star_emp_new_b[i,t-1], w)
            else:
                U_new[i,t-1] = U_fun(UU_nw,UE_nw, a, 0,theta_star_new[i, t - 1])
                E_new_b[i,t-1] = E_fun(EU_nw,EE_nw, a, a_star_emp_new_b[i,t-1], ww, max(theta_star_emp_new[i, t - 1],0), EEtilde_nw)
                E_prime_new_b[i,t-1] = E_funprime(E_prime_nw, a, a_star_emp_new_b[i,t-1], w)
        else:
            if (constraint[i, t - 1] == 0):
                U_new[i, t - 1] = U_fun(0, 0, a, a_star_new[i,t-1], theta_star_new[i, t - 1])
                E_new_b[i, t - 1] = E_fun(0, 0, a, a_star_emp_new_b[i, t - 1], ww, max(theta_star_emp_new[i, t - 1],0), 0)
                E_prime_new_b[i, t - 1] = E_funprime(0, a, a_star_emp_new_b[i, t - 1], w)
            else:
                U_new[i, t - 1] = U_fun(0, 0, a, 0, theta_star_new[i, t - 1]) # a_star_new[i,t-1], a_int,
                E_new_b[i, t - 1] = E_fun(0, 0, a, a_star_emp_new_b[i, t - 1], ww, max(theta_star_emp_new[i, t - 1],0), 0)
                E_prime_new_b[i, t - 1] = E_funprime(0, a, a_star_emp_new_b[i, t - 1], w)


    '''
    for i,k in enumerate(empl_grid):
        a = k[0]
        x = k[1]
        w = k[2]
        y = k[3]
        if t>1:
            if (constraint[i, t - 1] == 0):
                E_new[i,t-1] = E_fun(U_new[i,t-2],E_new[i,t-2], a, a_star_new[i,t-1], theta_star_old[i])  # a_star_new[i,t-1],a_emp_int,
                E_prime_new[i, t - 1] = E_funprime(E_prime_new[i, t - 2], a, a_star_emp_new[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int
            else:
                E_new[i,t-1] = E_fun(U_new[i,t-2],E_new[i,t-2], a, 0, theta_star_old[i])  # a_star_new[i,t-1],a_emp_int,
                E_prime_new[i, t - 1] = E_funprime(E_prime_new[i, t - 2], a, a_star_emp_new[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int
        else:
            if (constraint[i, t - 1] == 0):
                E_new[i, t - 1] = E_fun(0, 0, a, a_star_new[i, t - 1], theta_star_old[i]) # a_star_new[i, t - 1], a_emp_int
                E_prime_new[i, t - 1] = E_funprime(0, a, a_star_emp_new[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int
            else:
                E_new[i, t - 1] = E_fun(0, 0, a, 0, theta_star_old[i]) # a_star_new[i, t - 1], a_emp_int
                E_prime_new[i, t - 1] = E_funprime(0, a, a_star_emp_new[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int
    '''




################## STORING BEFORE INTERPOLATING ##################################


    E_old_b = E_new_b[:, t - 1]
    E_prime_old_b = E_prime_new_b[:, t - 1]
    U_old = U_new[:, t - 1]

    a_star_old = a_star_new[:, t - 1]
    theta_star_old = theta_star_new[:, t - 1]
    y_star_old = y_star_new[:, t - 1]
    wage_star_old = wage_star_new[:, t - 1]

    a_star_emp_old_b = a_star_emp_new_b[:, t - 1]
    theta_star_emp_old = theta_star_emp_new[:, t - 1]
    y_star_emp_old = y_star_emp_new[:, t - 1]
    wage_star_emp_old = wage_star_emp_new[:, t - 1]




################ FIXING MATRICES AND INTERPOLATING #################################

    if (interp_strategy == 'gpr'):
        if any(joint):  # [0]
            print("status collector is not empty:")
            print('we have to change sizes')

            unempl_grid_old = np.delete(unempl_grid_old, joint, axis=0)
            empl_grid_b_old = np.delete(empl_grid_b_old, joint, axis=0)

            a_star_old = np.delete(a_star_old, joint, axis=0)
            wage_star_old = np.delete(wage_star_old, joint, axis=0)
            theta_star_old = np.delete(theta_star_old, joint, axis=0)
            y_star_old = np.delete(y_star_old, joint, axis=0)

            a_star_emp_old_b = np.delete(a_star_emp_old_b, joint, axis=0)
            wage_star_emp_old = np.delete(wage_star_emp_old, joint, axis=0)
            theta_star_emp_old = np.delete(theta_star_emp_old, joint, axis=0)
            y_star_emp_old = np.delete(y_star_emp_old, joint, axis=0)

            U_old = np.delete(U_old, joint, axis=0)
            E_old_b = np.delete(E_old_b, joint, axis=0)
            E_prime_old_b = np.delete(E_prime_old_b, joint, axis=0)
            rsize_old = unempl_grid_old.shape[0]
            print('new row size is', rsize_old)
        else:
            print('rsize does not change')



    a_u_try = np.empty_like(a_grid)
    a_u_highprod = np.empty_like(a_grid)
    a_e_try = np.empty_like(a_grid)
    E_e_try = np.empty_like(a_grid)
    U_u_try = np.empty_like(a_grid)
    y_try = np.empty_like(a_grid)
    theta_try = np.empty_like(a_grid)
    theta_emp_try = np.empty_like(a_grid)

    for i, aval in enumerate(a_grid):
        if (constraint[i, t - 1] == 0):
            a_u_try[i] = max(np.asscalar(a_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False)),0)
        else:
            a_u_try[i] = max(np.asscalar(a_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False)),0)
        a_e_try[i] = np.asscalar(
            a_emp_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                               return_std=False))
        U_u_try[i] = np.asscalar(U_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
        E_e_try[i] = E_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                               return_std=False)
        y_try[i] = np.asscalar(y_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
        theta_try[i] = np.asscalar(theta_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
        theta_emp_try[i] = max(np.asscalar(theta_emp_func[t].predict(np.atleast_2d([aval, (high_x + low_x) / 2 ,
                 (low_w + high_w) / 2 , (low_y + high_y) / 2]), return_std=False)),0)


    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid_old[:, 0], a_star_old,s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
    #plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, U_u_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(unempl_grid_old[:,0], U_old)
    #plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, y_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(unempl_grid_old[:,0], y_star_old)
    #plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, theta_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(unempl_grid_old[:,0], theta_star_old)
    #plt.show()


    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, E_e_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(empl_grid_b_old[:, 0], E_old_b)
    #plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, a_e_try, 'r', lw=3, zorder=9, label='employed (interpolation)')
    plt.scatter(empl_grid_b_old[:, 0], a_star_emp_old_b, s=50, zorder=10, edgecolors=(0, 0, 0))


    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, theta_emp_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(unempl_grid_old[:,0], theta_star_emp_old)
    #plt.show()


    #plt.show()


    #fig, ax = plt.subplots(figsize=(9, 5))
    #plt.scatter(unempl_grid_old[:,0], wage_star_old)
    plt.show()



############# SIMULATION ##################

## initialization

wage_sim[0,:] = 0
y_sim[0,:] = 0
emp_status_sim[0,:] = 0

a_sim[0,:] = np.linspace(low_a+1.5, 1.5*(low_a+1.5), num=n_workers)
hum_k_sim[0,:] = np.random.uniform(low=low_x, high=high_x, size=(1, n_workers))

emp_matrix = np.random.uniform(low=0, high=1, size=(n_workers, life))
sep_matrix = np.random.uniform(low=0, high=1, size=(n_workers*2, life))

for ii in range(n_workers):
    for t in range(1,life):
        outcome_emp = emp_matrix[ii,t]
        outcome_sep = (sep_matrix[ii,t] + emp_matrix[ii,t])/2

        print(t)
        print('random', outcome_emp)
        print('random sep', outcome_sep)

        if emp_status_sim[t-1,ii]==1:
            print('employed problem')
            a_sim[t, ii] = R*max(a_emp_func[life-t+1].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii], wage_sim[t-1,ii], y_sim[t-1,ii]]),return_std=False),0)
            theta_sim[t, ii] = max(np.asscalar(theta_emp_func[life - t + 1].predict(np.atleast_2d([a_sim[t - 1, ii],hum_k_sim[t - 1, ii],
                                                        wage_sim[t - 1, ii],y_sim[t - 1, ii]]),return_std=False)),0)

            jobchange = m_fun(theta_sim[t, ii] )*pi

            print('prob of job change', jobchange)

            if outcome_sep < lamb:
                emp_status_sim[t, ii] = 0
                y_sim[t, ii] = y_sim[t - 1, ii]
                wage_sim[t, ii] = wage_sim[t - 1, ii]
                job_to_job_sim[t,ii] = 0
            else:
                emp_status_sim[t, ii] = emp_status_sim[t-1, 0]
                if jobchange > outcome_emp:
                    y_sim[t, ii] = np.asscalar(y_emp_func[t].predict(
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
            a_sim[t,ii] = R*max(np.asscalar(a_func[life-t+1].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii]]), return_std=False)),0)
            y_sim[t, ii] = np.asscalar(y_func[life-t+1].predict(np.atleast_2d([a_sim[t-1,ii], hum_k_sim[t-1,ii]]), return_std=False))
            theta_sim[t, ii] = np.asscalar(theta_func[life-t+1].predict(np.atleast_2d([a_sim[t - 1, ii], hum_k_sim[t-1,ii]]), return_std=False))
            wage_sim[t,ii] = wage(y_sim[t, ii], theta_sim[t, ii], hum_k_sim[t-1,ii], life-t)
            hum_k_sim[t, ii] =  hum_k_sim[t - 1, ii]
            job_to_job_sim[t, ii] = 0
            if outcome_emp < m_fun(theta_sim[t, ii]):
                emp_status_sim[t, ii] = 1
            else:
                emp_status_sim[t, ii] = emp_status_sim[ t - 1,ii]

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


print(np.linspace(1, life, num=life).T,mean_wage.T, mean_assets.T,unemployment_rate.T,transition_rate.T)


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),mean_wage[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),mean_assets[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),unemployment_rate[1:life], 'r', lw=3, zorder=9, label='agent 0')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2,life+1),transition_rate[1:life], 'r', lw=3, zorder=9, label='agent 0')



plt.show()

