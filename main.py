############################################################################################
############################# MAIN CODE FOR STUDENT LOANS PROJECT ##########################
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
    return f(A, y, x) - k_fun(y)/(cosa1*q_fun(theta))


def E_fun(U, E, a, a_prime, w):
    return u(c_e(w, a, a_prime)) + beta*((1-lamb)*E + lamb*U)


def E_funprime(E_prime, a, a_prime, w):
    return u_prime(c_e(w, a, a_prime)) + beta*(1-lamb)*E_prime  # E_prime function of a_prime

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
    y = p*10
    return x + phi*(y-x)*l_fun(y-x)

def g_fun_prime(p, x):
    y = p*10
    return phi*l_fun(y-x) + phi*(y-x)*l_fun_prime(y-x)


def foc_empl(a_emp,other):
    a = np.asscalar(other[0])
    x = np.asscalar(other[1])
    w = np.asscalar(other[2])
    y = np.asscalar(other[3])
    kkt = np.asscalar(other[4])
    if (interp_strategy == 'gpr'):
        a_prime_int = np.asscalar(a_emp_func.predict(np.atleast_2d([a_emp, g_fun(y,x), w, y]), return_std=False))
        if (kkt==0):
            a_int = max(np.asscalar(a_func.predict(np.atleast_2d([a_emp, x]), return_std=False)),0)
        else:
            a_int = 0
    elif (interp_strategy == 'standard'):
        a_prime_int = a_emp_func(a_emp[0], g_fun(y,x), w,y )
        a_int = a_func(a_emp[0], x)

    return -u_prime(c_e(w, a, a_emp)) + \
           beta * R * ((1 - lamb) * u_prime(c_e(w, a_emp, a_prime_int)) +
                       (lamb) * u_prime(c_u(a_emp, a_int)))


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
        a_emp_int = np.asscalar(a_emp_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        if (kkt==0):
            a_int = max(np.asscalar(a_func.predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
        else:
            a_int = 0.0
        UU = np.asscalar(U_func.predict(np.atleast_2d([a_prime,x]), return_std=False))
        EE = np.asscalar(E_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        E_prime = np.asscalar(Eprime_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
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
    foc2 = m_funprime(theta) * ( E_fun(UU, EE, a_prime, a_emp_int, w_wage)
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
            a_int = max(np.asscalar(a_func.predict(np.atleast_2d([a_prime, x]), return_std=False)),0)
        else:
            a_int = 0.0
        a_emp_int = np.asscalar(a_emp_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        UU = np.asscalar(U_func.predict(np.atleast_2d([a_prime,x]), return_std=False))
        EE = np.asscalar(E_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
        E_prime = np.asscalar(Eprime_func.predict(np.atleast_2d([a_prime,g_fun(y,x), w_wage,y ]), return_std=False))
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
    foc2 = m_funprime(theta) * ( E_fun(UU, EE, a_prime, a_emp_int, w_wage)
    - U_fun(UU, EE, a_prime, a_int, theta) ) + \
     theta*E_funprime(E_prime, a_prime, a_emp_int, w_wage)* q_funprime(theta) *  k_fun(y) / (cosa1*q_fun(theta))
    foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)
    foc4 = np.amax([0, -vincolo]) - a_prime

    return [foc1.item(), foc2.item(), foc3, foc4]  #np.array().ravel()


def interpolator():
    return 2
############################################################################################
############################################################################################

if (interp_strategy == 'gpr'):

    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e2))
    kernel_bis = C(1.0, (1e-3, 1e-3))* RBF(7, (1e-3, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e+1))
    #kernel = C(constant_value=1.0, constant_value_bounds=(1e-3, 1e-3)) * RBF(length_scale=1, length_scale_bounds=(1e-4, 1e-3))
    #  1e-3 in C  The RBF kernel with a large length-scale enforces this component to be smooth
    U_func = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=10)
    a_func = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=10)
    E_func = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=10)
    Eprime_func = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=10)
    a_emp_func = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=10)

    with warnings.catch_warnings():
        warnings.simplefilter("always")  # if you get useless warnings print "ignore"
        a_func.fit(unempl_grid, a_star_old)
        U_func.fit(unempl_grid, U_old)
        a_emp_func.fit(empl_grid, a_star_emp_old)
        E_func.fit(empl_grid, E_old)
        Eprime_func.fit(empl_grid, E_prime_old)

elif (interp_strategy == 'standard'):
    controls_unemp = np.where(status_collect < 1.0)
    controls_emp = np.where(status_collect > 1.0)
    joint = controls_unemp + controls_emp
    a_star_old = a_star_old.reshape((m_x, m, 1))
    U_old = a_star_old.reshape((m_x, m, 1))
    array = np.ma.masked_invalid(AA)
    newAA = AA[~array.mask]
    newWW = WW[~array.mask]
    newarr = a_star_old[~array.mask]
    newUrr = U_old[~array.mask]
    a_func = lambda q, h: interpolate.griddata((newAA, newWW), newarr.ravel(), (q, h), method='linear')
    U_func = lambda q, h: interpolate.griddata((newAA, newWW), newUrr.ravel(), (q, h), method='linear')
    a_emp_func = interpolate.Rbf(empl_assets, empl_prod, empl_wage, empl_firm, a_star_emp_old)
    E_func = interpolate.Rbf(empl_assets, empl_prod, empl_wage, empl_firm, E_old)
    Eprime_func = interpolate.Rbf(empl_assets, empl_prod, empl_wage, empl_firm, E_prime_old)



######################## UNEMPLOYED PROBLEM ##########################################

for t in range(1,life+1):

    print('this is age', str(life - t))

    print('*****', 'iteration number', t ,  '******')
    for i,xx in enumerate(unempl_grid):
        print("value", i, "is", xx)
        if (interp_strategy == 'gpr'):
            a_prime = np.asscalar(a_func.predict(np.atleast_2d([xx[0], xx[1]]), return_std=False))
            thetas = [xx[0]*0.55, 1.0, 0.7] #[xx[0]*0.55, 1.0, 0.7, 0]
            kkt = constraint[i,t-1]
            try:
                solsol = root(foccs, thetas, [xx[0],xx[1],kkt,t], method='hybr')
            except:
                pass
            try:
                solconstr = root(foccs_constrained, [max(0,solsol.x[0]), solsol.x[1], solsol.x[2],0], [xx[0],xx[1],kkt,t], method='hybr')
            except:
                pass
            x_p = xx[1]
        elif (interp_strategy == 'standard'):
            (x_point, a_point) = divmod(i, m)
            x_p = x_grid[x_point]
            a_p = a_grid[a_point]
            a_prime = a_func(a_p, x_p)
            thetas = [np.asscalar(a_prime)*0.8, 2.0, 0.72, 0]
            solsol = root(foccs, thetas, [a_p, x_p, t], method='lm')

        if solsol.success == False:
            print('***')
            print('error at point', i)
            print("but solution is", solsol.x[0],solsol.x[1],solsol.x[2])
            print('***')
            if (interp_strategy == 'gpr'):
                try:
                    solsol = root(foccs, [xx[0]*0.5, 0.99, 0.7], [xx[0]*0.6, xx[1]*0.9, t], method='lm') # [xx[0]*0.5, 0.99, 0.7, 0]
                except:
                    pass
                x_p = xx[1]
            elif (interp_strategy == 'standard'):
                (x_point, a_point) = divmod(i, m)
                thetas = [0.25*a_p + 0.05, 4.2, 0.72] #[0.25*a_p + 0.05, 4.2, 0.72, 0]
                solsol = root(foccs, thetas, [a_p, x_p, t], method='lm')

            if solsol.success == False:
                print('***')
                print('again an error at point', i)
                print("with solution is", solsol.x[0], solsol.x[1], solsol.x[2])
                print('***')
                status_collect[i, t - 1] = 0
            else:
                print("ok solution is", solsol.x[0], solsol.x[1], solsol.x[2])
                status_collect[i, t - 1] = 1
        else:
            print("ok solution is", solsol.x[0],solsol.x[1],solsol.x[2])
            print("constrained solution is", solconstr.x[0],solconstr.x[1],solconstr.x[2],solconstr.x[3])
            status_collect[i,t-1] = 1
        a_star_new[i, t - 1] = solsol.x[0]

        if solconstr.x[3] > 0:
            constraint[i, t - 1] = 1
            theta_star_new[i, t - 1] = solconstr.x[1]
            y_star_new[i, t - 1] = solconstr.x[2]
            wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)
        else:
            constraint[i, t - 1] = 0
            theta_star_new[i, t - 1] = solsol.x[1]
            y_star_new[i, t - 1] = solsol.x[2]
            wage_star_new[i, t - 1] = wage(y_star_new[i, t - 1], theta_star_new[i, t - 1], x_p, t)

        print('and wage is', wage_star_new[i, t - 1])

    controls_unemp = np.where(status_collect[:, t - 1] < 1.0)


    empl_grid_b[:, 2] = wage_star_new[:, t - 1]  # [:,0]
    empl_grid_b[:, 3] = y_star_new[:, t - 1]  # [:,0]
    empl_grid_b_old = empl_grid_b

    if (interp_strategy == 'gpr'):

        unempl_grid_old = unempl_grid
        empl_grid_old = empl_grid

    elif (interp_strategy == 'standard'):
        unempl_grid_old = unempl_grid
        empl_grid_old = empl_grid


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
        #print("value", i, "is", g)
        if (interp_strategy == 'gpr'):
            a_prime = np.asscalar(a_emp_func.predict(np.atleast_2d([g[0],g[1],g[2],g[3]]), return_std=False))
        elif (interp_strategy == 'standard'):
            a_prime = a_emp_func(g[0], g[1], g[2], g[3])
        otherr  = [g[0],g[1],g[2],g[3],constraint[i,t-1]]
        sol = root(foc_empl, g[2], otherr, method='hybr') # or hybr?
        a_star_emp_new_b[i,t-1] = sol.x
        if sol.success == False:
            print('***')
            print('error in employed b problem at', g)
            print("but solution is", sol.x)
            print('***')
            status_collect[i,t-1] = 0
        else:
            status_collect[i, t - 1] = 1
            #print("ok solution is", sol.x)

    controls_emp = np.where( status_collect[:,t-1] < 1.0 )
    joined =  controls_unemp + controls_emp
    joint = [x for xs in joined for x in xs]
    print('joint',joint)



##################### VALUE FUNCTIONS ##########################################


    for i,k in enumerate(unempl_grid):
        a_prime_emp = a_star_emp_new_b[i,t-1]
        a_prime = a_star_new[i,t-1]
        w = np.asscalar(wage_star_new[i, t - 1])
        y = np.asscalar(y_star_new[i, t - 1])
        if (interp_strategy == 'gpr'):
            a = k[0]
            x = k[1]
            a_emp_int = np.asscalar(
                a_emp_func.predict(np.atleast_2d([a_prime_emp, g_fun(y, x), w, y]), return_std=False))
            if (constraint[i, t - 1] == 0):
                a_int = max(np.asscalar(a_func.predict(np.atleast_2d([a_star_new[i,t-1], empl_grid[i,1]]), return_std=False)),0)
            else:
                a_int = 0
        elif (interp_strategy == 'standard'):
            (x_point, a_point) = divmod(i, m)
            x = x_grid[x_point]
            a = a_grid[a_point]
        if t>1:
            if (constraint[i, t - 1] == 0):
                U_new[i,t-1] = U_fun(U_new[i,t-2],E_new_b[i,t-2], a, a_star_new[i,t-1],theta_star_old[i]) # a_star_new[i,t-1],a_int
                E_new_b[i,t-1] = E_fun(U_new[i,t-2],E_new_b[i,t-2], a, a_star_emp_new_b[i,t-1], w) # a_star_emp_new_b[i,t-1],a_emp_int
                E_prime_new_b[i,t-1] = E_funprime(E_prime_new_b[i,t-2], a, a_star_emp_new_b[i,t-1], w) # a_star_emp_new_b[i,t-1],a_emp_int
            else:
                U_new[i,t-1] = U_fun(U_new[i,t-2],E_new_b[i,t-2], a, 0,theta_star_old[i]) # a_star_new[i,t-1],a_int
                E_new_b[i,t-1] = E_fun(U_new[i,t-2],E_new_b[i,t-2], a, a_star_emp_new_b[i,t-1], w) # a_star_emp_new_b[i,t-1],a_emp_int
                E_prime_new_b[i,t-1] = E_funprime(E_prime_new_b[i,t-2], a, a_star_emp_new_b[i,t-1], w) # a_star_emp_new_b[i,t-1],a_emp_int
        else:
            if (constraint[i, t - 1] == 0):
                U_new[i, t - 1] = U_fun(0, 0, a, a_star_new[i,t-1], theta_star_old[i]) # a_star_new[i,t-1], a_int,
                E_new_b[i, t - 1] = E_fun(0, 0, a, a_star_emp_new_b[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int,
                E_prime_new_b[i, t - 1] = E_funprime(0, a, a_star_emp_new_b[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int,
            else:
                U_new[i, t - 1] = U_fun(0, 0, a, 0, theta_star_old[i]) # a_star_new[i,t-1], a_int,
                E_new_b[i, t - 1] = E_fun(0, 0, a, a_star_emp_new_b[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int,
                E_prime_new_b[i, t - 1] = E_funprime(0, a, a_star_emp_new_b[i, t - 1], w) # a_star_emp_new_b[i, t - 1], a_emp_int,


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


    #E_old = E_new[:, t - 1]
    E_old_b = E_new_b[:, t - 1]
    #E_prime_old = E_prime_new[:, t - 1]
    E_prime_old_b = E_prime_new_b[:, t - 1]
    #a_star_emp_old = a_star_emp_new[:, t - 1]
    a_star_emp_old_b = a_star_emp_new_b[:, t - 1]
    U_old = U_new[:, t - 1]
    a_star_old = a_star_new[:, t - 1]
    theta_star_old = theta_star_new[:, t - 1]
    y_star_old = y_star_new[:, t - 1]
    wage_star_old = wage_star_new[:, t - 1]



################ FIXING MATRICES AND INTERPOLATING #################################

    if (interp_strategy == 'gpr'):
        #a_star_old = a_star_new[:,t-1]
        if any(joint):  # [0]
            print("status collector is not empty:")
            print('we have to change sizes')
            unempl_grid_old = np.delete(unempl_grid_old, joint, axis=0)
            #empl_grid_old = np.delete(empl_grid_old, joint, axis=0)
            empl_grid_b_old = np.delete(empl_grid_b_old, joint, axis=0)
            a_star_old = np.delete(a_star_old, joint, axis=0)
            #a_star_emp_old = np.delete(a_star_emp_old, joint, axis=0)
            a_star_emp_old_b = np.delete(a_star_emp_old_b, joint, axis=0)
            wage_star_old = np.delete(wage_star_old, joint, axis=0)
            #theta_star_old = np.delete(theta_star_old, joint, axis=0)
            y_star_old = np.delete(y_star_old, joint, axis=0)
            U_old = np.delete(U_old, joint, axis=0)
            #E_old = np.delete(E_old, joint, axis=0)
            E_old_b = np.delete(E_old_b, joint, axis=0)
            #E_prime_old = np.delete(E_prime_old, joint, axis=0)
            E_prime_old_b = np.delete(E_prime_old_b, joint, axis=0)
            rsize_old = unempl_grid_old.shape[0]
            print('new row size is', rsize_old)
        else:
            print('rsize does not change')

        #empl_grid_together = np.concatenate((empl_grid_old, empl_grid_b_old), axis=0)
        #E_old_together = np.concatenate((E_old, E_old_b), axis=0)
        #a_star_emp_old_together = np.concatenate((a_star_emp_old, a_star_emp_old_b), axis=0)
        #E_prime_old_together = np.concatenate((E_prime_old, E_prime_old_b), axis=0)

        kernel_E = 1.0 * RBF(7.0, (1e-3, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e+1)) #C(1.0, (1e-3, 1e-3))

        U_func = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
        a_func = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)
        E_func = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=10)
        Eprime_func = GaussianProcessRegressor(kernel=kernel_bis, n_restarts_optimizer=10)
        a_emp_func = GaussianProcessRegressor(kernel=kernel_E, n_restarts_optimizer=10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # if you get useless warnings print "ignore", otherwise "always"
            U_func.fit(unempl_grid_old, U_old)
            E_func.fit(empl_grid_b_old, E_old_b)
            a_func.fit(unempl_grid_old, a_star_old)
            a_emp_func.fit(empl_grid_b_old, a_star_emp_old_b)
            Eprime_func.fit(empl_grid_b_old, E_prime_old_b)


    elif (interp_strategy == 'standard'):
        a_star_old = a_star_old.reshape((m_x, m, 1))
        U_old = U_old.reshape((m_x, m, 1))
        if any(joint):
            con_work = joint[0]
            cont_size = con_work.shape
            print('shape of controls', cont_size)
            for i,j in enumerate(joint):
                (x_nan,a_nan) = divmod(j,m)
                AA[x_nan,a_nan] = np.PINF

            array = np.ma.masked_invalid(AA)
            newAA = AA[~array.mask]
            newWW = WW[~array.mask]
            newarr = a_star_old[~array.mask]
            newUrr = U_old[~array.mask]
            a_func = lambda q, h: interpolate.griddata((newAA,newWW),newarr.ravel(),(q,h),method='linear')
            U_func = lambda q, h: interpolate.griddata((newAA, newWW), newUrr.ravel(), (q, h), method='linear')


        else:
            #a_func = lambda q, h: interpn((a_grid, x_grid), a_star_old, (q, h), bounds_error=False, fill_value=None)  #
            #U_func = lambda q, h: interpn((a_grid, x_grid), U_old, (q, h), bounds_error=False, fill_value=None)  #
            a_star_old = a_star_old.reshape((m_x, m, 1))
            U_old = a_star_old.reshape((m_x, m, 1))
            array = np.ma.masked_invalid(AA)
            newAA = AA[~array.mask]
            newWW = WW[~array.mask]
            newarr = a_star_old[~array.mask]
            newUrr = U_old[~array.mask]
            a_func = lambda q, h: interpolate.griddata((newAA, newWW), newarr.ravel(), (q, h), method='linear')
            U_func = lambda q, h: interpolate.griddata((newAA, newWW), newUrr.ravel(), (q, h), method='linear')


        a_emp_func = interpolate.Rbf(empl_grid[:, 0], empl_grid[:, 1], empl_grid[:, 2], empl_grid[:, 3], a_star_emp_old)
        E_func = interpolate.Rbf(empl_grid[:, 0], empl_grid[:, 1], empl_grid[:, 2], empl_grid[:, 3], E_old)
        Eprime_func = interpolate.Rbf(empl_grid[:, 0], empl_grid[:, 1], empl_grid[:, 2], empl_grid[:, 3], E_prime_old)

    a_u_try = np.empty_like(a_grid)
    a_u_highprod = np.empty_like(a_grid)
    a_e_try = np.empty_like(a_grid)
    E_e_try = np.empty_like(a_grid)
    U_u_try = np.empty_like(a_grid)



    for i, aval in enumerate(a_grid):
        if (constraint[i, t - 1] == 0):
            a_u_try[i] = max(np.asscalar(a_func.predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False)),0)
        else:
            a_u_try[i] = max(np.asscalar(a_func.predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False)),0)
        a_e_try[i] = np.asscalar(
            a_emp_func.predict(np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                               return_std=False))
        U_u_try[i] = np.asscalar(U_func.predict(np.atleast_2d([aval, (high_x + low_x) / 2]), return_std=False))
        E_e_try[i] = E_func.predict(np.atleast_2d([aval, (high_x + low_x) / 2, (low_w + high_w) / 2, (low_y + high_y) / 2]),
                               return_std=False)



    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, a_u_try, 'r', lw=3, zorder=9, label='unemployed (interpolation)')
    plt.scatter(unempl_grid_old[:, 0], a_star_old,s=50, zorder=10, edgecolors=(0, 0, 0))  # , s=area, c=colors, alpha=0.5)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, U_u_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(unempl_grid_old[:,0], U_old)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, E_e_try, 'r', lw=3, zorder=9, label='VF (interpolation)')
    plt.scatter(empl_grid_b_old[:, 0], E_old_b)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, a_e_try, 'r', lw=3, zorder=9, label='employed (interpolation)')
    plt.scatter(empl_grid_b_old[:, 0], a_star_emp_old_b, s=50, zorder=10, edgecolors=(0, 0, 0))

    plt.show()


    fig, ax = plt.subplots(figsize=(9, 5))
    plt.scatter(unempl_grid_old[:,0], wage_star_old)
    plt.show()


