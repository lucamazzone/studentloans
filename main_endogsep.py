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
import quantecon as qe


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

def k_fun(y):
    return kappa*(y**gamma)/gamma

def k_funprime(y):
    return kappa*(y**(gamma-1))

def c_u(a, a_prime):
    return b + R * a - a_prime

def c_e(w, a, a_prime):
    return w + R * a - a_prime

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


def wage_high(y, theta, x, tau):
    cosa1 = (beta * (1 - lamb))
    add_year = 0
    new_x = x
    for i in range(tau-1):
        cosa1 = cosa1 +  (beta * (1 - lamb)) ** (i)
        new_x = g_fun(y, new_x)
        add_year = add_year + (f(A, y, new_x) - (max(adj * y - g_fun(y, new_x), 0))*train_cost)*((beta * (1 - lamb)) ** (i))


    return  (f(A, y, x) - (max(adj*y - x,0))*train_cost + add_year)/(cosa1) -  k_fun(y)/(cosa1*beta*q_fun(theta))
        #



def wage_low(y, theta, x, tau):
    cosa1 = 0
    for i in range(tau):
        cosa1 = cosa1 +  (beta * (1 - lamb)) ** (i)
    return  f(A, y, x)  -  (max(adj*y - g_fun(y,x),0))*train_cost - k_fun(y)/(cosa1*beta*q_fun(theta))


def f( A, y, x):
    return A * (y**alpha  + g_fun(y, x)**alpha )**(1/alpha) #A * (y**alpha)*g_fun(y, x)**(1-alpha)

def f_prime(A, y, x):  # rename as f_y
    return  y**(alpha-1)*A*(y**alpha  + g_fun(y, x)**alpha )**(1/alpha-1)  +\
            g_fun_prime(y,x)*g_fun(y, x)**(alpha-1)*A*(y**alpha  + g_fun(y, x)**alpha )**(1/alpha-1)  #A * (alpha * y ** (alpha - 1)*g_fun(y, x)**(1-alpha) +
            #    (1-alpha)*g_fun_prime(y, x)*(y**alpha)*g_fun(y, x)**(-alpha))

#####################################################################################################################

def E_fun(U, E, EU, a, a_prime, w,theta, E_tilde, t_f): #
    return u(c_e(w, a, a_prime)) + beta*(1-pi*m_fun(theta))*((1-lamb)*E + lamb*U) + \
           beta*pi*m_fun(theta)*(t_f*EU + (1-t_f)*E_tilde )


def E_funprime(E_prime, a, a_prime, theta, w):
    return u_prime(c_e(w, a, a_prime)) + beta*(1-lamb)*(1-pi*m_fun(theta))*E_prime   #E_prime function of a_prime

def U_fun(U, E, EU, a, a_prime, theta,t_f):
    return u(c_u(a, a_prime)) + beta*(m_fun(theta)*(t_f*EU + (1-t_f)*E) + (1-m_fun(theta))*U)

def EU_fun(U, E, EU, a, a_prime, theta,t_f,w):
    return u(c_e(w,a, a_prime)) + beta*(m_fun(theta)*(t_f*EU + (1-t_f)*E) + (1-m_fun(theta))*U)

#####################################################################################################################

x = 3.3
y = 0.9
theta = 1.5
control = np.zeros((15))
control[13] = 1

for i in range(1,15):
    if (control[i]==1):
        #print('exit')
        tau = i
        break


#print(wage_high(y, theta, x, tau))
#print(wage_low(y, theta, x, tau))

exp_u_low_y_p =  u( 1.0 + (0.5*wage_high(y, theta, x, tau) + 0.5*wage_low(y, theta, x, tau)) )
print("poor guy",exp_u_low_y_p)
exp_u_low_y_r = u( 5.0 + (0.5*wage_high(y, theta, x, tau) + 0.5*wage_low(y, theta, x, tau)) )
print("rich guy",exp_u_low_y_r )

y_gridd = np.linspace(0.8,1.9,100)
exp_u_high_y_p = np.empty((100,1))
exp_u_high_y_r = np.empty((100,1))

for i,y in enumerate(y_gridd):
    exp_u_high_y_p[i] =  u( 5.0 + ((1-t_fun(y,x))*wage_high(y, theta*1.1, x, tau) + t_fun(y,x)*wage_low(y, theta*1.1, x, tau)) )
    exp_u_high_y_r[i] = u( 5.0  + ((1-t_fun(y,x))*wage_high(y, theta, x, tau) +  t_fun(y,x)*wage_low(y, theta, x, tau)) )


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(y_gridd,exp_u_high_y_p, 'r', lw=3, zorder=9, label='high net assets')
ax.plot(y_gridd,exp_u_high_y_r, 'b', lw=3, zorder=9, label='low net assets')
plt.title('mean wage')
plt.show()

print([i for i, j in enumerate(exp_u_high_y_p) if j == max(exp_u_high_y_p)])
print([i for i, j in enumerate(exp_u_high_y_r) if j == max(exp_u_high_y_r)])

print([y_gridd[i] for i, j in enumerate(exp_u_high_y_r) if j == max(exp_u_high_y_r)])
print([t_fun(y_gridd[i],x) for i, j in enumerate(exp_u_high_y_r) if j == max(exp_u_high_y_r)])



####### REMEMBER TO USE NUMBA
'''
qe.util.tic()
k_fun_numba = jit(k_fun)
print(k_fun_numba(y))
time1 = qe.util.toc()

qe.util.tic()
print(k_fun(y))
time2 = qe.util.toc()
'''
######

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

