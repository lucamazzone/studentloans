import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy.matlib
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import broyden2
from scipy.interpolate import griddata
from scipy.interpolate import interpn


############################################################################################

def rouwen(rho, mu, step, num):
    #'''
    #Adapted from Lu Zhang and Karen Kopecky. Python by Ben Tengelsen.
    #Construct transition probability matrix for discretizing an AR(1)
    #process. This procedure is from Rouwenhorst (1995), which works
    #well for very persistent processes.

    #INPUTS:
    #rho  - persistence (close to one)
    #mu   - mean and the middle point of the discrete state space
    #step - step size of the even-spaced grid
    #num  - number of grid points on the discretized process

    #OUTPUT:
    #dscSp  - discrete state space (num by 1 vector)
    #transP - transition probability matrix over the grid
    #'''

    # discrete state space
    dscSp = np.linspace(mu -(num-1)/2*step, mu +(num-1)/2*step, num).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2], \
                    [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)], \
                    [(1-p)**2, (1-p)*q, q**2]]).T

    while transP.shape[0] <= num - 1:

        # see Rouwenhorst 1995
        len_P = transP.shape[0]
        transP = p*np.vstack((np.hstack((transP,np.zeros((len_P,1)))), np.zeros((1, len_P+1)))) \
        + (1-p)*np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
        + (1-q)*np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
        + q*np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.

    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP, dscSp


######################################################################################################


class OLG_model:

    def __init__(self,alpha=0.33,beta=0.98,gamma=2.0,phi=1.0,nu=2.0,delta=0.05):
        self.alpha,self.beta,self.gamma,self.phi,self.nu,self.delta=alpha,beta,gamma,phi,nu,delta

    def f_kprime(self, A, k, l):
        return A * self.alpha * (k ** (self.alpha - 1)) * l ** (1 - self.alpha)


    def f_lprime(self, A, k,l):
        return A * (1 - self.alpha) * (k ** self.alpha) * l ** (-self.alpha)

    def c_1(l, w, theta):
        return l * w - theta

    def c_i(l, w, theta, s1, s, r):
        return l * w - theta + s1 * s * r

    def c_N(l, w, s1, s, r):
        return l * w + s1 * s * r

    def u(self,c):
        return (c**(1-self.gamma))/(1-self.gamma)

    def u_prime(self,c):
        return c**(-self.gamma)

    def f(self,A,k,l):
        return A*(k**self.alpha)*(l**(1-self.alpha))


######################################################################################################

def chebyshev(x,n,t=None):
    t = np.zeros([n + 1])

    t[0] = 1;
    t[1] = x;

    for i in range(2,n+1):
        t[i] = 2 * x * t[i - 1] - t[i - 2]
    return t


######################################################################################################


rho_prod = 0.95
mu_prod = 1.0
step_prod = 0.025
num_prod = 5
Prob,Prod_vec = rouwen(rho_prod,mu_prod,step_prod,num_prod)


gg = OLG_model()
alpha,beta,gamma,phi,nu,delta = gg.alpha,gg.beta,gg.gamma,gg.phi,gg.nu,gg.delta
u, u_prime, f, f_kprime,f_lprime, c_1, c_i , c_N= gg.u, gg.u_prime, gg.f, gg.f_kprime,gg.f_lprime, gg.c_1, gg.c_i,gg.c_N


# model data
l1 = 0.5
l2 = 14.5
l3 = 0.5
L = l1+l2+l3
A = 1.5


# Cheby nodes
m = 9  # number of nodes
pi = 3.1416
M = (2*np.linspace(1.0, m, num=m)-1)/(2*m)*pi
Kstar = ((1/beta + delta - 1)*(1/alpha))**(1/(alpha-1))
a = 0.25*Kstar
b = 11*a
z = np.empty([m])
for i in range(m):
    cosm = math.cos(M[i])
    z[i] = -cosm
x = (z + 1)*(b-a)/2 + a
y = (z + 1)/2


k_guess = y + y[:, np.newaxis]
k_star = np.empty_like(k_guess)
k_pol = np.empty_like(k_guess)



th1_func = lambda bum: interpn((x,y), k_guess, bum, method='linear', bounds_error=False)
th2_func = lambda bum: interpn((x,y), k_guess*0.85, bum, method='linear', bounds_error=False)




theta1 = 1.89104967
theta2 = 1.11318524
s1 = x[5]
s = y[5]


r = (f_kprime( A, s1, L)+(1-delta))
c1 = l1*f_lprime(A, s1,L) - theta1
c2 =  s1*s*r + l2*f_lprime(A, s1,L) - theta2
#c3 = s1*(1-s)*r + l3*f_lprime(A,s1,L)

s1_f = theta1+theta2
s_f = theta1/(theta1+theta2)

r_f = (f_kprime( A, s1_f, L)+(1-delta))
#c1_f = l1*f_lprime(A, s1_f,L) - th1_func(np.array([s1_f,s_f]))
c2_f =  s1_f*s_f*r_f + l2*f_lprime(A, s1_f,L) - np.asscalar(th2_func(np.array([s1_f,s_f])))
c3_f = s1_f*(1-s_f)*r_f + l3*f_lprime(A,s1_f,L)

euler_eq_1 = u_prime(c1) - beta*r_f*u_prime(c2_f)
euler_eq_2 = u_prime(c2) - beta*r_f*u_prime(c3_f)

print(c1)

def euler_eq(thetas):
    theta1 = thetas[0]
    theta2 = thetas[1]

    r = (f_kprime(A, s1, L) + (1 - delta))
    c1 = l1 * f_lprime(A, s1, L) - theta1
    c2 = s1 * s * r + l2 * f_lprime(A, s1, L) - theta2
    # c3 = s1*(1-s)*r + l3*f_lprime(A,s1,L)

    s1_f = theta1 + theta2
    s_f = theta1 / (theta1 + theta2)

    r_f = (f_kprime(A, s1_f, L) + (1 - delta))
    # c1_f = l1*f_lprime(A, s1_f,L) - th1_func(np.array([s1_f,s_f]))
    c2_f = s1_f * s_f * r_f + l2 * f_lprime(A, s1_f, L) - np.asscalar(th2_func(np.array([s1_f, s_f])))
    c3_f = s1_f * (1 - s_f) * r_f + l3 * f_lprime(A, s1_f, L)

    euler_eq_1 = u_prime(c1) - beta * r_f * u_prime(c2_f)
    euler_eq_2 = u_prime(c2) - beta * r_f * u_prime(c3_f)

    return np.array([euler_eq_1,euler_eq_2])


#print(euler_eq(np.array([5.87451,5.442])))
solv = broyden2(euler_eq, [9.1,8.25], f_tol=1e-2)

print(solv)
