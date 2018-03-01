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
from scipy.interpolate import interp1d


#####################################################################################

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


#############################################################################################


class default_model:

    def __init__(self,beta=0.92,gamma=2,r = 0.02):
        self.beta,self.gamma,self.r = beta, gamma,r

    def u(self,c):
        return (c**(1-gamma))/(1-gamma)

    def u_prime(self,c):
        return c**(-gamma)

    def q_fun(self,P):
        return (P)/(1+r)

#############################################################################################

def coleman_operator(b_guess,P_guess,j):
    b_pol = np.empty_like(b_grid)

    # === Apply interpolation to guess === #
    #borr_func = interp1d(b_grid, b_guess[0, :], kind='cubic')
    #P_func = interp1d(b_grid, P_guess[0, :], kind='linear')
    # === Apply polynomial interpolation to guess === #
    poly = np.polyfit(b_grid, b_guess[j, :], deg=1)
    borr_func = lambda x: np.polyval(poly, x)  # np.polynomial.chebyshev.chebval(x, coeffs)
    poly2 = np.polyfit(b_grid, P_guess[j, :], deg=1)
    P_func = lambda x: np.polyval(poly2, x)  # np.polynomial.chebyshev.chebval(x, coeffs)

    for i, b in enumerate(b_grid):
        def h(b_tilde):
            cont_value = 0.0
            c = GDP[j] + b - b_tilde * q_fun(P_func(b_tilde))  # b_grid[20]*q_fun(P_func(b_grid[20]))
            for y, Y in enumerate(GDP):
                c_prime = Y + b_tilde - borr_func(b_tilde) * q_fun(P_func(borr_func(b_tilde)))
                cont_value = cont_value + P_func(b_tilde) * u_prime(c_prime) * GDP_Prob[j][y]
            return u_prime(c) * P_func(b)*q_fun(P_func(b_tilde)) - beta * cont_value

        b_pols = fsolve(h, b_grid[i])
        b_pol[i] = b_pols

    return b_pol

#############################################################################################

def P_fun(b_guess,P_guess,j):
    p_def = np.empty_like(b_grid)
    poly2 = np.polyfit(b_grid, P_guess[j, :], deg=1)
    P_func = lambda x: np.polyval(poly2, x)  # np.polynomial.chebyshev.chebval(x, coeffs)

    for i,b in enumerate(b_grid):
        def hh(p_tilde):
            c = GDP[j] + b - b_guess[j,i] * q_fun(P_func(b_guess[j,i]))
            c_def = min(GDP[j], theta * np.mean(GDP))
            cont_value = 0.0
            for y, Y in enumerate(GDP):
                poly3 = np.polyfit(b_grid, P_guess[y, :], deg=1)
                P_funcc = lambda x: np.polyval(poly3, x)  # np.polynomial.chebyshev.chebval(x, coeffs)
                poly = np.polyfit(b_grid, b_guess[y,: ], deg=1)
                borr_func = lambda x: np.polyval(poly, x)  # np.polynomial.chebyshev.chebval(x, coeffs)
                c_prime = Y + b_star[i] - borr_func(b_star[i]) * q_fun(P_funcc(borr_func(b_star[i])))
                c_primedef = min(Y, theta * np.mean(GDP))
                iaoo = sigmaepsilon * np.log(1-P_funcc(borr_func(b_star[i])))
                cont_value = cont_value - beta * (iaoo) * GDP_Prob[j][y]

            return u(c) - u(c_def) - sigmaepsilon*(np.log(p_tilde) - np.log(1-p_tilde)) + cont_value

        p_def[i] = fsolve(hh,P_guess[j,i])

    return p_def



#############################################################################################
####################################  MODEL SOLUTION  #######################################
#############################################################################################

# parameters governing debt grid
b_num= 20  #2000
b_num_interm=100
b_inf = -4.0  #-3.30
b_sup =  -1.05
# gdp grid
rho = 0.945
mu = 0.0
step = 0.02
num = 7
GDP_Prob,GDP_vec = rouwen(rho,mu,step,num)
GDP = 10*np.exp(GDP_vec)

print(GDP)
theta = 0.75
sigmaepsilon = 0.25

gg = default_model()
beta, gamma, r = \
    gg.beta, gg.gamma, gg.r

u, u_prime, q_fun = gg.u, gg.u_prime, gg.q_fun




b_grid = np.linspace(b_inf, b_sup, num=b_num)
b_guess = b_grid + 0*GDP_vec[:, np.newaxis]
P_guess = np.linspace(0.5,0.9,num=b_num)
P_guess = P_guess + 0*GDP_vec[:, np.newaxis]




print(P_guess)

for j,Y in enumerate(GDP):
    pol_error = 5.0
    print(j)
    while (pol_error > 1e-6):
        b_star = coleman_operator(b_guess,P_guess,j)
        pol_error = np.max(abs(b_guess[j,:]-b_star))
        b_guess[j, :] = b_star
        #P_guess[j, :] = P_fun(b_star,P_guess,j)

for j in range(num):
    print(j)
    P_guess[j,:] = P_fun(b_guess,P_guess,j)


print(b_guess)


for j,Y in enumerate(GDP):
    pol_error = 5.0
    print(j)
    while (pol_error > 1e-6):
        b_star = coleman_operator(b_guess,P_guess,j)
        pol_error = np.max(abs(b_guess[j,:]-b_star))
        b_guess[j, :] = b_star
        #P_guess[j, :] = P_fun(b_star,P_guess,j)

print(P_guess)
print(b_guess)

for j in range(num):
    P_guess[j,:] = P_fun(b_guess,P_guess,j)




for j,Y in enumerate(GDP):
    pol_error = 5.0
    print(j)
    while (pol_error > 1e-6):
        b_star = coleman_operator(b_guess,P_guess,j)
        pol_error = np.max(abs(b_guess[j,:]-b_star))
        b_guess[j, :] = b_star
        #P_guess[j, :] = P_fun(b_star,P_guess,j)

print(P_guess)
print(b_guess)

for j in range(num):
    P_guess[j,:] = P_fun(b_guess,P_guess,j)




for j,Y in enumerate(GDP):
    pol_error = 5.0
    print(j)
    while (pol_error > 1e-6):
        b_star = coleman_operator(b_guess,P_guess,j)
        pol_error = np.max(abs(b_guess[j,:]-b_star))
        b_guess[j, :] = b_star
        #P_guess[j, :] = P_fun(b_star,P_guess,j)

print(P_guess)
print(b_guess)

for j in range(num):
    P_guess[j,:] = P_fun(b_guess,P_guess,j)



for j,Y in enumerate(GDP):
    pol_error = 5.0
    print(j)
    while (pol_error > 1e-6):
        b_star = coleman_operator(b_guess,P_guess,j)
        pol_error = np.max(abs(b_guess[j,:]-b_star))
        b_guess[j, :] = b_star
        #P_guess[j, :] = P_fun(b_star,P_guess,j)

print(P_guess)
print(b_guess)
