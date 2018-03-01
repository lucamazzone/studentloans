import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy.matlib
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.optimize import broyden2
from scipy.optimize import broyden1
from scipy.interpolate import CubicSpline
from scipy.optimize import newton_krylov


class First_Extension:
    def __init__(self, A = 0.7, alpha=0.2, beta=0.99, eta = 4.5, gamma = 1.5, b = 0.0, nu= 1.0 , lamb = 0.03, R = 1.05, kappa = 10):
        # type: (object, object, object, object, object, object) -> object
        self.A, self.alpha, self.beta, self.gamma, self.b, self.nu,\
        self.lamb, self.R, self.eta, self.kappa = A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa


    def m_fun(self, theta):
        return (1 - np.exp(- self.eta * theta))

    def m_funprime(self, theta):
        return self.eta * np.exp(-self.eta * theta)


    def firmfoc_inf(self,y,theta):
        gtheta  = ((1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2)
        return beta * q_fun(theta) * f_prime(A, y)  - V_funprime(y)*gtheta

    def q_fun(self, theta):
        return (1 - np.exp(-self.eta * theta)) / (theta)

    def q_funprime(self, theta):
        return -(1 / (theta ** 2)) + (self.eta * np.exp(-self.eta * theta)) / (theta )

    def u(self, c):
        if (self.nu == 1):
            return np.log(c)
        else:
            return (c ** (1 - self.nu)) / (1 - self.nu)

    def uprime_inv(self, c):
        return c ** (-1 / self.nu)

    def u_prime(self, c):
        return c ** (-self.nu)

    def f(self, A, y):
        return A * y ** self.alpha

    def f_prime(self, A, y):
        return A * self.alpha * y ** (self.alpha - 1)

    def V_fun(self, y):
        return 0.15*y**(1/self.gamma)  #(1.5 + y ** (self.gamma)) ** (1 / self.gamma)

    def V_funprime(self, y):
        return  0.15*(1/self.gamma)*y**(1/self.gamma-1) #((1.5 + y ** (self.gamma)) ** ((1/self.gamma) - 1)) * (y ** (self.gamma - 1))

    def wage(self,theta, y):
        return f(A, y) - kappa * (1 - beta * (1 - lamb)) / (beta * q_fun(theta)) - V_fun(y) * (
            (1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2) / (
                                                                                   beta * q_fun(theta))

    def foccs(self,thetas):
        a_prime = thetas[0]
        theta = thetas[1]
        y = fsolve(firmfoc_inf, 207.2, theta)
        w = f(A, y) - kappa * (1 - beta * (1 - lamb)) / (beta * q_fun(theta)) - V_fun(y) * (
            (1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2) / (
                                                                                beta * q_fun(theta))
        c_u = b + R * a - a_prime
        c_ut = b + R * a_prime - a_func(a_prime)
        c_e = R * a - a_prime + w
        c_et = R * a_prime - a_emp_func(a_prime) + w
        dw = V_fun(y) * (1 - beta) * (1 - beta * (1 - lamb)) * q_funprime(theta) / (q_fun(theta) ** 2)
        EE = E_fun(U_guess_int, E_guess_int, a_prime, a_emp_func(a_prime), w)
        UU = U_fun(U_guess_int, E_guess_int, a_prime, a_func(a_prime), theta)
        focc1 = -u_prime(c_u) + beta * R * (m_fun(theta) * u_prime(c_et) + (1 - m_fun(theta)) * u_prime(c_ut))
        focc2 = m_fun(theta) * u_prime(c_et) * dw + m_funprime(theta) * (EE - UU)  # u(c_et) - u(c_ut)

        return np.array([focc1, focc2])

    def E_fun(self,U, E, a,a_prime,w):
        c_e = R * a - a_prime + w
        return u(c_e) + beta*((1-lamb)*E(a) + lamb*U(a))

    def U_fun(self, U, E, a, a_prime, theta):
        c_u = b + R * a - a_prime
        return u(c_u) + beta*(m_fun(theta)*E(a) + (1-m_fun(theta))*U(a))


######################################################################################################

gg = First_Extension()
A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa = gg.A, gg.alpha, gg.beta, gg.gamma, gg.b, gg.nu, gg.lamb, gg.R, gg.eta, gg.kappa
u, u_prime, uprime_inv,m_fun,m_funprime, firmfoc_inf, q_fun, q_funprime, f,\
f_prime, V_fun, V_funprime, wage, foccs, E_fun, U_fun = gg.u, gg.u_prime, gg.uprime_inv, gg.m_fun, gg.m_funprime, gg.firmfoc_inf,\
                    gg.q_fun, gg.q_funprime, gg.f , gg.f_prime, gg.V_fun, gg.V_funprime,gg.wage,gg.foccs, gg.E_fun, gg.U_fun



low_a = 1
high_a = 70
m = 35  # number of nodes

a_grid = np.linspace(low_a,high_a,num=m)


E_guess = np.zeros_like(a_grid)
U_guess = np.zeros_like(a_grid)
E = np.empty_like(a_grid)
U = np.empty_like(a_grid)
E_guess_int = CubicSpline(a_grid,E_guess)
U_guess_int = CubicSpline(a_grid,U_guess)




a = a_grid[13]
a_func = CubicSpline(a_grid, 0.7*a_grid)
a_emp_func = CubicSpline(a_grid, 1.001*a_grid)


theta = 0.8
a_prime = 0.825*a
y = fsolve(firmfoc_inf, 207.2, theta)
print(y)
w = f(A, y) - kappa * (1 - beta * (1 - lamb)) / (beta * q_fun(theta)) - V_fun(y) * (
            (1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2) / (
                                                                                beta * q_fun(theta))
c_u = b + R * a - a_prime
c_ut = b + R * a_prime - a_func(a_prime)
c_e = R * a - a_prime + w
c_et = R * a_prime - a_emp_func(a_prime) + w
dw = V_fun(y) * (1 - beta) * (1 - beta * (1 - lamb)) * q_funprime(theta) / (q_fun(theta) ** 2)
EE = E_fun(U_guess_int, E_guess_int, a_prime, a_emp_func(a_prime), w)
UU = U_fun(U_guess_int, E_guess_int, a_prime, a_func(a_prime), theta)
focc2 = m_fun(theta) * u_prime(c_et) * dw + m_funprime(theta) * (EE - UU)  # u(c_et) - u(c_ut)


#print(fsolve(foccs,theta))


print(w)
#print(fsolve(foccs,a))

try:
    print(newton_krylov(  foccs,[0.75*a,0.87],method='lgmres', verbose=0))
except:
    print(broyden2(foccs, [0.75 * a, 0.87]))

