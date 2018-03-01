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


######################################################################################################

class Homogeneous_agents:
    def __init__(self, A = 1.25, alpha=0.22, beta=0.99, eta = 5.5, gamma = 1.5, b = 0.0, nu= 1.0 , lamb = 0.03, R = 1.05, kappa = 10):
        # type: (object, object, object, object, object, object) -> object
        self.A, self.alpha, self.beta, self.gamma, self.b, self.nu,\
        self.lamb, self.R, self.eta, self.kappa = A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa

    def m_fun(self, theta):
        return (1 - np.exp(- self.eta * theta))

    def m_funprime(self, theta):
        return self.eta * np.exp(-self.eta * theta)

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

    def firmfoc(self, y, theta):
        return +beta * q_fun(theta) * f_prime(A, y) - V_funprime(y)

    def firmfoc_inf(self,y,theta):
        gtheta  = ((1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2)
        return beta * q_fun(theta) * f_prime(A, y)  - V_funprime(y)*gtheta

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
        dw = V_fun(y) * (1 - beta) * (1 - beta * (1 - lamb)) * q_funprime(theta) / q_fun(theta) ** 2
        EE = E_fun(U_guess_int, E_guess_int, a_prime, a_emp_func(a_prime), w)
        UU = U_fun(U_guess_int, E_guess_int, a_prime, a_func(a_prime), theta)
        focc1 = -u_prime(c_u) + beta * R * (m_fun(theta) * u_prime(c_et) + (1 - m_fun(theta)) * u_prime(c_ut))
        focc2 = m_fun(theta) * u_prime(c_et) * dw + m_funprime(theta) * (EE - UU)  # u(c_et) - u(c_ut)

        return np.array([focc1, focc2])

    def foc_emp(self, a_prime, w):
        c_e = R * a - a_prime + w
        c_et = R * a_prime - a_emp_func(a_prime) + w
        c_ut = b + R * a_prime - a_func(a_prime)

        return -u_prime(c_e) + beta * R * (lamb * u_prime(c_ut) + (1 - lamb) * u_prime(c_et))

    def E_fun(self,U, E, a,a_prime,w):
        c_e = R * a - a_prime + w
        return u(c_e) + beta*((1-lamb)*E(a) + lamb*U(a))

    def U_fun(self, U, E, a, a_prime, theta):
        c_u = b + R * a - a_prime
        return u(c_u) + beta*(m_fun(theta)*E(a) + (1-m_fun(theta))*U(a))

    def wage(self,theta, y):
        return f(A, y) - kappa * (1 - beta * (1 - lamb)) / (beta * q_fun(theta)) - V_fun(y) * (
            (1 - beta * (1 - lamb)) * (1 - beta * (1 - q_fun(theta))) - lamb * q_fun(theta) * beta ** 2) / (
                                                                                   beta * q_fun(theta))



######################################################################################################

gg = Homogeneous_agents()
A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa = gg.A, gg.alpha, gg.beta, gg.gamma, gg.b, gg.nu, gg.lamb, gg.R, gg.eta, gg.kappa
u, u_prime, f, f_prime, uprime_inv, m_fun, m_funprime, \
q_fun, q_funprime, V_fun, V_funprime, firmfoc, firmfoc_inf, foccs, foc_emp, U_fun, E_fun, wage = gg.u, gg.u_prime, gg.f, gg.f_prime, gg.uprime_inv, \
    gg.m_fun, gg.m_funprime, gg.q_fun, gg.q_funprime, gg.V_fun, gg.V_funprime, gg.firmfoc,\
                                                            gg.firmfoc_inf, gg.foccs, gg.foc_emp,  gg.U_fun, gg.E_fun, gg.wage






low_a = 1
high_a = 70
m = 35  # number of nodes

a_grid = np.linspace(low_a,high_a,num=m)
a_func = CubicSpline(a_grid, 0.3*a_grid)
a_star = np.empty_like(a_grid)
theta_star = np.empty_like(a_grid)
E_guess = np.zeros_like(a_grid)
U_guess = np.zeros_like(a_grid)
E = np.empty_like(a_grid)
U = np.empty_like(a_grid)
E_guess_int = CubicSpline(a_grid,E_guess)
U_guess_int = CubicSpline(a_grid,U_guess)
y_star = np.empty_like(a_star)
wage_star = np.empty_like(a_star)
a_emp_func = CubicSpline(a_grid, 1.0*a_grid)
a_emp_star = np.empty_like(a_grid)



w = 0.9
for i in range(13):
    for j in range(m):
        a = a_grid[j]
        a_emp_star[j] = fsolve(foc_emp, a, w)

    a_emp_func = CubicSpline(a_grid, a_emp_star)


print(a_emp_star)






for j in range(11):
    for i in range(m):
        a = a_grid[i]
        a_prime = a/3
        thetas = [a_prime, 0.6]
        #print(a)
        try:
            solv2 = broyden2(foccs, thetas, f_tol=1e-5)
        except:
            pass
        #solv2 = newton_krylov(foccs, thetass, method='lgmres', verbose=0)
        #solv2 = broyden1(foccs, thetas, f_tol=1e-5)
        #print(solv2)
        #theta = solv2[1]
        a_star[i] = solv2[0]
        theta_star[i] = solv2[1]
        try:
            y_star[i] = fsolve(firmfoc_inf, 207.2, theta_star[i])
        except:
            pass
        wage_star[i] = wage(theta_star[i], y_star[i])
        E[i] = E_fun(U_guess_int, E_guess_int, a_star[i], a_emp_func(a_star[i]),wage_star[i])
        U[i] = U_fun(U_guess_int, E_guess_int, a_star[i], a_func(a_star[i]), theta_star[i])

    print(theta_star)
    a_func = CubicSpline(a_grid, a_star)
    E_guess_int = CubicSpline(a_grid, E)
    U_guess_int = CubicSpline(a_grid, U)

for i in range(m):
    y_star[i] = fsolve(firmfoc_inf, 207.2, theta_star[i])
    wage_star[i] = wage(theta_star[i],y_star[i])


print(wage_star)


print(a_star)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, y_star , label='productivity')
plt.xlabel('assets')
ax.legend(loc='lower right')
#plt.show()


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, m_fun(theta_star) , label='probability')
plt.xlabel('assets')
ax.legend(loc='upper right')
#plt.show()



fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, a_star , label='a_star unemp')
ax.plot(a_grid, a_emp_func(a_grid), label='a_star emp')
ax.plot(a_grid, a_grid , label='no savings line')
plt.xlabel('assets')
ax.legend(loc='lower right')
#plt.show()



fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, E , label='value of employment')
ax.plot(a_grid, U , label='value of unemployment')
plt.xlabel('assets')
ax.legend(loc='lower right')
plt.show()

U_func = CubicSpline(a_grid, U)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.linspace(low_a,high_a,num=200), U_func(np.linspace(low_a,high_a,num=200)) , label='interpolated value of unemployment')
plt.xlabel('assets')
ax.legend(loc='lower right')
plt.show()



