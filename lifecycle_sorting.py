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
from scipy.optimize import root
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn





class First_Extension:
    def __init__(self, A = 1.0, alpha=0.5, beta=0.955, eta = 3.5, gamma = 0.8, b = 1.5, nu= 2.0 , lamb = 0.055, R = 1.05, kappa = 1):
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
        return (y**(self.gamma))/(self.gamma) #(0.5 + y ** (self.gamma)) ** (1 / self.gamma)

    def V_funprime(self, y):
        return  y**(self.gamma-1) #((0.5 + y ** (self.gamma)) ** ((1/self.gamma) - 1)) * (y ** (self.gamma - 1))

    def c_u(self,a,a_prime):
        return self.b + self.R * a - a_prime

    def c_e(self,w, a,a_prime):
        return w + self.R * a - a_prime

    def firmfoc_last(self, y, theta):
        return +beta * q_fun(theta) * f_prime(A, y) - V_funprime(y)*(1-beta)*(1+q_fun(theta))

    def wage_last(self,y,theta,t):
        if (t==1):
            wage = f(self.A,y) - (V_fun(y)*(1-self.beta)*(1+q_fun(theta))+self.kappa)/(self.beta*q_fun(theta))
        else:
            cosa1 = 0
            for i in range(t):
                cosa1 = cosa1 + (self.beta*(1-self.lamb))**(i)

            cosa2 = 0
            for i in range(1, t):
                cosa2 = cosa2 + self.lamb * q_fun(theta) * (self.beta ** (i + 1)) * (1 - self.lamb) ** (i - 1)

            wage = f(A,y) - kappa/(self.beta*q_fun(theta)*cosa1) - V_fun(y)*(1-self.beta*(1-q_fun(theta)) - self.beta**(t+1) *
                                            (1-self.lamb**(t-1))*q_fun(theta) - cosa2)/(self.beta*q_fun(theta)*cosa1)
        return wage


    def E_fun(self,U, E, a,a_prime,w):
        return u(c_e(w,a,a_prime)) + beta*((1-lamb)*E + lamb*U)

    def U_fun(self, U, E, a, a_prime, theta):
        return u(c_u(a,a_prime)) + self.beta*(m_fun(theta)*E + (1-m_fun(theta))*U)

    def foccs(self, thetas):
        a_prime = thetas[0]
        theta = thetas[1]
        y = fsolve(firmfoc_last, 100, theta)
        cu = c_u(a, a_prime)
        cut = c_u(a_prime, a_func(a_prime))
        cet = c_e(wage_last(y, theta,t), a_prime, a_emp_func(a_prime,wage_last(y, theta,t)))
        foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut))
        foc2 = m_funprime(theta) * (u(cet) - u(cut)) + theta * u_prime(cet) * q_funprime(theta) * (
        (1 - beta) * V_fun(y) + kappa) / (beta * q_fun(theta))

        return np.array([foc1, foc2]).ravel()

    def foc_empl(self, a_emp, w):
        a_prime = a_emp
        return -u_prime(c_e(w, a, a_prime)) + beta * R * (
            (1 - lamb) * u_prime(c_e(w, a_prime,a_emp_func(a_prime,w)))  #
            + (lamb) * u_prime(c_u(a_prime, a_func(a_prime))))


######################################################################################################

gg = First_Extension()
A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa = gg.A, gg.alpha, gg.beta, gg.gamma, gg.b, gg.nu, gg.lamb, gg.R, gg.eta, gg.kappa
u, u_prime, uprime_inv,m_fun,m_funprime, firmfoc_last, q_fun, q_funprime, f,\
f_prime, V_fun, V_funprime, E_fun, U_fun, c_u, c_e, wage_last, foccs, foc_empl= gg.u, gg.u_prime, gg.uprime_inv, gg.m_fun, \
                    gg.m_funprime, gg.firmfoc_last,\
                    gg.q_fun, gg.q_funprime, gg.f , gg.f_prime, gg.V_fun, \
            gg.V_funprime, gg.E_fun, gg.U_fun, gg.c_u, gg.c_e, gg.wage_last, gg.foccs, gg.foc_empl




print( '*******************' )
print( '*******************' )
print( '*******************' )

low_a = 1
high_a = 60
m = 41  # number of nodes

a_grid = np.linspace(low_a,high_a,num=m)
a_func = CubicSpline(a_grid, 0.5*a_grid)
theta = 0.8

y = fsolve(firmfoc_last, 100, theta)

print(y)
print('wage if T-1', wage_last(y,theta,1))
print('wage if T-2' , wage_last(y,theta,2))
print('wage if T-3' , wage_last(y,theta,3))
print('wage if T-4' , wage_last(y,theta,4))
print('wage if T-5' , wage_last(y,theta,5))
print('wage if T-6' , wage_last(y,theta,6))






wage_grid = np.linspace(5,28,m)
AA, WW = np.meshgrid(a_grid, wage_grid)
U = np.zeros_like(a_grid)
E = np.zeros_like(AA)
a_star = np.zeros_like(a_grid)
theta_star = np.zeros_like(a_grid)
y_star = np.empty_like(a_grid)
wage_star = np.empty_like(a_grid)
E_func = lambda x,y: interpn((a_grid, wage_grid), E, (x,y), bounds_error = False, fill_value = None )
U_func = CubicSpline(a_grid,U)
a_star_emp_old = np.zeros_like(AA)
a_star_emp_new = np.zeros_like(AA)
a_func = CubicSpline(a_grid,a_star )
a_emp_func = lambda x,y: interpn((a_grid, wage_grid), a_star_emp_old, (x,y), bounds_error = False, fill_value = None )





print(a_emp_func(20,20))



t = 1

for i,a in enumerate(a_grid):
    thetass = np.array([0.55 * a, 0.48])
    sol = root(foccs, thetass, method='hybr')
    a_star[i] = sol.x[0]
    theta_star[i] = sol.x[1]
    y_star[i] = fsolve(firmfoc_last, 100, theta_star[i])
    wage_star[i]  = wage_last(y_star[i],theta_star[i],t)



for j,w in enumerate(wage_grid):
    for i,a in enumerate(a_grid):
        sol = fsolve(foc_empl,a,w)
        a_star_emp_new[i,j] = sol
        print(a_emp_func(sol,w))

a_star_emp_old = a_star_emp_new



a_func = CubicSpline(a_grid,a_star )
theta_func = CubicSpline(a_grid,theta_star)


t = 2

for i,a in enumerate(a_grid):
    thetass = np.array([0.55 * a, 0.48])
    sol = root(foccs, thetass, method='hybr')
    a_star[i] = sol.x[0]
    theta_star[i] = sol.x[1]
    y_star[i] = fsolve(firmfoc_last, 100, theta_star[i])
    wage_star[i]  = wage_last(y_star[i],theta_star[i],t)

for j,w in enumerate(wage_grid):
    for i,a in enumerate(a_grid):
        sol = fsolve(foc_empl,a,w)
        a_star_emp_new[i,j] = sol
        print(a_emp_func(sol,w))

a_star_emp_old = a_star_emp_new




fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, a_star_emp_new[:,26] , label='a_star_emp')
ax.plot(a_grid, a_star , label='a_star unemp')
ax.plot(a_grid, a_grid , label='no savings line')
plt.xlabel('assets')
ax.legend(loc='upper right')


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, wage_star , label='wage')
#ax.plot(a_grid, a_emp_func(a_grid), label='a_star emp')
plt.xlabel('assets')
ax.legend(loc='upper right')
#plt.show()



for i,a in enumerate(a_grid):
    u_u = U_func(a)
    e_u = E_func(a,wage_star[i])
    U[i] =   U_fun( u_u, e_u, a, a_func(a), theta_func(a))
    for j,w in enumerate(wage_grid):
        e_e = E_func(a, w)
        E[i,j] = E_fun(u_u, e_e, a,a_func(a), w)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(AA, WW, E, color='b')


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, theta_star , label='theta')
plt.xlabel('assets')
ax.legend(loc='upper right')
#plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, U , label='U')
ax.plot(a_grid, E , label='E')
plt.xlabel('assets')
ax.legend(loc='upper right')
plt.show()




'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(AA, WW, a_star_emp, 100, cmap='binary')


def foc1(a_prime):
    return   -u_prime(c_u(a,a_prime)) + beta*R* ( m_fun(theta) * u_prime(c_e(wage_last(y,theta),a_prime,0))
                                                  + (1-m_fun(theta))*u_prime(c_u(a_prime,0))  )

a_sols = np.empty_like(a_grid)
for j in range(10):
    for i,a in enumerate(a_grid):
        a_sol = fsolve(foc1,a)
        a_sols[i] = a_sol

    eee = np.linalg.norm(a_sols - a_func(a_grid))
    a_func = CubicSpline(a_grid, a_sols)
    if (eee<1e-5):
        break




#plt.show()


def foc2(theta):
    y = fsolve(firmfoc_last, 100, theta)
    return m_funprime(theta)*( u(c_e(wage_last(y,theta),a_prime,0)) - u(c_u(a_prime,0)) )\
        + theta*u_prime(c_e(wage_last(y,theta),a_prime,0))*q_funprime(theta)*\
          (V_fun(y)*(1-beta)+kappa)/(beta*q_fun(theta))


yy = np.empty_like(a_grid)
ww = np.empty_like(a_grid)
for i,a in enumerate(a_grid):
    a_prime = a_func(a)
    tt = fsolve(foc2, 0.85)
    yy[i] = fsolve(firmfoc_last, 100, tt)
    ww[i] = wage_last(yy[i],tt)





theta_sols = np.empty_like(a_grid)
y_sols = np.empty_like(a_grid)
for j in range(10):
    for i,a in enumerate(a_grid):
        theta_sol = fsolve(foc2,0.7)
        theta_sols[i] = a_sol
        y_sols[i] = fsolve(firmfoc_last, 100, theta_sol)






tet = fsolve(foc2,0.6)
print('solution of foc2 ',tet)

ips = fsolve(firmfoc_last, 100, tet)
print('corresponding productivity', ips)

print('and wage is ', wage_last(ips,tet))


'''
