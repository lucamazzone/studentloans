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


######################################################################################################
######################################################################################################
######################################################################################################


class First_Extension:
    def __init__(self, A =  3.0, alpha=0.2, beta=0.9, eta = 2.0, gamma = 10.0,  b = 1.0, nu= 2.0, lamb = 0.045, R = 1.02, kappa = 8.0):
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
        return A * (y ** self.alpha)*x**(1-self.alpha)

    def f_prime(self, A, y):
        return A * self.alpha * y ** (self.alpha - 1)*x**(1-self.alpha)

    def k_fun(self, y):
        return self.kappa*(y**self.gamma)/self.gamma # self.kappa*(1.0+y**gamma)**(1/gamma)

    def k_funprime(self, y):
        return  self.kappa*(y**(self.gamma-1)) # self.kappa*(y**(gamma-1))*(1.0+y**gamma)**(1/gamma-1)

    def c_u(self,a,a_prime):
        return self.b + self.R * a - a_prime

    def c_e(self,w, a,a_prime):
        return w + self.R * a - a_prime

    def firmfoc_last(self, y, theta,t):
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta*(self.beta * (1 - self.lamb)) ** (i)
        return f_prime(A, y)*q_fun(theta)*cosa1 - k_funprime(y)


    def wage(self,y,theta,t):
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta * (self.beta * (1 - self.lamb)) ** (i)
        return  f(A,y) - k_fun(y)/(cosa1*q_fun(theta))


    def E_fun(self,U, E, a,a_prime,w):
        return u(c_e(w,a,a_prime)) + beta*((1-lamb)*E + lamb*U)

    def E_funprime(self,E_prime, a,a_prime,w):
        return u_prime(c_e(w,a,a_prime)) + self.beta*(1-self.lamb)*E_prime  #E_prime function of a_prime

    def U_fun(self, U, E, a, a_prime, theta):
        return u(c_u(a,a_prime)) + self.beta*(m_fun(theta)*E + (1-m_fun(theta))*U)


    def foc_empl(self, a_emp, w):
        a_prime = a_emp
        return -u_prime(c_e(w, a, a_prime)) + beta * R * (
            (1 - lamb) * u_prime(c_e(w, a_prime,a_emp_func(a_prime,w)))  #
            + (lamb) * u_prime(c_u(a_prime, a_func(a_prime))))


    def foccs(self, thetas):
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta* (self.beta * (1 - self.lamb)) ** i
        a_prime = thetas[0]
        theta = thetas[1]
        y = thetas[2]
        vincolo = thetas[3]
        w_wage = wage(y, theta,t)
        cu = c_u(a, a_prime)
        cut = c_u(a_prime, a_func(a_prime))
        cet = c_e(w_wage, a_prime, a_emp_func(a_prime,w_wage))
        UU = U_func(a_prime)
        EE = E_func(a_prime)
        E_prime = E_prime_func(a_prime)

        foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) + np.amax([0,+vincolo])
        foc2 = m_funprime(theta) * ( E_fun(UU,EE,a_prime,a_emp_func(a_prime,w_wage),w_wage) - U_fun(UU,EE,a_prime,a_func(a_prime),theta) ) + \
               theta * E_funprime(E_prime,a_prime,a_emp_func(a_prime,w_wage),w_wage)  * q_funprime(theta) *  k_fun(y) / ( q_fun(theta))  # u_prime(cet)
        foc3 = f_prime(A, y)*q_fun(theta)*cosa1 - k_funprime(y)
        foc4 = np.amax([0,-vincolo]) - a_prime

        return [foc1, foc2, foc3,foc4]  #np.array().ravel()

    def l_fun(self,x):
        return 1/(1+np.exp(-2*1000*(x)))

    def l_fun_prime(self,x):
        return 2*1000*np.exp(2*1000*x)/(1+np.exp(2*1000*(x)))**2


##############################################################################################################################

class Human_Capital_Accumulation:
    def __init__(self, A =  4.5, alpha=0.2, beta=0.9, eta = 1.0, gamma = 8.0, b = 0.5, nu= 2.0, lamb = 0.045, R = 1.02, kappa = 8.0, phi = 0.2):
        # type: (object, object, object, object, object, object) -> object
        self.A, self.alpha, self.beta, self.gamma, self.b, self.nu,\
        self.lamb, self.R, self.eta, self.kappa, self.phi = A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa, phi


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

    def f(self, A, y,x):
        return A * (y ** self.alpha)*g_fun(y,x)**(1-self.alpha)

    def f_prime(self, A, y, x):
        return A * (self.alpha * y ** (self.alpha - 1)*g_fun(y,x)**(1-self.alpha) + (1-self.alpha)*g_fun_prime(y,x)*(y**self.alpha)*g_fun(y,x)**(-self.alpha))

    def k_fun(self, y):
        return self.kappa*(y**self.gamma)/self.gamma # self.kappa*(1.0+y**gamma)**(1/gamma)

    def k_funprime(self, y):
        return  self.kappa*(y**(self.gamma-1)) # self.kappa*(y**(gamma-1))*(1.0+y**gamma)**(1/gamma-1)

    def c_u(self,a,a_prime):
        return self.b + self.R * a - a_prime

    def c_e(self,w, a,a_prime):
        return w + self.R * a - a_prime

    def firmfoc_last(self, y, params):
        theta = params[0]
        x = params[1]
        t = params[2]
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta*(self.beta * (1 - self.lamb)) ** (i)
        return f_prime(A, y ,x )*q_fun(theta)*cosa1 - k_funprime(y)


    def wage(self,y,theta,x,t):
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta * (self.beta * (1 - self.lamb)) ** (i)
        return  f(A,y,x) - k_fun(y)/(cosa1*q_fun(theta))


    def E_fun(self,U, E, a,a_prime,w):
        return u(c_e(w,a,a_prime)) + beta*((1-lamb)*E + lamb*U)

    def E_funprime(self,E_prime, a,a_prime,w):
        return u_prime(c_e(w,a,a_prime)) + self.beta*(1-self.lamb)*E_prime  #E_prime function of a_prime

    def U_fun(self, U, E, a, a_prime, theta):
        return u(c_u(a,a_prime)) + self.beta*(m_fun(theta)*E + (1-m_fun(theta))*U)


    def foc_empl(self, a_emp, w):
        a_prime = a_emp
        return -u_prime(c_e(w, a, a_prime)) + beta * R * (
            (1 - lamb) * u_prime(c_e(w, a_prime,a_emp_func(a_prime,w)))  #
            + (lamb) * u_prime(c_u(a_prime, a_func(a_prime))))


    def foccs(self,thetas,params):
        a = params[0]
        x = params[1]
        t = params[2]
        cosa1 = 0
        for i in range(t):
            cosa1 = cosa1 + self.beta* (self.beta * (1 - self.lamb)) ** i
        a_prime = thetas[0]
        theta = thetas[1]
        y = thetas[2]
        vincolo = thetas[3]
        w_wage = wage(y,theta,x,t)
        cu = c_u(a, a_prime)
        cut = c_u(a_prime, a_func(a_prime,x))
        cet = c_e(w_wage, a_prime, a_emp_func(a_prime,w_wage,g_fun(y,x)))
        UU = U_func(a_prime,x)
        EE = E_func(a_prime,x)
        E_prime = E_prime_func(a_prime,x)

        foc1 = -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) + np.amax([0,+vincolo])
        foc2 = m_funprime(theta) * ( E_fun(UU,EE,a_prime,a_emp_func(a_prime,w_wage,g_fun(y,x)),w_wage) - U_fun(UU,EE,a_prime,a_func(a_prime,x),theta) ) + \
               theta * E_funprime(E_prime,a_prime,a_emp_func(a_prime,w_wage,g_fun(y,x)),w_wage)  * q_funprime(theta) *  k_fun(y) / ( q_fun(theta))  # u_prime(cet)
        foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)
        foc4 = np.amax([0,-vincolo]) - a_prime

        return [foc1.item(), foc2.item(), foc3,foc4]  #np.array().ravel()

    def l_fun(self,dis):
        if np.abs(dis):
            dis = dis/np.abs(dis)
        return 1/(1+np.exp(-2*100*(dis)))

    def l_fun_prime(self,dis):
        if np.abs(dis):
            dis = dis/np.abs(dis)
        return 2*100*np.exp(2*100*dis)/(1+np.exp(2*100*(dis)))**2

    def g_fun(self,p,x):
        y = p*10
        return x + phi*(y-x)*l_fun(y-x)

    def g_fun_prime(self,p,x):
        y = p*10
        return phi*l_fun(y-x) + phi*(y-x)*l_fun_prime(y-x)


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


gg = First_Extension()
A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa = gg.A, gg.alpha, gg.beta, gg.gamma, gg.b, gg.nu, gg.lamb, gg.R, gg.eta, gg.kappa
m_fun, m_funprime, q_fun, q_funprime, u, uprime_inv, u_prime, f, f_prime, k_fun, k_funprime, c_u, c_e, firmfoc_last, wage, \
    E_fun,E_funprime, U_fun, foc_empl, foccs, l_fun, l_fun_prime = \
    gg.m_fun, gg.m_funprime, gg.q_fun,gg.q_funprime, gg.u, gg.uprime_inv, gg.u_prime, gg.f, gg.f_prime, gg.k_fun,  \
                        gg.k_funprime, gg.c_u, gg.c_e, gg.firmfoc_last, gg.wage, gg.E_fun,gg.E_funprime, gg.U_fun, gg.foc_empl, \
                                                                   gg.foccs, gg.l_fun, gg.l_fun_prime


## grids
low_a = 0
high_a = 16.5
m = 45  # number of nodes
life = 9 # number of life periods (and thus of cohorts)
a_grid = np.linspace(low_a,high_a,num=m)
wage_grid = np.linspace(8,21,m)
lifegrid = np.linspace(life,1,life)
print(lifegrid)

## result grids
theta_star = np.zeros((m,life))
y_star = np.empty_like(a_grid)
wage_star = np.zeros((m,life))
AA, WW = np.meshgrid(a_grid, wage_grid)
amesh1,amesh2 = np.meshgrid(lifegrid,a_grid)

U_new = np.zeros((m,life))
E_new = np.zeros((m,life))
E_prime_new = np.zeros(m)
a_star_new = np.zeros((m,life))
a_star_emp_new = np.zeros((m,m,life))



## DECLARE STARTING VALUES AT THE BEGINNING OF ITERATION
## notice: since the life-cycle problem is solved backwards, a_star and a_emp_star = 0 in T
U_old = np.zeros_like(a_grid)
E_old = np.zeros_like(a_grid)
E_prime_old = np.zeros_like(a_grid)
a_star_old = np.zeros_like(a_grid)
a_star_emp_old = np.zeros_like(AA)


## interpolation
U_func = CubicSpline(a_grid,U_old)
a_func = CubicSpline(a_grid, a_star_old)
E_func = CubicSpline(a_grid,E_old)
E_prime_func = CubicSpline(a_grid,E_prime_old)
a_emp_func = lambda x,y: interpn((a_grid, wage_grid), a_star_emp_old, (x,y), bounds_error = False, fill_value = None) #


## solution

x = 6.5

for t in range(1,life+1):

    for j,w in enumerate(wage_grid):
        for i,a in enumerate(a_grid):
            sol = fsolve(foc_empl,a,w)
            a_star_emp_new[i,j,t-1] = sol


    for jj,a in enumerate(a_grid):
        thetass = np.array([0.55 * a, 0.85, 1.2, 0])
        sol = root(foccs, thetass, method='hybr')
        if (sol.status==1):
            a_star_new[jj,t-1] = sol.x[0]
            theta_star[jj,t-1] = sol.x[1]
            y_star[jj] =  sol.x[2]
        else:
            a_star_new[jj,t-1] = 2*a_star_new[jj-1,t-1]- a_star_new[jj-2,t-1]
            theta_star[jj, t - 1] = 2 * theta_star[jj-1, t - 1] - theta_star[jj-2, t - 1]
            y_star[jj] = 2 * y_star[jj-1] - y_star[jj-2]

        wage_star[jj,t-1] = wage(y_star[jj], theta_star[jj,t-1], t)
        #print(y_star[jj],m_fun(theta_star[jj,t-1]),wage_star[jj,t-1],a_star_new[jj,t-1])
        U_new[jj,t-1] = U_fun(U_old[jj],E_old[jj], a_star_new[jj,t-1],a_func(a_star_new[jj,t-1]),theta_star[jj,t-1])
        E_new[jj,t-1] = E_fun(U_old[jj],E_old[jj], a_star_new[jj,t-1],a_emp_func(a_star_new[jj,t-1], wage_star[jj,t-1]), wage_star[jj,t-1])
        E_prime_new[jj] = E_funprime(E_prime_old[jj], a_star_new[jj,t-1] ,a_emp_func(a_star_new[jj,t-1],wage_star[jj,t-1]), wage_star[jj,t-1])
        #print(E_prime_new[jj])




    a_star_emp_old = a_star_emp_new[:,:,t-1]
    a_star_old = a_star_new[:,t-1]
    U_old = U_new[:,t-1]
    E_old = E_new[:,t-1]
    E_prime_old = E_prime_new


    U_func = CubicSpline(a_grid,U_old)
    E_func = CubicSpline(a_grid,E_old)
    a_func = CubicSpline(a_grid, a_star_old)
    E_prime_func = CubicSpline(a_grid,E_prime_old)


#######################################################



fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, a_star_emp_new[:,41,:] , label='a_star_emp high wage')
ax.plot(a_grid, a_star_new[:,:] , label='a_star unemp')
ax.plot(a_grid, a_grid , label='45 degrees line')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, a_star_emp_new[:,1,:] , label='a_star_emp low wage')
ax.plot(a_grid, a_star_new[:,:] , label='a_star unemp')
ax.plot(a_grid, a_grid , label='45 degrees line')



fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a_grid, m_fun(theta_star[:,:]) , label='wage age 6')
#ax.plot(a_grid, m_fun(theta_star[:,t-2]), label='wage age 5')
#ax.plot(a_grid, m_fun(theta_star[:,t-3]) , label='wage age 4')
#ax.plot(a_grid, m_fun(theta_star[:,t-4]) , label='a_star_emp age 3')
#ax.plot(a_grid, m_fun(theta_star[:,t-5]) , label='a_star_emp age 2')
#ax.plot(a_grid, m_fun(theta_star[:,t-6]) , label='a_star_emp age 1')
#ax.plot(a_grid, m_fun(theta_star[:,t-5]) , label='a_star_emp age 7')
#ax.plot(a_grid, m_fun(theta_star[:,t-6]) , label='a_star_emp age 6')
#ax.plot(a_grid, m_fun(theta_star[:,t-7]) , label='a_star_emp age 5')
#ax.plot(a_grid, m_fun(theta_star[:,t-8]) , label='a_star_emp age 4')
#ax.plot(a_grid, m_fun(theta_star[:,t-9]) , label='a_star_emp age 3')
#ax.plot(a_grid, m_fun(theta_star[:,t-10]) , label='a_star_emp age 2')
#ax.plot(a_grid, m_fun(theta_star[:,t-11]) , label='a_star_emp age 1')

plt.xlabel('assets')
ax.legend(loc='lower left')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(amesh1, amesh2, a_star_emp_new[:,30,:], color='b')
plt.title('Policy Function for Savings of Employed')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(amesh1, amesh2, a_star_new[:,:], color='b')
plt.title('Policy Function for Savings of UnEmployed')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(amesh1*5+10, amesh2, wage_star, color='b')
#plt.title('Equilibrium Wages')
ax.set_xlabel('age')
ax.set_ylabel('assets holding')


plt.show()


############################################################################


y = 0.65
theta = 0.9
t =  1
a = 4
a_prime = 0.0
vincolo = 0

cosa1 = 0
for i in range(t):
    cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i

w_wage = wage(y, theta,t)
cu = c_u(a, a_prime)
cut = c_u(a_prime, a_func(a_prime))
cet = c_e(w_wage, a_prime, a_emp_func(a_prime,w_wage))
UU = U_func(a_prime)
EE = E_func(a_prime)
E_prime = E_prime_func(a_prime)

print('wage', w_wage)
print('cu', cu)
print('cut', cut)
print('cet', cet)
print('UU', UU)
print('EE', EE)
print('E_prime', E_prime)

foc1 =  -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) + np.amax([0,+vincolo])

print('foc1',foc1)

foc2 = m_funprime(theta) * ( E_fun(UU,EE,a_prime,a_emp_func(a_prime,w_wage),w_wage) - U_fun(UU,EE,a_prime,a_func(a_prime),theta) ) + \
               theta * E_funprime(E_prime,a_prime,a_emp_func(a_prime,w_wage),w_wage)  * q_funprime(theta) *  k_fun(y) / ( q_fun(theta))

print('foc2',foc2)


foc3 = f_prime(A, y)*q_fun(theta)*cosa1 - k_funprime(y)
foc4 = np.amax([0,-vincolo]) - a_prime

print('foc3',foc3)
print('foc4',foc4)

#############################################################################

gg = Human_Capital_Accumulation()
A, alpha, beta, gamma, b, nu, lamb, R, eta, kappa, phi = gg.A, gg.alpha, gg.beta, gg.gamma, gg.b, gg.nu, gg.lamb, gg.R, gg.eta, gg.kappa, gg.phi
m_fun, m_funprime, q_fun, q_funprime, u, uprime_inv, u_prime, f, f_prime, k_fun, k_funprime, c_u, c_e, firmfoc_last, wage, \
    E_fun,E_funprime, U_fun, foc_empl, foccs, l_fun, l_fun_prime, g_fun,g_fun_prime = \
    gg.m_fun, gg.m_funprime, gg.q_fun,gg.q_funprime, gg.u, gg.uprime_inv, gg.u_prime, gg.f, gg.f_prime, gg.k_fun,  \
                        gg.k_funprime, gg.c_u, gg.c_e, gg.firmfoc_last, gg.wage, gg.E_fun,gg.E_funprime, gg.U_fun, gg.foc_empl, \
                                                                   gg.foccs, gg.l_fun, gg.l_fun_prime, gg.g_fun, gg.g_fun_prime


life = 1
lifegrid = np.linspace(life,1,life)
x_points=11
y_points=11
x_grid = np.linspace(6.0,7.0,x_points)
y_grid = np.linspace(0.5,1,y_points)

## result grids
theta_star = np.zeros((m,x_points,life))
y_star = np.empty_like((a_grid,x_points))  #
wage_star = np.zeros((m,x_points,life))
AA, WW = np.meshgrid(a_grid, wage_grid, sparse=False)
#xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
amesh1,amesh2 = np.meshgrid(lifegrid,a_grid)

U_new = np.zeros((m,x_points,life))
E_new = np.zeros((m,x_points,y_points,life))
E_prime_new = np.zeros((m,x_points,y_points))
a_star_new = np.zeros((m,x_points,life))
a_star_emp_new = np.zeros((m,m,x_points,y_points,life))

## DECLARE STARTING VALUES AT THE BEGINNING OF ITERATION
## notice: since the life-cycle problem is solved backwards, a_star and a_emp_star = 0 in T
U_old = np.zeros((m,x_points))
E_old = np.zeros((m,x_points))
E_prime_old = np.zeros((m,x_points))
a_star_old = np.zeros((m,x_points))
a_star_emp_old = np.zeros((m,m,x_points))


## interpolation
#U_func = CubicSpline(a_grid,U_old)
#a_func = CubicSpline(a_grid, a_star_old)
#E_func = CubicSpline(a_grid,E_old)
#E_prime_func = CubicSpline(a_grid,E_prime_old)

a_func = lambda q,h: interpn((a_grid,x_grid), a_star_old, (q,h), bounds_error = False, fill_value = None) #
U_func = lambda q,h: interpn((a_grid,x_grid), U_old, (q,h), bounds_error = False, fill_value = None) #
E_func = lambda q,h: interpn((a_grid,x_grid), E_old, (q,h), bounds_error = False, fill_value = None) #
E_prime_func = lambda q,h: interpn((a_grid,x_grid), E_prime_old, (q,h), bounds_error = False, fill_value = None) #
a_emp_func = lambda q,h,z: interpn((a_grid, wage_grid,x_grid), a_star_emp_old, (q,h,z), bounds_error = False, fill_value = None) #



t = 1



#print(g_fun(0.65,x))
#print(g_fun_prime(0.65,x))
#print(f(A,0.65,x))





testfun = np.empty_like(x_grid)
testfunprime = np.empty_like(x_grid)
#for j,y in enumerate(x_grid):
#    x = x_grid[j]
#    testfun[j] = firmfoc_last(0.65,[0.9,1])#f(A,0.65)
#    testfunprime[j] = f_prime(A,0.65)



#fig, ax = plt.subplots(figsize=(9, 5))
#ax.plot(x_grid, testfun, label='f(a,y)')
#ax.plot(a_grid, a_star_new[:,:] , label='a_star unemp')
#ax.plot(a_grid, a_grid , label='45 degrees line')

#plt.show()

#y = 0.66
#params = [0.7,6.0,1]
#sol = fsolve(firmfoc_last,y,params)
#print(sol)



#print( fsolve(firmfoc_last,y,params))



x = 7.8
y = 0.7
theta = 0.8
t =  1
a = 11
a_prime = 0.0
vincolo = 0

cosa1 = 0
for i in range(t):
    cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i

cu = c_u(a, a_prime)
cut = c_u(a_prime, a_func(a_prime,x))
cet = c_e(w_wage, a_prime, a_emp_func(a_prime,w_wage,g_fun(y,x)))
UU = U_func(a_prime,x)
EE = E_func(a_prime,x)
E_prime = E_prime_func(a_prime,x)


print('wage',w_wage)
print('cu',cu)
print('cut',cut)
print('cet', cet)
print('UU',UU)
print('EE',EE)
print('E_prime',E_prime)


foc1 =  -u_prime(cu) + beta * R * (m_fun(theta) * u_prime(cet) + (1 - m_fun(theta)) * u_prime(cut)) + np.amax([0,+vincolo])

print('foc1',foc1)

foc2 = m_funprime(theta) * ( E_fun(UU,EE,a_prime,a_emp_func(a_prime,w_wage,g_fun(y,x)),w_wage) - U_fun(UU,EE,a_prime,a_func(a_prime,g_fun(y,x)),theta) ) + \
               theta * E_funprime(E_prime,a_prime,a_emp_func(a_prime,w_wage,g_fun(y,x)),w_wage)  * q_funprime(theta) *  k_fun(y) / ( q_fun(theta))

print('foc2',foc2)


foc3 = f_prime(A, y, x)*q_fun(theta)*cosa1 - k_funprime(y)

print('foc3',foc3)

foc4 = np.amax([0,-vincolo]) - a_prime

#print('foc3',foc3)
print('foc4',foc4)


params = [a,x,t]
thetass = [a_prime,theta,x*1.3,vincolo]
sol = root(foccs,thetass,params, method='lm')
if  (sol.status>1):
    print('try other method')
    sol = root(foccs, thetass, params, method='hybr')
    if (sol.status > 1):
        print('try yet another method')
        sol = root(foccs, thetass, params, method='anderson')


print(sol)
print(m_fun(sol.x[1]))

##############################################################################################################

params = [x,t]
#thetas = [a_prime,theta,y,vincolo]

#print('focs', foccs(thetas,params))

#
#print(sol)
#print(a_func(4,6.6))
#theta_stars = sol.x[1]
#print(theta_stars)
#w_wage = wage(y, theta,x,t)
#print(c_e(w_wage, a_prime, a_emp_func(a_prime,w_wage,g_fun(y,x))))


'''
t = 1
y = 50
theta = 1.1
cosa1 = 0
for i in range(t):
    cosa1 = cosa1 + beta*(beta * (1 - lamb)) ** (i)
print(f_prime(A, y)*q_fun(theta)*cosa1)
print(k_funprime(y))
y = fsolve(firmfoc_last,100,(theta,t))
print(y)
print(wage(y,theta,t))



eta = 0.25
x = 0.6
def cosette(x):
    return m_fun(1.0)*u_prime(1+11)*A*(1-alpha)*(0.6**alpha)*(x**(-alpha))*( -l_fun_prime(0.6-x)*(0.6-eta*(0.6-x)) + eta*l_fun(0.6-x)  )

#print(cosette)
#print(m_fun(1.0)*u_prime(1+12)*A*(1-alpha)*(0.7**alpha)*(x**(-alpha)))
print( -l_fun_prime(0.7-x)*(0.7-eta*(0.7-x))  )
print(+ eta*l_fun(0.7-x))

sol = root(cosette, 0.7, method='hybr')
print(sol)

'''
