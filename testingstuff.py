import numpy as np
#import sys
#import os
#import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
#import numpy.matlib
from scipy.optimize import fsolve
#from scipy.optimize import brentq
#from scipy.interpolate import griddata
#from scipy.interpolate import CubicSpline
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
#from scipy.interpolate import interpn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C
import warnings
#from numpy import linalg as LA


n = 40
status_collect = np.ones(n*n)
status_collect[20] = 0
status_collect[300] = 0
status_collect[400] = 0

a_grid = np.linspace(-1,1,n)
x_grid = np.linspace(-1,1,n)

AA, WW = np.meshgrid(a_grid, x_grid)
A_A = AA.reshape((n*n, 1))
W_W = WW.reshape((n*n, 1))
#aw = np.column_stack((A_A, W_W))

a_star_old =  (1/np.pi)*np.exp( np.abs(AA  -WW)**(2))

print('a_star_old.shape',a_star_old.shape)

#status_collect.reshape((n,n))

controls_unemp = np.where(status_collect < 1.0)

status_collect[430] = 3

controls_emp = np.where(status_collect > 1.0)

joint = controls_unemp + controls_emp




for i,j in enumerate(joint[0]):
    #aw[j,:] = np.PINF
    (x_nan, a_nan) = divmod(j, n)
    #print(a_nan,x_nan)
    AA[a_nan,x_nan] = np.PINF


print(AA.shape)
array = np.ma.masked_invalid(AA)
newAA = AA[~array.mask]
newWW = WW[~array.mask]
newarr = a_star_old[~array.mask]

print('newAA.shape' , newAA.shape)
print('newWW.shape' , newWW.shape)
print('newarr.shape', newarr.shape)
a_func = lambda q,h: interpolate.griddata((newAA,newWW), newarr.ravel(),(q,h),method='cubic')

''' 
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(AA, WW, a_func(AA,WW), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

print(a_func(a_grid[20],x_grid[7]), (1/np.pi)*np.exp( np.abs(a_grid[20]  - x_grid[7])**(2)) )
print(a_func(a_grid[20],x_grid[0]), (1/np.pi)*np.exp( np.abs(a_grid[20]  - x_grid[0])**(2)) )
'''

n = 10j
(x1,x2,x3,x4) = np.mgrid[0:5:n,0:5:n,0:5:n,0:5:2j]


yy = x1*x2*x3*x4


#print(x1.shape)
x1 = x1.reshape((10*10*10*2,1))
x2 = x2.reshape((10*10*10*2,1))
x3 = x3.reshape((10*10*10*2,1))
x4 = x4.reshape((10*10*10*2,1))
yy = yy.reshape((10*10*10*2,1))


#a_emp_func = lambda q, h, s: interpolate.interpn((x1,x2,x3),yy, (q, h, s), bounds_error=False, fill_value=None)
                                                  #bounds_error=False, fill_value=None)  #


#print(x1.shape,yy.shape)
a_emp_func = interpolate.Rbf(x1,x2,x3,x4,yy)
#print(a_emp_func(2,2,2,2))
#print(a_emp_func(2,2,2))


dicts = {}
E_func = {}
keys = range(4)
values = ["Hi", "I", "am", "John"]
for i in keys:
        dicts[i] = values[i]
        E_func[i] = 'E_func_'+str(str(i))
print(dicts[1])
print(E_func)

kernel = C(1.0, (1e-3, 1e-3)) * RBF(7, (1e-3, 1e2))
E_func[1] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

print(E_func[1])


''' 
#a_star_old = a_star_old.reshape((n,n))

array = np.ma.masked_invalid(AA[:,0])
newAA = AA[~array.mask]
newWW = WW[~array.mask]
newarr = a_star_old[~array.mask]




print(a_func(newAA,newWW))
#griddata

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(AA, WW, a_func(AA,WW), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


array = np.ma.masked_invalid(a_star_old)
newaw = aw[~array.mask]
newarr = a_star_old[~array.mask]

print(array)


a_u_try = np.empty_like(grid)

for i,j in enumerate(grid):
    a_u_try[i] = a_func(i)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(grid, a_u_try, label='vabbe')
plt.legend(loc='upper left')


plt.show()
'''