###############################################################
########## DECLARES MATRICES###################################
###############################################################

import numpy as np
from params import *

##############################################################


## THESE ARE THE MATRICES USED TO FIT THE INTERPOLATOR

U_old = np.zeros((rsize,1))  # value function of unemployed
a_star_old = np.zeros((rsize,1))  # stores policy function of unemployed for assets
theta_star_old = np.zeros((rsize,1)) # stores policy function of unemployed for vacancy ratio
y_star_old = np.zeros((rsize,1))  # stores policy function of unemployed  for matched firm
wage_star_old = np.zeros((rsize,1)) # stores policy function of unemployed for bargained wage

a_star_emp_old = np.zeros((rsize, 1))  # stores policy function of employed for assets
theta_star_emp_old = np.zeros((rsize,1)) # stores policy function of employed for vacancy ratio
y_star_emp_old = np.zeros((rsize,1))  # stores policy function of employed  for matched firm
wage_star_emp_old = np.zeros((rsize,1)) # stores policy function of employed for bargained wage


E_old = np.zeros((rsize, 1))  # value function of employed  (used to update U)
E_old_b = np.zeros((rsize, 1))  # value function of employed (used to update E)
E_prime_old = np.zeros((rsize, 1))  # derivative of value function of employed (wrt w)
E_prime_old_b = np.zeros_like(E_prime_old)


## THE SAME MATRICES, KEPT FOR STORAGE

U_new = np.zeros((rsize,life))  # value function of unemployed
a_star_new = np.zeros((rsize,life))  # stores policy function of unemployed for assets
theta_star_new = np.zeros((rsize,life)) # stores policy function of unemployed for vacancy ratio
y_star_new = np.zeros((rsize,life))  # stores policy function of unemployed  for matched firm
wage_star_new = np.zeros((rsize,life)) # stores policy function of unemployed for bargained wage

a_star_emp_new = np.zeros((rsize, life))  # stores policy function of employed for assets
a_star_emp_new_b = np.empty_like(a_star_emp_new)

theta_star_emp_new = np.zeros((rsize,life)) # stores policy function of employed for vacancy ratio
y_star_emp_new = np.zeros((rsize,life))  # stores policy function of employed  for matched firm
wage_star_emp_new = np.zeros((rsize,life)) # stores policy function of employed for bargained wage






## GRIDS

a_grid = np.linspace(low_a, high_a, num=m)
a_u_try = np.empty((m,life))
a_u_highprod = np.empty((m,life))
a_e_try = np.empty((m,life))

x_grid = np.linspace(low_x, high_x, num=m_x)
wage_grid = np.linspace(low_w, high_w, num=m_x)
E_new = np.zeros((rsize, life))  # value function of employed
E_prime_new = np.zeros((rsize, life))  # derivative of value function of employed (wrt w)
E_prime_new_b = np.zeros_like(E_prime_new)
E_new_b = np.zeros_like(E_new)

## SIMULATION GRIDS

a_sim = np.empty((life,n_workers))
theta_sim = np.empty((life,n_workers))
wage_sim = np.empty((life,n_workers))
y_sim = np.empty((life,n_workers))
emp_status_sim = np.empty((life,n_workers))
hum_k_sim = np.empty((life,n_workers))
job_to_job_sim = np.empty((life,n_workers))

## useful stuff

state = np.empty((3))
e_state  = np.empty((5))

a_u_try = np.empty_like(a_grid)
a_u_highprod = np.empty_like(a_grid)
a_e_try = np.empty_like(a_grid)
E_e_try = np.empty_like(a_grid)
U_u_try = np.empty_like(a_grid)
y_try = np.empty_like(a_grid)
theta_try = np.empty_like(a_grid)
theta_emp_try = np.empty_like(a_grid)


if (interp_strategy == 'gpr'):

    empl_grid = np.random.uniform(low=0, high=1, size=(rsize, 4))
    empl_grid_b = np.empty_like(empl_grid)

    empl_grid[:, 0] = empl_grid[:, 0]*(high_a - low_a) + low_a  # first column is assets
    empl_grid[:, 1] = empl_grid[:, 1]*(high_x - low_x) + low_x  # second column is worker productivity
    empl_grid[:, 2] = empl_grid[:, 2]*(high_w - low_w) + low_w  # third column is wage
    empl_grid[:, 3] = empl_grid[:, 3]*(high_y - low_y) + low_y  # fourth column is firm productivity

    empl_grid_b[:, 0] = empl_grid[:, 0]  # first column is assets
    empl_grid_b[:, 1] = empl_grid[:, 1] # second column is worker productivity

    unempl_grid = empl_grid[:, [0, 1]]
    unempl_grid_old = unempl_grid


elif (interp_strategy == 'standard'):
    empl_grid = np.random.uniform(low=0, high=1, size=(rsize, 4))
    empl_grid_b = np.empty_like(empl_grid)
    AA, WW = np.meshgrid(a_grid, x_grid)
    A_A = AA.reshape((m * m_x, 1))
    W_W = WW.reshape((m * m_x, 1))
    aw = np.column_stack((A_A, W_W))

    empl_grid[:, [0, 1]] = aw  # first column is assets, second is worker productivity
    empl_grid[:, 2] = empl_grid[:, 2]*(high_w - low_w) + low_w  # third column is wage
    empl_grid[:, 3] = empl_grid[:, 3]*(high_y - low_y) + low_y  # fourth column is firm productivity
    unempl_grid = empl_grid[:, [0, 1]]

    empl_assets = A_A
    empl_prod = W_W
    empl_wage = empl_grid[:, 2]*(high_w - low_w) + low_w
    empl_firm = empl_grid[:, 3]*(high_y - low_y) + low_y

    a_star_emp_old = np.zeros_like(empl_assets)  # stores policy function of employed for assets
    a_star_emp_new = np.zeros((rsize, life))  # stores policy function of employed for assets
    a_star_emp_new_b = np.zeros((rsize, life))  # stores policy function of employed for assets

    #empl_grid_b[:,[0,1]] = aw
    emp_grid_b = empl_grid

    #a_star_emp_old = np.zeros((m, m_x, m_w, m_x ))  # stores policy function of employed for assets
    #a_star_emp_new = np.zeros((m, m_x, m_w, m_x, life))  # stores policy function of employed for assets

    #E_old = np.zeros((m, m_x, m_w, m_x, 1))  # value function of employed  (used to update U)
    #E_old_b = np.zeros((m, m_x, m_w, m_x, 1))  # value function of employed (used to update E)
    #E_prime_old = np.zeros((m, m_x, m_w, m_x, 1))  # derivative of value function of employed (wrt w)

    #E_new = np.zeros((m, m_x, m_w, m_x, life))  # value function of employed
    #E_prime_new = np.zeros((m, m_x, m_w, m_x, life))  # derivative of value function of employed (wrt w)

    #a_star_emp_new_b = np.empty_like(a_star_emp_new)
    #E_prime_old_b = np.zeros_like(E_prime_old)
    #E_prime_new_b = np.zeros_like(E_prime_new)
    #E_new_b = np.zeros_like(E_new)

    #U_old.reshape((m,m_x,1))  # value function of unemployed
    #a_star_old.reshape((m,m_x,1))  # stores policy function of unemployed for assets
    #theta_star_old.reshape((m,m_x,1))  # stores policy function of unemployed for vacancy ratio
    #y_star_old.reshape((m,m_x,1))  # stores policy function of unemployed  for matched firm
    #wage_star_old.reshape((m,m_x,1))  # stores policy function of unemployed for bargained wage

    #U_new.reshape((m,m_x,life))  # value function of unemployed
    #a_star_new.reshape((m,m_x,life))  # stores policy function of unemployed for assets
    #theta_star_new.reshape((m,m_x,life))  # stores policy function of unemployed for vacancy ratio
    #y_star_new.reshape((m,m_x,life))  # stores policy function of unemployed  for matched firm
    #wage_star_new.reshape((m,m_x,life))  # stores policy function of unemployed for bargained wage

