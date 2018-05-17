##################################################################
########################### PARAMETERS ###########################
##################################################################


A = 1.0  # TFP parameter
alpha = 0.5 # productivity parameter of labor
beta = 0.99 # time discounting
eta = 1.5 # parameter in matching function
gamma = 4.0 # parameter in k(y)
kappa = 8.0  # parameter in k(y)
b = 2.0 # home production
nu = 2.0 # elasticity of intertemporal substitution
lamb = 0.045 #  exogenous probability of separation
R = 1.03 # gross interest rate
phi = 0.05 # human capital adjustment rate


low_a = 0 # lowest value of asset holdings in asset grid
high_a = 8.0 # highest value of asset holdings in asset grid
low_w = 4.0 # lowest value of asset holdings in wage grid
high_w = 6.0 # highest value of asset holdings in wage grid
low_x = 1.55 # lowest value of asset holdings in productivity grid
high_x = 1.6 # highest value of asset holdings in productivity grid
low_y = 0.7 # lowest value of asset holdings in firm productivity grid
high_y = 1.0 # highest value of asset holdings in firm productivity grid

m = 45  # number of nodes in asset grid
m_x = int(m/3) # number of nodes in productivity grid
m_w = 10

rsize = int(m*m_x) # row size of grid (number of points to be sampled)  if m = 60 ==> rsize = 1200
life = 7    # number of life periods (and thus of cohorts)
x_points=21 # number of nodes in labor productivity grid

interp_strategy = 'gpr' # gpr for gaussian processes, standard for interpn


## the problem now is what happens when y > 1