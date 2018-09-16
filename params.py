##################################################################
########################### PARAMETERS ###########################
##################################################################


A = 1.0  # TFP parameter
alpha = 0.5 # productivity parameter of labor
beta = 0.99 # time discounting
eta = 1.0 # parameter in matching function
gamma = 4.0 # parameter in k(y)
kappa = 4.5  # parameter in k(y)
b = 2.5 # home production
nu = 2.0 # elasticity of intertemporal substitution
lamb = 0.035 #  exogenous probability of separation
R = 1.01 # gross interest rate
phi = 0.05 # human capital adjustment rate
pi = 0.125 # probability employed work can search

low_a = 0 # lowest value of asset holdings in asset grid
high_a = 7.0 # highest value of asset holdings in asset grid
low_w = 7.00 # lowest value of asset holdings in wage grid
high_w = 10.0 # highest value of asset holdings in wage grid
low_x = 2.8 # lowest value of asset holdings in productivity grid
high_x = 4.0 # highest value of asset holdings in productivity grid
low_y = 0.7 # lowest value of asset holdings in firm productivity grid
high_y = 2.2 # highest value of asset holdings in firm productivity grid

m = 24  # number of nodes in asset grid
m_x = int(m/3) # number of nodes in productivity grid
m_w = 10

rsize = int(m*m_x) # row size of grid (number of points to be sampled)  if m = 60 ==> rsize = 1200
life = 61      # number of life periods (and thus of cohorts)
x_points=21 # number of nodes in labor productivity grid

interp_strategy = 'gpr' # gpr for gaussian processes, standard for interpn

n_workers = 1000  # number of simulated workers

adj = 2.0



## the problem now is what happens when y > 1