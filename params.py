##################################################################
########################### PARAMETERS ###########################
##################################################################


A = 1.0  # TFP parameter
alpha = 0.5 # productivity parameter of labor
beta = 0.99 # time discounting
eta = 0.75 # parameter in matching function
gamma = 4.5 # parameter in k(y)
kappa = 4.0  # parameter in k(y)
b = 2.5 # home production
nu = 2.0 # elasticity of intertemporal substitution
lamb = 0.035 #  exogenous probability of separation
R = 1.01 # gross interest rate
phi = 0.06 # human capital adjustment rate
pi = 0.15 # probability employed work can search

low_a = 0 # lowest value of asset holdings in asset grid
high_a = 6.7 # highest value of asset holdings in asset grid
low_w = 7.5 # lowest value of asset holdings in wage grid
high_w = 10.5 # highest value of asset holdings in wage grid
low_x = 2.8 # lowest value of asset holdings in productivity grid
high_x = 4.1 # highest value of asset holdings in productivity grid
low_y = 0.7 # lowest value of asset holdings in firm productivity grid
high_y = 2.1 # highest value of asset holdings in firm productivity grid

m = 21  # number of nodes in asset grid
m_x = int(m/3) # number of nodes in productivity grid
m_w = 10

rsize = int(m*m_x) # row size of grid (number of points to be sampled)  if m = 60 ==> rsize = 1200
life = int(80) # number of life periods * 2 (and thus of cohorts * 2) has to be even number
years = int(life/2)
x_points=24 # number of nodes in labor productivity grid

interp_strategy = 'gpr' # gpr for gaussian processes, standard for interpn

n_workers = 1000  # number of simulated workers
half_workers = 500

adj = 2.2
train_cost = 1.0


plot_stuff = 1 # 1 : yes, 0 : no