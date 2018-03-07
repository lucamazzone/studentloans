import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C
import warnings
from numpy import linalg as LA



#Ex 4
#number of nodes
m = 9
#degree of polynomial
n = m-1
kStar = 7.2112
#set parameters
beta = 0.965
alpha = 0.36
delta = 0.06

#calculate cheby nodes
nodes = [-np.cos((2.*(j+1.)-1.)*np.pi/(2.*m)) for j in range(m)]
#rescale nodes 
nodes1 = [(nodes[j]+1)*kStar/2. + 0.5*kStar for j in range(m)]


#determine optimal polynomial coefficients using Time Point iteration
#staring value
cHat = [(1.-0.8*delta)*i for i in nodes1]
#create matrix with cheby poly using cheby nodes (z!! hence nodes)
T = np.polynomial.chebyshev.chebvander(nodes,n)
T = T.transpose()
coefs = []
for i in range(n+1):
	auxT = T[i,:]
	coefs.append(np.sum(cHat*auxT)/(np.sum(auxT**2.)))

epsilon = 10e-6
counter = 0
def func41(k1): 
	#define C_t using FOC
	cons = (1.-delta+alpha*k**(alpha-1.))*k+(1.-alpha)*k**alpha-k1
	#define k_t+2 using linear interpolation
	k2 = np.interp(k1,nodes1,g0)
	#define C_t+1 using interpolated k2 (=k'')
	cons1 = (1.-delta+alpha*k1**(alpha-1.))*k1+(1.-alpha)*k1**alpha-k2
	#define Euler equation to be solved
	euler = cons1-beta*(1.+alpha*k1**(alpha-1.)-delta)*cons
	return(euler)

#start time iteration algorithm
g0 = np.linspace(0.5*kStar,1.5*kStar,m)
check = 1.
while check>epsilon:
	counter += 1
	print('Iteration ', counter)
	gnew = []
	for i in range(len(nodes1)):
		k = nodes1[i]
		gnew.append(fsolve(func41,nodes1[i]))
	gnew = [i[0] for i in gnew] 
	#determine infty-norm
	check = np.max(np.abs(np.asarray(g0)-np.asarray(gnew)))
	g0 = gnew

#calculate chebyceff polynomials
cHat = [(1.-delta+alpha*nodes1[i]**(alpha-1.))*nodes1[i]+(1.-alpha)*nodes1[i]**alpha-g0[i] for i in range(len(nodes1))]
coefs = []
for i in range(n+1):
	auxT = T[i,:]
	coefs.append(np.sum(cHat*auxT)/(np.sum(auxT**2.)))
	
#do plotting on 100 equidistant points
kVals = np.linspace(0.5*kStar,1.5*kStar,100)
auxFunc = lambda x: np.polynomial.chebyshev.chebvander(-1+2.*(x-0.5*kStar)/(kStar),n)
cVals = [np.sum(np.asarray(coefs)*auxFunc(i)) for i in kVals]
plt.plot(kVals,cVals,'r*')
plt.plot(kVals,cVals,'b-')
plt.xlabel(r'$k$')
plt.ylabel(r'$C(k)$')
plt.title(r'$\tilde{C}(K)$ using Chebycheff Polynomials')
#plt.show()
plt.savefig('Ex4_consumption.pdf')
#####################


#simulation of K over t periods
t = 100
K = [0.5*kStar]

for i in range(0,t):
	    auxC = np.dot(np.asarray(coefs).transpose(),np.polynomial.chebyshev.chebvander(-1.+2.*(K[i]-0.5*kStar)/(kStar),n)[0]) 
	    auxK = (1.+alpha*K[i]**(alpha-1.)-delta)*K[i]+(1.-alpha)*K[i]**(alpha) - auxC
	    K.append(auxK)

#do plotting
plt.plot(range(t+1),K,'r*')
plt.plot(range(t+1),K,'b--')
plt.xlabel(r'$t$')
plt.ylabel(r'$K_t$')
plt.title(r'path of capital')
#plt.show()
plt.savefig('Ex4_capitalpath.pdf')


#####################################################################################################################
#####################################################################################################################


#def fxn():
#    warnings.warn("deprecated", DeprecationWarning)

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#	fxn()

#fxn()

def intrate(k):
	return alpha*(k**(alpha-1))

def wage(k):
	return (1-alpha)*k**(alpha)

def c_t(k,k_pol):
	return (1+intrate(k)-delta)*k + wage(k) - k_pol

def foc(k_pol,k):
	k_int = gp.predict(np.atleast_2d(k_pol), return_std=False) #
	#k_int = np.interp(k_pol, kgrid, k_sols)
	return beta*(1+intrate(k_pol) - delta)*c_t(k,k_pol) - c_t(k_pol,k_int)

np.random.seed(1)
rng = np.random.RandomState(0)
#kgrid = rng.uniform(0.45*kStar, 1.2*kStar, 25)#[:, np.newaxis]
kgrid = np.linspace(0.5*kStar, 1.4*kStar, 30)
k_sols = np.zeros_like(kgrid)+kgrid[8]
k_solint = np.zeros_like(kgrid)
status_collect = np.ones_like(kgrid)
X = np.atleast_2d(kgrid).T
y = k_sols.ravel()
#dy = 0.5 + 1.0 * np.random.random(y.shape)
#noise = np.random.normal(0, dy)
#y += noise

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e-3)) * RBF(10, (1e-3, 1e2))
gp = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=9)


# Fit to data using Maximum Likelihood Estimation of the parameters
perror = 20
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	gp.fit(X, k_sols)

status_collect[15] = 2.0
while (perror>5e-4):
	controls = np.where( status_collect > 1.0 )
	cons_pos = c_t(kgrid,k_sols)
	cons_controls = np.where( cons_pos < 0.0 )
	joined =  cons_controls + controls
	joint = [x for xs in joined for x in xs]
	print(joint)
	if any(joint):  #[0]
		print("not empty")
		kgrid = np.delete(kgrid, joint)
		k_sols = np.delete(k_sols, joint)
		status_collect = np.delete(status_collect, joint)
		k_solint = np.delete(k_solint, joint)
		X = np.atleast_2d(kgrid).T

	for i,k in enumerate(kgrid):
		sol = root(foc, kStar, kgrid[i],method = 'hybr')
		k_sols[i] = sol.x
		status_collect[i] = sol.status
		k_solint[i] =  gp.predict(np.atleast_2d(kgrid[i]),return_std=False)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		gp.fit(X, k_sols)

	perror = LA.norm(k_sols-k_solint)
	print(perror)


status_collect[20] = 2
controlliamo = np.where( status_collect > 1.0 )

print(controlliamo)
new_stat_collect = np.delete(status_collect, controlliamo)
print(np.where( new_stat_collect > 0.0 ))




'''

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, k_sols)
for i,k in enumerate(kgrid):
	k_sols[i] = fsolve(foc, kgrid[i], kgrid[i])


print(k_sols)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, k_sols)
for i,k in enumerate(kgrid):
	k_sols[i] = fsolve(foc, kgrid[i], kgrid[i])


print(k_sols)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, k_sols)
for i,k in enumerate(kgrid):
	k_sols[i] = fsolve(foc, kgrid[i], kgrid[i])


print(k_sols)



'''

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0.5*kStar, 1.4*kStar, 1000)).T


# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
#plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, k_sols, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.plot(x,x,'g-', label=u'diagonal')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


