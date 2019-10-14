from scipy.stats import levy_stable
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as ln
import seaborn as sns
from math import pi
from cmath import exp
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def ecf(u,r):

	total = 0.0

	for element in r:
		total = total+exp(element*u*(1j))
	return total/len(r)

def aecf(u,r):

	return np.absolute(ecf(u,r))

def why(u,r):

	V = np.imag(ecf(u,r))
	U = np.real(ecf(u,r))
	return np.arctan2(V,U)

def intAlpBet(r):

	nuA = [2.439,2.5,2.6,2.7,2.8,3.0,3.2,3.5,4.0,5.0,6.0,8.0,10,15,25]
	nuB = [0,0.1,0.2,0.3,0.5,0.7,1]


	alphaTab=np.array([[2.000,2.000,2.000,2.000,2.000,2.000,2.000],
	[1.916,1.924,1.924,1.924,1.924,1.924,1.924],
	[1.808,1.813,1.829,1.829,1.829,1.829,1.829],
	[1.729,1.730,1.737,1.745,1.745,1.745,1.745],
	[1.664,1.663,1.663,1.668,1.676,1.676,1.676],
	[1.563,1.560,1.553,1.548,1.547,1.547,1.547],
	[1.484,1.480,1.471,1.460,1.448,1.438,1.438],
	[1.391,1.386,1.378 ,1.364 ,1.337 ,1.318 ,1.318],
	[1.279 ,1.273 ,1.266 ,1.250 ,1.210 ,1.184 ,1.150],
	[1.128 ,1.121 ,1.114 ,1.101 ,1.067 ,1.027 ,0.973],
	[1.029 ,1.021 ,1.014 ,1.004 ,0.974 , 0.935 ,0.874],
	[0.896 ,0.892 ,0.887 ,0.883 ,0.855 ,0.823 ,0.769],
	[0.818 ,0.812 ,0.806 ,0.801 ,0.780 ,0.756 ,0.691],
	[0.698 ,0.695 ,0.692 ,0.689 ,0.676 ,0.656 ,0.595],
	[0.593 ,0.590 ,0.588 ,0.586 ,0.579 ,0.563 ,0.513]]).T
	
	betaTab=np.array([[0.000, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
	[0.000, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
    [0.000, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
    [0.000, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
    [0.000, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
    [0.000, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
    [0.000, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
    [0.000, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
    [0.000, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
    [0.000, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
    [0.000, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
    [0.000, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
    [0.000, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
    [0.000, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
    [0.000, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]]).T

	Xpcts = np.percentile(r,[95,75,50,25,5]) 
	nuAlpha = (Xpcts[0] - Xpcts[4])/(Xpcts[1] - Xpcts[3])
	nuBeta = (Xpcts[0] + Xpcts[4] - 2*Xpcts[2])/(Xpcts[0] - Xpcts[4])

	if (nuAlpha < 2.4390):
		nuAlpha = 2.4391
	elif (nuAlpha > 25):
		nuAlpha = 24.999

	s = np.sign(nuBeta)

	a,b = np.meshgrid(nuA,nuB)

	alpha = interp.griddata(np.array([a.ravel(),b.ravel()]).T,alphaTab.ravel(),np.array([nuAlpha,np.absolute(nuBeta)]).T)
	beta = s*interp.griddata(np.array([a.ravel(),b.ravel()]).T,betaTab.ravel(),np.array([nuAlpha,np.absolute(nuBeta)]).T)
	if (beta>1):
		beta = 1
	elif (beta<-1):
		beta = -1

	return alpha,beta

def intGamDel(r,alpha,beta):

	alphaIndex = np.array([2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5])
	betaIndex = np.array([0,0.25,.5,0.75,1.0])

	a,b = np.meshgrid(alphaIndex,betaIndex)

	gammaTab = np.array([[1.908,1.908,1.908,1.908,1.908],
		[1.914,1.915,1.916,1.918,1.921],
		[1.921,1.922,1.927,1.936,1.947],
		[1.927,1.930,1.943,1.961,1.987],
		[1.933,1.940,1.962,1.997,2.043],
		[1.939,1.1952,1.988,2.045,2.116],
		[1.946,1.967,2.022,2.106,2.211],
		[1.955,1.984,2.067,2.188,2.333],
		[1.965,2.007,2.125,2.294,2.491],
		[1.980,2.040,2.205,2.435,2.696],
		[2.00,2.085,2.311,2.624,2.973],
		[2.040,2.149,2.461,2.886,3.356],
		[2.098,2.244,2.676,3.265,3.912],
		[2.189,2.392,3.004,3.844,4.775],
		[2.337,2.635,3.542,4.808,6.247],
		[2.588,3.073,4.534,6.636,9.144]]).T

	deltaTab = np.array([[0,0,0,0,0],
		[0,-0.017,-0.032,-0.049,-0.064],
		[0,-0.030,-0.061,-0.092,-0.123],
		[0,-0.043,-0.088,0.132,-0.179],
		[0,-0.056,-0.11,-0.17,-0.232],
		[0,-0.066,-0.134,-0.206,-0.283],
		[0,-0.075,-0.154,-0.241,-0.335],
		[0,-0.084,-0.173,-0.276,-0.39],
		[0,-0.090,-0.192,-0.31,-0.447],
		[0,-0.095,-0.208,-0.346,-0.508],
		[0,-0.098,-0.223,-0.383,-0.576],
		[0,-0.099,-0.237,-0.424,-0.652],
		[0,-0.096,-0.25,-0.469,-0.742],
		[0,-0.089,-0.262,-0.52,-0.853],
		[0,-0.078,-0.272,-0.581,-0.997],
		[0,-0.061,-0.279,-0.659,-1.198]]).T


	Xpcts = np.percentile(r,[75,50,25])

	phi3 = interp.griddata(np.array([a.ravel(),b.ravel()]).T,gammaTab.ravel(),np.array([alpha,np.absolute(beta)]).T)
	gamma = (Xpcts[0]-Xpcts[2])/phi3

	phi5 = interp.griddata(np.array([a.ravel(),b.ravel()]).T,deltaTab.ravel(),np.array([alpha,np.absolute(beta)]).T)
	s = np.sign(beta)
	epsi = Xpcts[1] + gamma*s*phi5
	delta = epsi - beta*gamma*np.tan(pi*alpha/2)

	return gamma,delta

def chooseK(alpha,N):
	alpha = max([alpha,0.3])
	alpha = min([alpha,1.9])
	N = max([N,200])
	N = min([N,1600])
	al = [1.9,1.5,1.3,1.1,0.9,0.7,0.5,0.3]
	n = [200,800,1600]

	a,b = np.meshgrid(al,n)

	Kmat = np.array([[9,9,9],
		[11,11,11],
		[22,16,14],
		[24,18,15],
		[28,22,18],
		[30,24,20],
		[86,68,56],
		[134,124,118]]).T

	K = interp.griddata(np.array([a.ravel(),b.ravel()]).T,Kmat.ravel(),np.array([alpha,N]).T)

	return int(round(K[0]))

def chooseL(alpha,N):
	alpha = max([alpha,0.3])
	alpha = min([alpha,1.9])
	N = max([N,200])
	N = min([N,1600])
	al = [1.9,1.5,1.1,0.9,0.7,0.5,0.3]
	n = [200,800,1600]

	a,b = np.meshgrid(al,n)

	Kmat = np.array([[9,10,11],
			[12,14,15],
			[16,18,17],
			[14,14,14],
			[24,16,16],
			[40,24,20],
			[70,68,66]]).T

	L = interp.griddata(np.array([a.ravel(),b.ravel()]).T,Kmat.ravel(),np.array([alpha,N]).T)

	return int(round(L[0]))


def estimate(r):

	iterate = 0
	maxiter = 5
	alphaold,betaold = intAlpBet(r)
	gamold,deltaold = intGamDel(r,alphaold,betaold)

	N = len(r)

	alpha = alphaold
	beta = betaold
	gam = gamold
	delta = deltaold

	r = [(element-delta)/gam for element in r]


	while iterate < maxiter:

		import pdb
		#pdb.set_trace()
	

		if gam ==0:
			gam = np.std(r)

		K = chooseK(alpha,N)

		test = [pi*k/25 for k in range(1,K+1)]

		X = ln(test).reshape(-1,1)
		Y = np.array([ln(-ln(aecf(t,r)**2)) for t in test]).T

		clf = linear_model.LinearRegression(fit_intercept = True)
		clf.fit(X,Y)

		alpha = clf.coef_[0]
		gamhat = clf.intercept_
		gamhat = (np.exp(gamhat)/2.0)**(1/alpha)
		gam = gam*gamhat

		r = [element/gamhat for element in r]

		alpha = np.max([alpha,0])
		alpha = np.min([alpha,2])
		beta = np.min([beta,1])

		beta = np.max([beta,-1])
		gam = np.max([gam,0])

		L = chooseL(alpha,N)

		unit = [pi*k/50 for k in range(1,L+1)]
		X1 = np.array(unit)
		X2 = np.array([np.sign(u)*np.absolute(u)**alpha for u in unit])
		X = np.array([X1,X2]).T
		#X2 = X2.reshape(-1,1)
	
		Y1 = np.array([why(u,r) for u in unit]).T

		clf.fit(X,Y1)

		beta = clf.coef_[1]/np.tan(alpha*pi/2.0)
		delta = delta + gam*clf.coef_[0]

		r = [element-clf.coef_[0] for element in r]

		print(iterate)

		iterate = iterate + 1


	alpha = np.max([alpha,0])
	alpha = np.min([alpha,2])
	beta = np.min([beta,1])
	beta = np.max([beta,-1])
	gam = np.max([gam,0])

	return [alpha, alphaold], [beta,betaold], [gam,gamold], [delta,deltaold] 




if __name__ == "__main__":


	alphas = np.linspace(0.1,2,5)
	betas = np.linspace(-1,1,2)
	errorsa = np.zeros((len(alphas),len(betas)))
	errorsb = np.zeros((len(alphas),len(betas)))

	for i in range(len(alphas)):
		alpha = alphas[i]
		for j in range(len(betas)):
			beta = betas[j]
			r = levy_stable.rvs(alpha, beta,loc =0, scale = 1, size=10000)

			k = estimate(r)

			errorsa[i][j] = np.absolute((k[0][0]) - alpha)/alpha
			errorsb[i][j] = np.absolute(k[1][0]-beta)/beta


	'''
	pd.DataFrame(errorsa,index = alphas,columns = betas).to_csv('/alpha_err.csv')
	pd.DataFrame(errorsb,index = alphas,columns = betas).to_csv('/beta_err.csv')
	'''
	

	np.savetxt('errorsa.csv',errorsa,fmt='%.4e',delimiter = ',')
	np.savetxt('errorsb.csv',errorsb,fmt='%.4e',delimiter = ',')


	
	



	
	
