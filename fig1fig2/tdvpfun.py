import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import os

from functools import partial

### for spin 1/2 only

def _eta_from_narr(narr):
	"""Return the cyclic environment weights from ``narr`` in O(K) work.

	``narr`` may have shape ``(K,)`` or ``(K, n_times)``.  The original
	implementation explicitly formed every cyclic product and therefore used
	O(K^2) memory and work.  The same definition obeys the recurrence

	    eta[i] = 1 + narr[i-1] * eta[i-1],

	with the periodic value of ``eta[0]`` fixed by one backward product around
	the cell.  This form is algebraically identical and makes a high-resolution
	parameter scan practical.
	"""

	narr = np.asarray(narr)
	K = narr.shape[0]
	beta = np.prod(narr, axis=0)
	backward_indices = (-np.arange(1, K + 1)) % K
	backward_products = np.cumprod(narr[backward_indices], axis=0)
	eta = np.empty_like(narr, dtype=np.result_type(narr, np.float64))
	eta[0] = 1.0 + np.sum(backward_products, axis=0) / (1.0 - beta)
	for site in range(1, K):
		eta[site] = 1.0 + narr[site - 1] * eta[site - 1]
	return eta


def get_eta(theta):
	"""Return spin-1/2 environment weights for ``theta`` of shape ``(K, ...)``."""

	narr = -np.sin(np.asarray(theta) / 2.0) ** 2
	return _eta_from_narr(narr)


def deri_thetaphi(theta,phi):
	"""get the jocobian of thetaphi"""
	K = theta.size
	zerodiv_eps = 1e-5
	theta = np.where(theta==0,theta+zerodiv_eps,theta)
	phi = np.where(theta==0,phi+np.pi,phi)

	theta_add1 = theta[np.arange(1,K+1)%K]
	theta_add2 = theta[np.arange(2,K+2)%K]
	theta_n1 = theta[np.arange(-1,K-1)%K]
	phi_add1 = phi[np.arange(1,K+1)%K]
	phi_n1 = phi[np.arange(-1,K-1)%K]
	narr =  -np.sin(theta/2)**2 # -1+np.cos(theta/2)**(4*J)
	beta = np.prod(narr)

	ii,jj = np.meshgrid(np.arange(K),np.arange(K),indexing='ij')
	eta = 1+np.sum(np.cumprod(narr[(ii-jj-1)%K],axis=1),axis=1)/(1-beta)
	eta_n1 = eta[np.arange(-1,K-1)%K]
	eta_add1 = eta[np.arange(1,K+1)%K]

	# for i=i0: (i-1),(i-1)*(i-1),...,(i-1)\cdots(i-K)
	#  for j=1, we should sum (i-1)\cdots(i-(i-1)) to (i-1)\cdots(i-i)(i-(i+1))\cdots(i-K)
	# so we can reverse the order after cumprod and then cumsum !!
	# you can use j=i-1 case to check it
	d_onsite = 1/np.tan(theta/2)
	temp = np.cumprod(narr[(ii-jj-1)%K],axis=1)
	numerator_deri=np.cumsum(temp[:,::-1],axis=1)[np.arange(K).reshape(K,1),(jj-ii)%K]*d_onsite[ii]
	d_eta=(numerator_deri*(1-beta) + np.sum(temp,axis=1)[ii]*beta*d_onsite[ii])/(1-beta)**2
	d_eta_n1 = d_eta[np.arange(-1,K-1)%K]


	cond1 = np.where((ii+1-jj)%K==0,1,0)
	cond2 = np.where((ii+2-jj)%K==0,1,0)
	condn1 = np.where((ii-1-jj)%K==0,1,0)
	cond = np.where((ii-jj)%K==0,1,0)

	# calculate \partial_theta_j theta_i
	etaterm_deri = (d_eta_n1*eta[ii]-eta_n1[ii]*d_eta)/eta[ii]**2 # calculate derivative of eta_n1/eta
	dtheta_theta = -np.sin(phi[ii])*np.sin(theta[jj]/2)*cond1
	dtheta_theta +=  eta_n1[ii]/(eta[ii])*np.sin(phi_n1[ii])*(np.cos(theta[jj]/2)*cond*np.sin(theta_n1[ii])*(1/2)+ np.sin(
		theta[ii]/2)*np.cos(theta[jj])*condn1) + np.sin(theta[ii]/2)*np.sin(phi_n1[ii])*np.sin(theta_n1[ii])*etaterm_deri

	# calculate \partial_phi_j theta_i
	dtheta_phi = 2*np.cos(phi[jj])*cond*np.cos(theta_add1[ii]/2)
	dtheta_phi +=  eta_n1[ii]/(eta[ii])*np.cos(phi[jj])*condn1*np.sin(theta[ii]/2)*np.sin(theta_n1[ii])

	# calculate \partial_theta_j \phi_i
	dphi_theta = -np.sin(theta[jj]/2)*cond1*np.cos(phi[ii])/np.tan(theta[ii]) + 2*np.cos(theta_add1[ii]/2)*np.cos(
		phi[ii])*(-1/np.sin(theta[ii])**2)*cond
	dphi_theta +=  - np.cos(phi_add1[ii])*(-np.sin(theta[jj]/2)*(1/2)*cond2*np.tan(theta_add1[ii]/2)+np.cos(
		theta_add2[ii]/2)/np.cos(theta[jj]/2)**2*(1/2)*cond1)
	dphi_theta += - eta_n1[ii]/(2*eta[ii])*np.cos(phi_n1[ii])*(np.sin(theta[jj]/2)/(np.cos(theta[jj]/2)**2)*(1/2)*cond*np.sin(theta_n1[ii])+1/np.cos(
		theta[ii]/2)*np.cos(theta[jj])*condn1) - np.cos(phi_n1[ii])*np.sin(theta_n1[ii])/(2*np.cos(theta[ii]/2))*etaterm_deri
	dphi_theta += - np.cos(phi[ii])*eta[ii]/(2*eta_add1[ii])*(np.cos(theta[jj])*cond*np.sin(theta_add1[ii]/2)*np.tan(theta_add1[ii]/2)+np.sin(
		theta[ii])*(1/4)*(3+np.cos(theta[jj]))*np.sin(theta[jj]/2)/(np.cos(theta[jj]/2)**2)*cond1) -np.cos(
			phi[ii])*np.sin(theta[ii])*np.sin(theta_add1[ii]/2)*np.tan(theta_add1[ii]/2)*(1/2)*etaterm_deri[np.arange(1,K+1)%K]

	# calculate \partial_phi_j phi_i
	dphi_phi = -2*np.cos(theta_add1[ii]/2)*np.sin(phi[ii])/np.tan(theta[ii])
	dphi_phi += np.sin(phi[jj])*condn1*np.sin(theta_n1[ii])*eta_n1[ii]/(2*eta[ii]*np.cos(theta[ii]/2))
	dphi_phi += np.sin(phi[jj])*cond*np.sin(theta[ii])*np.sin(theta_add1[ii]/2)*np.tan(theta_add1[ii]/2)*eta[ii]/(2*eta_add1[ii])
	dphi_phi += np.cos(theta_add2[ii])*np.sin(phi[jj])*cond1*np.tan(theta_add1[ii]/2)

	return dtheta_theta, dtheta_phi, dphi_theta, dphi_phi



def get_qleak(thetaphi):
	"""return sqrt(Gamma^2)"""
	K = round(thetaphi.shape[0]/2)
	theta = thetaphi[:K]
	eta = get_eta(theta)
	eta_add1 = eta[np.arange(1,K+1)%K]
	theta_add1 = theta[np.arange(1,K+1)%K]

	qleak = np.sqrt(np.abs(np.sum(np.sin(theta/2)**2*np.sin(theta_add1/2)**2*eta*(1-eta)/eta_add1,axis=0)*(1/K)))
	return qleak



def eom(t,y,mu,chi):
	"""this function is defined for solve_ivp"""

	K = round(y.size/2)
	theta = y[:K]
	phi = y[K:]
	# theta = np.where(theta==0,theta+zerodiv_eps,theta)
	# phi = np.where(theta==0,phi+np.pi,phi)
	# theta = np.where(np.cos(theta)==1,np.arccos(np.cos(theta+zerodiv_eps)),np.arccos(np.cos(theta)))
	# phi = np.where(np.cos(theta)==1,(phi+np.pi)%(2*np.pi),phi%(2*np.pi))

	eta = get_eta(theta)

	#eta = get_eta(theta)
	eta_n1 = eta[np.arange(-1,K-1)%K]
	eta_add1 = eta[np.arange(1,K+1)%K]
	theta_add1 = theta[np.arange(1,K+1)%K]
	theta_add2 = theta[np.arange(2,K+2)%K]
	theta_n1 = theta[np.arange(-1,K-1)%K]
	phi_add1 = phi[np.arange(1,K+1)%K]
	phi_n1 = phi[np.arange(-1,K-1)%K]

	Delta = np.array([2*mu-chi,2*mu+chi]*round(K/2))
	dtheta = (2*np.cos(theta_add1/2)*np.sin(phi) + eta_n1*np.sin(theta_n1)*np.sin(theta/2)*np.sin(phi_n1) / eta)

	dphi = ( 2*np.cos(theta_add1/2)*np.cos(phi)/np.tan(theta) + 2*Delta -np.cos(
			theta_add2/2)*np.cos(phi_add1)*np.tan(theta_add1/2) - eta_n1*np.sin(
			theta_n1)*np.cos(phi_n1) / (2*eta*np.cos(theta/2)) - eta*np.sin(
			theta)*np.cos(phi)*np.sin(theta_add1/2)* np.tan(theta_add1/2)/ (2*eta_add1) )

	return np.hstack((dtheta,dphi))


def jac(t,y):
	K = round(y.shape[0]/2)
	theta = y[:K]
	phi = y[K:]
	# theta = np.where(theta==0,theta+zerodiv_eps,theta)
	# phi = np.where(theta==0,phi+np.pi,phi)
	zerodiv_eps = 1e-5
	theta = np.where(np.cos(theta)==1,np.arccos(np.cos(theta+zerodiv_eps)),np.arccos(np.cos(theta)))
	phi = np.where(np.cos(theta)==1,(phi+np.pi)%(2*np.pi),phi%(2*np.pi))

	narr =  -np.sin(theta/2)**2 # -1+np.cos(theta/2)**(4*J)
	beta = np.prod(narr,axis=0) # return shape (t_num,)

	# use advanced indexing!!!
	#ii,jj = np.meshgrid(np.arange(K),np.arange(K),indexing='ij')
	row_vec = np.arange(K)[:,None]
	col_vec = np.arange(K)
	narr_cumprod = np.cumprod(narr[(row_vec-col_vec-1)%K],axis=1) # (ii-jj-1)%K, return shape (K,K,t_num)
	eta = 1 + np.sum(narr_cumprod,axis=1)/(1-beta)
	#eta = 1+np.sum(np.cumprod(narr[(ii-jj-1)%K],axis=1),axis=1)/(1-beta)


	# for i=i0: (i-1),(i-1)*(i-1),...,(i-1)\cdots(i-K)
	#  for j=1, we should sum (i-1)\cdots(i-(i-1)) to (i-1)\cdots(i-i)(i-(i+1))\cdots(i-K)
	# so we can reverse the order after cumprod and then cumsum !!
	# you can use j=i-1 case to check it
	d_onsite = 1/np.tan(theta/2)
	row_vec_2D = row_vec + np.zeros(K,dtype=np.intp)
	d_onsite_2D = d_onsite[row_vec_2D]
	# for concatence, d_onsite is of shape (1,K),
	# we need use d_onsite[row_vec_2D] such that multiply d_onsite at each time with former factor (K,K,t_num)
	numerator_deri=np.cumsum(narr_cumprod[:,::-1],axis=1)[row_vec_2D,(col_vec-row_vec)%K]*d_onsite_2D
	#numerator_deri=np.cumsum(narr_cumprod[:,::-1],axis=1)[np.arange(K).reshape(K,1),(jj-ii)%K]*d_onsite[ii]
	d_eta=(numerator_deri*(1-beta) + np.sum(narr_cumprod,axis=1)[row_vec_2D]*beta*d_onsite_2D)/(1-beta)**2
	d_eta_n1 = d_eta[np.arange(-1,K-1)%K]

	cond1 = np.where((row_vec+1-col_vec)%K==0,1,0) # (ii+1-jj)%K
	cond2 = np.where((row_vec+2-col_vec)%K==0,1,0) # (ii+1-jj)%K
	condn1 = np.where((row_vec-1-col_vec)%K==0,1,0) # (ii+1-jj)%K
	cond = np.where((row_vec-col_vec)%K==0,1,0) # (ii+1-jj)%K
	# cond1 = np.repeat(np.where((row_vec+1-col_vec)%K==0,1,0)[:,:,None],y.shape[1],axis=2) # (ii+1-jj)%K
	# cond2 = np.repeat(np.where((row_vec+2-col_vec)%K==0,1,0)[:,:,None],y.shape[1],axis=2)
	# condn1 = np.repeat(np.where((row_vec-1-col_vec)%K==0,1,0)[:,:,None],y.shape[1],axis=2)
	# cond = np.repeat(np.where((row_vec-col_vec)%K==0,1,0)[:,:,None],y.shape[1],axis=2)

	eta_2D = eta[row_vec_2D]
	eta_add1_2D = eta_2D[np.arange(1,K+1)%K]
	eta_n1_2D = eta_2D[np.arange(-1,K-1)%K]

	theta_2D = theta[row_vec_2D]
	theta_add1_2D = theta_2D[np.arange(1,K+1)%K]
	theta_add2_2D = theta_2D[np.arange(2,K+2)%K]
	theta_n1_2D = theta_2D[np.arange(-1,K-1)%K]
	phi_2D = phi[row_vec_2D]
	phi_n1_2D = phi_2D[np.arange(-1,K-1)%K]
	phi_add1_2D = phi_2D[np.arange(1,K+1)%K]


	col_vec_2D = col_vec + np.zeros(K,dtype=np.intp)[:,None]
	theta_2D_col = theta[col_vec_2D]
	phi_2D_col = phi[col_vec_2D]


	# calculate \partial_theta_j theta_i
	etaterm_deri = (d_eta_n1*eta_2D-eta_n1_2D*d_eta)/eta_2D**2 # calculate derivative of eta_n1/eta
	dtheta_theta = -np.sin(phi_2D)*np.sin(theta_2D_col/2)*cond1
	dtheta_theta +=  eta_n1_2D/(eta_2D)*np.sin(phi_n1_2D)*(np.cos(theta_2D_col/2)*cond*np.sin(theta_n1_2D)*(1/2)+ np.sin(
		theta_2D/2)*np.cos(theta_2D_col)*condn1) + np.sin(theta_2D/2)*np.sin(phi_n1_2D)*np.sin(theta_n1_2D)*etaterm_deri

	# calculate \partial_phi_j theta_i
	dtheta_phi = 2*np.cos(phi_2D_col)*cond*np.cos(theta_add1_2D/2)
	dtheta_phi +=  eta_n1_2D/(eta_2D)*np.cos(phi_2D_col)*condn1*np.sin(theta_2D/2)*np.sin(theta_n1_2D)

	# calculate \partial_theta_j \phi_i
	dphi_theta = -np.sin(theta_2D_col/2)*cond1*np.cos(phi_2D)/np.tan(theta_2D) + 2*np.cos(theta_add1_2D/2)*np.cos(
		phi_2D)*(-1/np.sin(theta_2D)**2)*cond
	dphi_theta +=  - np.cos(phi_add1_2D)*(-np.sin(theta_2D_col/2)*(1/2)*cond2*np.tan(theta_add1_2D/2)+np.cos(
		theta_add2_2D/2)/np.cos(theta_2D_col/2)**2*(1/2)*cond1)
	dphi_theta += - eta_n1_2D/(2*eta_2D)*np.cos(phi_n1_2D)*(np.sin(theta_2D_col/2)/(np.cos(theta_2D_col/2)**2)*(1/2)*cond*np.sin(theta_n1_2D)+1/np.cos(
		theta_2D/2)*np.cos(theta_2D_col)*condn1) - np.cos(phi_n1_2D)*np.sin(theta_n1_2D)/(2*np.cos(theta_2D/2))*etaterm_deri
	dphi_theta += - np.cos(phi_2D)*eta_2D/(2*eta_add1_2D)*(np.cos(theta_2D_col)*cond*np.sin(theta_add1_2D/2)*np.tan(theta_add1_2D/2)+np.sin(
		theta_2D)*(1/4)*(3+np.cos(theta_2D_col))*np.sin(theta_2D_col/2)/(np.cos(theta_2D_col/2)**2)*cond1) -np.cos(
			phi_2D)*np.sin(theta_2D)*np.sin(theta_add1_2D/2)*np.tan(theta_add1_2D/2)*(1/2)*etaterm_deri[np.arange(1,K+1)%K]

	# calculate \partial_phi_j phi_i
	dphi_phi = -2*np.cos(theta_add1_2D/2)*np.sin(phi_2D)/np.tan(theta_2D)
	dphi_phi += np.sin(phi_2D_col)*condn1*np.sin(theta_n1_2D)*eta_n1_2D/(2*eta_2D*np.cos(theta_2D/2))
	dphi_phi += np.sin(phi_2D_col)*cond*np.sin(theta_2D)*np.sin(theta_add1_2D/2)*np.tan(theta_add1_2D/2)*eta_2D/(2*eta_add1_2D)
	dphi_phi += np.cos(theta_add2_2D)*np.sin(phi_2D_col)*cond1*np.tan(theta_add1_2D/2)

	return np.vstack((np.hstack((dtheta_theta,dtheta_phi)),np.hstack((dphi_theta,dphi_phi))))



def run_qleak(mu,chi,J,N,K,t_eval,thetaphi0):
	"""get the quantum leakage rate
	"""
	theta,phi = thetaphi0[:K],thetaphi0[K:]
	yini= np.concatenate([theta,phi])
	start1=time.time()
	mint,maxt = t_eval[0], t_eval[-1]
	# RK45 BDF LSODA
	solution = solve_ivp(eom,[mint,maxt],yini,t_eval=t_eval,method = 'RK45',jac=None,args=(mu,chi),rtol=1e-3,atol = 1e-5)
	end1=time.time()
	print(f'solve_ivp time: {end1-start1}')
	thetaphi = np.array(solution.y)
	#theta,phi = get_thetaphi_from_PQ(QP[:K],QP[K:2*K],QP[2*K:3*K],QP[3*K:])
	#thetaphi = np.concatenate([theta,phi])
	deltat = np.diff(t_eval) # deltat has length t_num-1
	q_leak_teval = get_qleak((thetaphi[:,:-1]+thetaphi[:,1:])/2)
	q_leak_cum = np.cumsum(q_leak_teval*deltat)
	q_leak_rate = q_leak_cum[-1]/(maxt-mint)

	return q_leak_rate



# For parallel computing
def TDVP_qleak(J,N,K,mu_list,chi_list,t_eval):
	J=J # default 1/2
	sps = round(2*J+1)
	N,K = N,K

	#Omega = np.ones(K)
	mid_site = N//2 - 1  # 01000101

	# initial state: defect at mid_site of Z2 state
	bias = 1e-3
	theta0 = np.array([bias,np.pi-bias]*(N//2)).flatten() # z2 configuration
	theta0[mid_site] = bias # add a defect at the mid_site
	#theta0 = np.array([bias for i in range(11)]+[np.pi-bias]+[bias for i in range(12)]) # central defect
	#theta0 = np.array([np.pi-bias for i in range(round(K/2)-1)]+[np.pi-2]+[np.pi-bias for i in range(K-round(K/2))]) # central defect with pi background
	phi0 = np.array([bias for i in range(K)]).flatten()

	# use quantum leakage to characterize the valid parameter regime
	thetaphi0 = np.concatenate([theta0,phi0])

	run_qleak_fixed = partial(run_qleak, J=J, N=N, K=K, t_eval=t_eval, thetaphi0=thetaphi0)

	# parallel computing using ProcessPoolExecutor
	# q_leak_cum_rate1 = np.zeros((len(mu_list),len(chi_list)))
	# with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
	# 	futures = {}
	# 	for i,a in enumerate(mu_list):
	# 		for j,b in enumerate(chi_list):
	# 			future = executor.submit(run_qleak_fixed, a,b)
	# 			futures[future] = (i,j)

	# 	for future in tqdm(as_completed(futures), total=len(futures)):
	# 		i,j = futures[future]
	# 		q_leak_cum_rate1[i,j] = future.result()

	# parallel computing using joblib
	tasks = [(mu,chi) for mu in mu_list for chi in chi_list]
	results = Parallel(n_jobs=os.cpu_count())(delayed(run_qleak_fixed)(mu,chi) for mu,chi in tqdm(tasks))
	q_leak_cum_rate2 = np.array(results).reshape(len(mu_list),len(chi_list))

	#q_leak_cum_rate = np.array(q_leak_cum_rate).reshape(len(mu_list),len(chi_list))

	return q_leak_cum_rate2
