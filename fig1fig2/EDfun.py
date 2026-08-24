from quspin.operators import hamiltonian
import math
import time
import numpy as np
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import os
from functools import partial
import pxpbasisS



def mpsmanifold(theta,phi,basis):
    '''find the coefficients of mps onto user's defined basis
    return an (1,num_basis) array
    '''

    num_basis = basis.Ns # total basis number
    sps = round(basis.sps)
    #j = basis.N - j - 1 # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
    #b = sps**j               # obtain mask integer b
    #occ = (s//b))%sps        # compute occupation on site j
    occarr = np.array([[(basis[i]//(sps**(basis.N - j - 1 )))%sps for j in range(basis.N)] for i in range(num_basis)])
    J = (sps-1)/2

    nb,n_site = occarr.shape # nb is basis number, n is site number
    coeffarr=np.zeros((1,nb),dtype=complex)
    K = theta.size
    if (n_site % K) != 0:
        raise  ValueError("the period should be divided by the site number")

    A_J = np.zeros((n_site,sps,2,2),dtype=complex)
    for j in range(n_site):
        A_J[j,0] = np.array([[np.cos(theta[j%K]/2)**(sps-1),0],[1,0]])
        for ms in range(1,sps):
            A_J[j,ms] = np.array([[0,np.sqrt(math.factorial(sps-1)/(math.factorial(
                sps-1-ms)*math.factorial(ms)))*np.cos(theta[j%K]/2)**(sps-1-ms)*(
                    np.sin(theta[j%K]/2)*np.exp(-1j*phi[j%K]))**(ms)],[0,0]])

    for i in range(nb):
        resulti =  np.eye(2) # Aup (Adn) is of size 2 by 2
        for j in range(n_site):
            # define site matrix of mps
            #Adn=np.array([[np.cos(theta[j%K]/2),0],[1,0]])
            #Aup=np.array([[0,np.exp(-1j*phi[j%K])*np.sin(theta[j%K]/2)],[0,0]])
            #A=np.array([Adn,Aup])
            resulti = np.dot(resulti,A_J[j,occarr[i,j]])
            #resulti_temp = np.dot(resulti_temp,A[occarr[i,j]])
        coeffarr[0,i]=np.trace(resulti) # calculate the coefficient of mps on the basis

    coeffnorm = np.linalg.norm(coeffarr)
    #print(coeffnorm)
    #if abs(coeffnorm-0)<= 1e-10:
    #    raise ValueError("the normalization of mps is problematic")

    return coeffarr/coeffnorm



def run_width(mu, chi,J,N,K,psi,t_eval):

    # to avoid pickle error in ProcessPoolExecutor, basis is constructed explicitly
    J,sps=J,round(2*J+1) # default 1/2
    N,K = N,K
    kblock,pblock = None, None
    basis = pxpbasisS.constrained_basis(sps,N,kblock,pblock)

    # mu, chi,N,K,basis,psi,z_list, zz_cor_list= args
    Delta = np.array([2*mu-chi,2*mu+chi]*(K//2)+[2*mu-chi]* (K%2) )  # period K
    # Note we normalize the Hamiltonian by 1/J to accomodate with changing J
    Omega = np.ones(K)
    Omega_list  = [[Omega[i%K]/(2*J),i] for i in range(N)]
    z_static_list = [[Delta[i%K]/J,i] for i in range(N)]

    # operator string lists
    static = [["+",Omega_list],["-",Omega_list],["z",z_static_list]]
    dynamic = []

    # compute Hamiltonian, no checks
    no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
    H = hamiltonian(static,dynamic,basis=basis,**no_checks)

    z_list = [hamiltonian([["z",[[1.0/J,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(N)]

    # construct zz correlation (1/J^2) \ave{z_j z_{mid}} for j in range(M-L/4,M+L/4)
    mid_site = 2*((N+2)//4)-1
    zz_cor_list = [hamiltonian([["zz",[[1.0/(J*J),(i)%N, mid_site]]]],[],basis=basis,dtype=np.float64,**no_checks)
                    for i in range(N)]

    starttime=time.time()
    psi_t = H.evolve(psi.flatten(),t_eval[0],t_eval)
    endtime=time.time()
    print(f'evolve time: {endtime-starttime} for mu={mu}, chi={chi}')
    #psi_t = H.evolve(psi,t_eval[0],t_eval,iterate=True) # use "yield" to return an object
    #psi_t = H.evolve(psi.flatten(),t_eval[0],t_eval)
    z_t = np.vstack([z.expt_value(psi_t).real for z in z_list])

    # connected correlator dimension (N,t_num)
    zz_t = np.vstack([zz.expt_value(psi_t).real for zz in zz_cor_list])

    #zz_t = zz_t - z_t*z_t[np.arange(1,N+1)%N]
    zz_t = zz_t - z_t*z_t[mid_site]
    # width of the correlator
    #zz_ave = (np.sum(np.abs(zz_t).T*(np.arange(N)),axis=1)/np.sum(np.abs(zz_t),axis=0)).flatten()
    # use mid_site rather than zz_ave to define the center
    width_zz = np.sqrt(np.sum(np.abs(zz_t).T*(np.arange(N)-mid_site)**2,axis=1)/np.sum(np.abs(zz_t),axis=0)).flatten()

    # average over the last 10 time steps
    #width_zz_fin = np.mean(width_zz[-10:])

    return width_zz



def ED_width(J,N,K,mu_list,chi_list,t_eval):
    """mu is the mass term and chi is the confinement potential
    """

    J=J # default 1/2
    sps = round(2*J+1)

    N,K = N,K

    kblock = None
    pblock = None

    basis = pxpbasisS.constrained_basis(sps,N,kblock,pblock)

    #print(f"translation kblock = {kblock}, parity pblock = {pblock}")

    # initial state: defect at mid_site of Z2 state
    bias = 1e-3
    theta0 = np.array([bias,np.pi-bias]*(N//2)).flatten() # z2 configuration
    mid_site = 2*((N+2)//4)-1
    theta0[mid_site] = bias # add a defect at the mid_site
    phi0 = np.array([bias for i in range(K)]).flatten()

    psi = mpsmanifold(theta0,phi0,basis) # return a row vector

    # the notation is following https://arxiv.org/pdf/2301.07717
    #width_zz = np.zeros((len(mu_list),len(chi_list),t_num))

    run_width_fixed = partial(run_width,J=J,N=N,K=K,psi=psi,t_eval=t_eval)

    # parallel computing using ProcessPoolExecutor
    width_zz_fin1 = np.zeros((len(mu_list),len(chi_list),len(t_eval)))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for i,a in enumerate(mu_list):
            for j,b in enumerate(chi_list):
                future = executor.submit(run_width_fixed, a,b)
                futures[future] = (i,j)

        for future in tqdm(as_completed(futures), total=len(futures)):
            i,j = futures[future]
            width_zz_fin1[i,j] = future.result()

    # parallel computing using joblib
    # tasks = [(mu,chi) for mu in mu_list for chi in chi_list]
    # results = Parallel(n_jobs=os.cpu_count())(delayed(run_width_fixed)(mu,chi) for mu,chi in tqdm(tasks))
    # width_zz_fin2 = np.array(results).reshape(len(mu_list),len(chi_list))

    return  width_zz_fin1


def ED_width_selectedpoints(J,N,K,muchi_list,t_eval):
    """mu is the mass term and chi is the confinement potential
    """

    J=J # default 1/2
    sps = round(2*J+1)

    N,K = N,K

    kblock = None
    pblock = None

    basis = pxpbasisS.constrained_basis(sps,N,kblock,pblock)

    #print(f"translation kblock = {kblock}, parity pblock = {pblock}")

    # initial state: defect at mid_site of Z2 state
    bias = 1e-3
    theta0 = np.array([bias,np.pi-bias]*(N//2)).flatten() # z2 configuration
    mid_site = 2*((N+2)//4)-1
    theta0[mid_site] = bias # add a defect at the mid_site
    phi0 = np.array([bias for i in range(K)]).flatten()

    psi = mpsmanifold(theta0,phi0,basis) # return a row vector

    # the notation is following https://arxiv.org/pdf/2301.07717
    #width_zz = np.zeros((len(mu_list),len(chi_list),t_num))

    run_width_fixed = partial(run_width,J=J,N=N,K=K,psi=psi,t_eval=t_eval)

    # parallel computing using ProcessPoolExecutor
    width_zz_fin1 = np.zeros((muchi_list.shape[0],len(t_eval)))
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(muchi_list.shape[0]):
            a = muchi_list[i,0]
            b = muchi_list[i,1]
            future = executor.submit(run_width_fixed, a,b)
            futures[future] = (i)

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            width_zz_fin1[i] = future.result()

    # parallel computing using joblib
    # tasks = [(mu,chi) for mu in mu_list for chi in chi_list]
    # results = Parallel(n_jobs=os.cpu_count())(delayed(run_width_fixed)(mu,chi) for mu,chi in tqdm(tasks))
    # width_zz_fin2 = np.array(results).reshape(len(mu_list),len(chi_list))

    return  width_zz_fin1
