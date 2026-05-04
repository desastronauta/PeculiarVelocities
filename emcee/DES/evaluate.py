import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from astropy.cosmology import LambdaCDM
import numpy as np
import emcee, time
from multiprocessing import Pool
# import my module
sys.path.append(os.path.expanduser('~'))
import sne_functions as sne
from sne_functions import c

nmod = 243
ndim = 8 ; nwalkers = 50 ; nsteps = 15000 ; save_interval = 50
savefilename = 'emcee8_DES_semSHOES2.h5'
path_savefile = os.getcwd()

mvecc, zvec_CMB, zvec_HEL, rdirs, cov_pan_fil = sne.DESY5_data()

def logprior(theta):
    OL, Om, Ob, sig8, h0, M, gamma, sigv = theta
    Ok = 1-Om-OL
    #################### priors
    if not (4*Ok**3 + 27*OL*Om**2 > 0): return -np.inf

    if not (-0.6<Ok<0.6)    : return -np.inf
    if not (0<OL<1)         : return -np.inf
    if not (0<Om<1.5)       : return -np.inf 
    if not (0.005<Ob<0.2)   : return -np.inf
    if not (Om>Ob)          : return -np.inf
    if not (0<sig8<2)       : return -np.inf
    if not (0.4<h0<1)       : return -np.inf
    if not (20<M<26)    : return -np.inf
    if not (-1<gamma<3)     : return -np.inf
    if not (0<sigv<325)   : return -np.inf

    #### prior BBN
    prior_bbn = ((Ob*h0*h0 - 0.02196) / 0.00063)**2
    return - 0.5 * prior_bbn

def logprob_8(theta):
    OL, Om, Ob, sig8, h0, M, gamma, sigv = theta
    #################### theory - model (Astropy)
    cosmo = LambdaCDM(H0=h0*100, Om0=Om, Ode0=OL, Ob0=Ob)
    lum_dis = cosmo.comoving_transverse_distance(zvec_CMB).value*h0*100/c*(1+zvec_HEL)
    vecdif = mvecc - 5*np.log10(lum_dis) - M#(Mag-5*np.log10(100*h0/c) +25)

    cov_tot = cov_pan_fil + sne.cov_PV_mod_8(theta,zvec_CMB, rdirs,lum_dis,nmod)

    if np.all( np.linalg.eigvals(cov_tot) > 0):
        pass
    else: return -np.inf
    
    inv_cov_tot = np.linalg.inv(cov_tot)
    signo, log_det_tot = np.linalg.slogdet(cov_tot)
    inv_L_tot_fil = np.linalg.cholesky(inv_cov_tot)
    ####################

    mahalanobis = np.linalg.multi_dot([vecdif,inv_L_tot_fil,inv_L_tot_fil.T,vecdif])
    return -0.5 * mahalanobis - 0.5 * log_det_tot


def logprob(theta):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return logprob_8(theta) + lp

ombh2 = 0.022
omch2 = 0.12
H0 = 68
tau = 0.07
mnu = 0.06
nnu = 3.046
gamma = 0.55
As = 2.1e-9
ns = 0.96
Mag = -19.253
sigv = 240
sig8 = 0.8137630712881473

Om = (ombh2 + omch2)/(H0/100)**2
Ob = ombh2/(H0/100)**2
OL = 1 - Om - 0.2

# OL, Om, Ob, sig8, h, Mag, gamma, sigv = theta
theta = [OL, Om, Ob, sig8, H0/100, Mag, gamma, sigv]
# theta = [0.65,0.35,0.042,0.8,0.73,-19.253,0.55,240]
start_time = time.perf_counter()
#print('params:     ', theta)
print('likelihood: ', logprob_8(theta))
end_time = time.perf_counter()

avg_time = end_time - start_time
print(f"Average over 1 evaluations: {avg_time:.6f} seconds")