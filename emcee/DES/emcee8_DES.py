import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from astropy.cosmology import LambdaCDM
import numpy as np
import emcee
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


p0 = [[ 0.6 + 0.07 * np.random.randn(),
        0.3 + 0.05 * np.random.randn(),
        0.042 + 0.002 * np.random.randn(),
        0.8 + 0.3 * np.random.uniform(low=-1, high=1),
        0.73 + 0.01 * np.random.uniform(low=-1, high=1),
        23.8 + 0.5 * np.random.randn(),
        0.5 + 0.4 * np.random.uniform(low=-1, high=1),
        150 + 100 * np.random.uniform(low=-1, high=1)
        ] for i in range(nwalkers)]

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)
    # Run in chunks of save_interval steps
    total_chunks = nsteps // save_interval
    initial_positions = p0
    for chunk in range(total_chunks):
        print(f"Running chunk {chunk + 1}/{total_chunks} ({save_interval} steps)...")
        
        # Run the sampler for save_interval steps
        sampler.run_mcmc(initial_positions, save_interval, progress=True)

        # Save progress to HDF5 file
        print(f"Saving progress to {savefilename}...")
        sne.save_to_h5_emcee(os.path.join(path_savefile, savefilename), sampler, save_interval)

        # Update initial positions to the current state for the next chunk
        initial_positions = sampler.get_chain()[-1, :, :]  # Last position of each walker