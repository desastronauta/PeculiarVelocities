import numpy as np
import os, glob, sys, argparse
import multiprocessing as mp
from getdist import loadMCSamples
from tqdm import tqdm
from scipy.interpolate import interp1d
import emcee
from multiprocessing import Pool
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sne_functions import c
import sne_functions as sne

ndim = 4 ; nwalkers = 60 ; nsteps=50000 ; save_interval = 10000
path_savefile = "/share/storage3/bets/camilo_storage3/Pantheon/"

# savefilename = 'emcee_onlyDESI_RSD.h5'
# extrapoint = {'z':0.07, 'fsig8':0.4497, 'fsig8_err':0.055, 'tracer':'BGS+PV'}

savefilename = 'emcee_onlyDESI_Pan.h5'
extrapoint = {'z':0.024, 'fsig8':0.31, 'fsig8_err':0.19, 'tracer':'Pantheon+'}

# Define data NOTE: the first point comes from PVs (Pantheon+, DESY5, DESI BGS+PV)

fs8_data = {'z':[extrapoint['z'],0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[extrapoint['fsig8'],0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[extrapoint['fsig8_err'],0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS+PV','BGS','LRG1','LRG2','LRG3','ELG2','QSO']}

def compute_fsigma8(theta_fs8, redshifts=fs8_data['z']):
    Om, Ok, sig8, gamma = theta_fs8

    z_all = np.concatenate([[0.0], redshifts])
    
    # --- Compute growth factor D(z) for f ---
    # --- 1. Create a z grid ---
    z_dense = np.linspace(0, max(z_all), 500)

    # --- 2. Omega_m(z) ---
    Ez2_dense = Om * (1 + z_dense)**3 + Ok * (1 + z_dense)**2 + (1 - Om - Ok)
    Omega_m_dense = Om * (1 + z_dense)**3 / Ez2_dense

    # --- 3. f(z) ---
    f_dense = Omega_m_dense**gamma

    # --- 4. Manual cumulative integral ---
    integrand = f_dense / (1 + z_dense)
    dz = np.diff(z_dense)
    I = np.zeros_like(z_dense)
    I[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)

    # --- 5. Growth factor ---
    D_dense = np.exp(-I)

    # --- 6. sigma8 ---
    sigma8_dense = sig8 * D_dense

    # --- 7. fsigma8 ---
    fsigma8_dense = f_dense * sigma8_dense

    # --- 8. Interpolate ---
    interp_fsigma8 = interp1d(z_dense, fsigma8_dense, kind='cubic')
    fs8 = interp_fsigma8(redshifts)
    return fs8

def loglike_fs8(theta_fs8):
    fs8_theory = compute_fsigma8(theta_fs8)
    chi2 = np.sum(((fs8_data['fsig8'] - fs8_theory) / fs8_data['fsig8_err'])**2)
    return -0.5 * chi2

def logprior(theta_fs8):
    Om, Ok, sig8, gamma = theta_fs8
    
    OL = 1-Om-Ok
    #################### priors
    if not (4*Ok**3 + 27*OL*Om**2 > 0): return -np.inf

    if not (-0.6<Ok<0.6)    : return -np.inf
    if not (0<OL<1)         : return -np.inf
    if not (0<Om<1.5)       : return -np.inf 
    if not (0<sig8<2)       : return -np.inf
    if not (0<gamma<3)     : return -np.inf
    return 0

def logprob(theta):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return loglike_fs8(theta) + lp

p0 = [[ 0.35 + 0.02 * np.random.randn(),
        0.0 + 0.02 * np.random.uniform(low=-1, high=1),
        0.8 + 0.4 * np.random.uniform(low=-1, high=1),
        0.55 + 0.4 * np.random.uniform(low=-1, high=1)
        ] for i in range(nwalkers)]

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)  
    # Run in chunks of save_interval steps
    initial_positions = p0
    total_chunks = nsteps // save_interval
    for chunk in range(total_chunks):
        print(f"Running chunk {chunk + 1}/{total_chunks} ({save_interval} steps)...")
        
        # Run the sampler for save_interval steps
        sampler.run_mcmc(initial_positions, save_interval, progress=True)
        
        # Save progress to HDF5 file
        print(f"Saving progress to {savefilename}...")
        sne.save_to_h5_emcee(os.path.join(path_savefile, savefilename), sampler, save_interval)

        # Update initial positions to the current state for the next chunk
        initial_positions = sampler.get_chain()[-1, :, :]  # Last position of each walker