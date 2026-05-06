import numpy as np
import os, glob, sys, argparse
import multiprocessing as mp
from getdist import loadMCSamples
from tqdm import tqdm
from scipy.interpolate import interp1d

# Define data
fs8_data = {'z':[0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS','LRG1','LRG2','LRG3','ELG2','QSO']}

def compute_fsigma8(theta_fs8, redshifts=fs8_data['z']):
    Om, Ok, sig8, gamma = theta_fs8

    z_all = np.concatenate([[0.0], redshifts])
    
    # --- Compute growth factor D(z) for f ---
    # --- 1. Create a z grid ---
    z_dense = np.linspace(0, max(z_all), 500)
    #z_dense = np.linspace(0, 1.5, 5000)

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

def chi2_fs8(theta_fs8):
    fs8_theory = compute_fsigma8(theta_fs8)
    chi2 = np.sum(((fs8_data['fsig8'] - fs8_theory) / fs8_data['fsig8_err'])**2)
    return chi2


Om, Ok, sig8, gamma = 0.32, 0.0, 0.8, 0.55
chi2 = chi2_fs8([Om, Ok, sig8, gamma])
print(chi2)         # 4.08381278537722
print(-0.5*chi2)    #-2.04190639268861

