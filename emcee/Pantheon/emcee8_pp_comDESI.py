import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from astropy.cosmology import LambdaCDM
import numpy as np
import emcee
from multiprocessing import Pool
from scipy.interpolate import interp1d
# import camb
camb_mod_path = "/home/camilo/cocoa/Cocoa/external_modules/code/CAMB_GammaPrime_Growth"
sys.path.insert(0, camb_mod_path)
import camb
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
# import my module
sys.path.append(os.path.expanduser('~'))
from sne_functions import c
import sne_functions as sne

nmod = 0 #628
ndim = 9 ; nwalkers = 60 ; nsteps=30000 ; save_interval = 5000 #50
path_savefile = "/share/storage3/bets/camilo_storage3/Pantheon/"
savefilename = 'emcee9_pp_semPV_comDESI.h5'

mvecc, zvec_CMB, zvec_HEL, rdirs, cov_pan_fil = sne.Pantheon_data()

def logprior(theta):
    Ok, Om, Ob, sig8, h0, M, gamma, sigv, ns = theta
    OL = 1-Om-Ok
    #################### priors
    if not (4*Ok**3 + 27*OL*Om**2 > 0): return -np.inf

    if not (-0.6<Ok<0.6)    : return -np.inf
    if not (0<OL<1.5)       : return -np.inf
    if not (Ob<Om<1.5)      : return -np.inf
    if not (0.005<Ob<0.2)   : return -np.inf
    if not (0<sig8<2)       : return -np.inf
    if not (0.4<h0<1)       : return -np.inf
    #if not (-21<Mag<-18)    : return -np.inf
    if not (22<M<26)        : return -np.inf
    if not (-1<gamma<3)     : return -np.inf
    if not (125<sigv<325)   : return -np.inf
    if not (0<sig8<2)   : return -np.inf
    if not (0.8<ns<1.2)     : return -np.inf

    #### prior BBN
    prior_bbn = ((Ob*h0*h0 - 0.02196) / 0.00063)**2
    return - 0.5 * prior_bbn

cov_tot = cov_pan_fil
inv_cov_tot = np.linalg.inv(cov_tot)
signo, log_det_tot = np.linalg.slogdet(cov_tot)
inv_L_tot_fil = np.linalg.cholesky(inv_cov_tot)

def logprob_8(theta):
    Ok, Om, Ob, sig8, h0, M, gamma, sigv, ns = theta
    OL = 1-Om-Ok
    #################### theory - model (Astropy)
    cosmo = LambdaCDM(H0=h0*100, Om0=Om, Ode0=OL, Ob0=Ob)
    lum_dis = cosmo.comoving_transverse_distance(zvec_CMB).value*h0*100/c*(1+zvec_HEL)
    vecdif = mvecc - 5*np.log10(lum_dis) - M

    mahalanobis = np.linalg.multi_dot([vecdif,inv_L_tot_fil,inv_L_tot_fil.T,vecdif])
    return -0.5 * mahalanobis - 0.5 * log_det_tot


# Define fs8 data
fs8_data = {'z':[0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS','LRG1','LRG2','LRG3','ELG2','QSO']}
redshifts = fs8_data['z']

def chi2_fs8(theta_fs8):
    Ok, Om, Ob, sig8, h0, M, gamma, sigv, ns = theta_fs8
    OL = 1-Om-Ok
    H0 = h0 * 100
    Ombh2 = Ob * h0**2
    Ocdmh2 = (Om - Ob) * h0**2
    # Set basic cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Ocdmh2, omk=Ok, gamma0=gamma, gamma1=0.0)
    pars.InitPower.set_params(As=2e-9, ns=ns)

    z_all = np.concatenate([[0.0], redshifts])
    sort_idx = np.argsort(z_all)[::-1]
    z_sorted = z_all[sort_idx]
    pars.set_matter_power(redshifts=z_sorted, kmax=10.0)
    # Run CAMB
    results = camb.get_results(pars)
    sigma8 = results.get_sigma8()[0]
    As= 2e-9*(sig8/sigma8)**2


    # --- Compute growth factor D(z) for f ---
    # --- 1. Create a z grid ---
    z_dense = np.linspace(0, max(z_all), 500)

    # --- 2. Omega_m(z)  ---
    Ez2_dense = np.sqrt(Om*(1+z_dense)**3 + Ok*(1+z_dense)**2 + OL)
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
    sigma8_0 = sig8  # sigma8 at z=0 from CAMB
    sigma8_dense = sigma8_0 * D_dense

    # --- 7. fsigma8 ---
    fsigma8_dense = f_dense * sigma8_dense

    # --- 8. Interpolate ---
    #interp_sigma8 = interp1d(z_dense, sigma8_dense, kind='cubic')
    #interp_f = interp1d(z_dense, f_dense, kind='cubic')
    interp_fsigma8 = interp1d(z_dense, fsigma8_dense, kind='cubic')
    #interp_Om = interp1d(z_dense, Omega_m_dense, kind='cubic')

    #sigma8_final = interp_sigma8(z_all)
    #f_final = interp_f(z_all)
    fs8 = interp_fsigma8(redshifts)
    chi2 = np.sum(((fs8_data['fsig8'] - fs8) / fs8_data['fsig8_err'])**2)
    return chi2

def logprob(theta):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logprob_8(theta) + (-0.5 * chi2_fs8(theta))


p0 = [[ 0.00 + 0.05 * np.random.randn(),
        0.35 + 0.02 * np.random.randn(),
        0.042 + 0.002 * np.random.randn(),
        0.8 + 0.3 * np.random.uniform(low=-1, high=1),
        0.73 + 0.01 * np.random.uniform(low=-1, high=1),
        23.81 + 0.5 * np.random.randn(),
        0.5 + 0.4 * np.random.uniform(low=-1, high=1),
        240 + 30 * np.random.randn(),
        0.965 + 0.05 * np.random.uniform(low=-1, high=1)
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