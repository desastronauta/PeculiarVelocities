import numpy as np
import os, glob, sys
import multiprocessing as mp
from getdist import loadMCSamples
from tqdm import tqdm
from scipy.interpolate import interp1d
camb_mod_path = "/home/camilo/cocoa/Cocoa/external_modules/code/CAMB_GammaPrime_Growth"
sys.path.insert(0, camb_mod_path)
import camb
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

H0, Ombh2, Ocdmh2 = 70, 0.022, 0.12
As, ns = 2e-9, 0.96
Ok, gamma, Alens = 0.0, 0.55, 1

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Ocdmh2, omk=Ok, gamma0=gamma, gamma1=0.0)
pars.InitPower.set_params(As=As, ns=ns)

z_all = np.concatenate([[0.0], [0.295,0.510,0.706,0.930,1.317,1.491]])
sort_idx = np.argsort(z_all)[::-1]
z_sorted = z_all[sort_idx]
pars.set_matter_power(redshifts=z_sorted, kmax=10.0)

# Run CAMB
results = camb.get_results(pars)
kh, z, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)

####### getting the right sigma8
# --- Compute growth factor D(z) ---
# --- 1. Dense grid ---
z_dense = np.linspace(0, max(z_all), 500)

# --- 2. Omega_m(z) from CAMB ---
H_dense = results.hubble_parameter(z_dense)

Omega_m0 = (Ombh2 + Ocdmh2) / (H0/100.0)**2
Ez2_dense = (H_dense / H0)**2
Omega_m_dense = Omega_m0 * (1 + z_dense)**3 / Ez2_dense

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
sigma8_0 = results.get_sigma8()[-1]  # sigma8 at z=0 from CAMB
print("σ8(z=0):", sigma8_0)
sigma8_dense = sigma8_0 * D_dense

# --- 7. fsigma8 ---
fsigma8_dense = f_dense * sigma8_dense

# --- 8. Interpolate to your z_all ---
interp_sigma8 = interp1d(z_dense, sigma8_dense, kind='cubic')
interp_f = interp1d(z_dense, f_dense, kind='cubic')
interp_fsigma8 = interp1d(z_dense, fsigma8_dense, kind='cubic')
interp_Om = interp1d(z_dense, Omega_m_dense, kind='cubic')

sigma8_final = interp_sigma8(z_all)
f_final = interp_f(z_all)
fsigma8_final = interp_fsigma8(z_all)
Omega_m_final = interp_Om(z_all)
#######


print("f(z):", f_final)
print("fσ₈(z):", fsigma8_final)

data = np.column_stack((
    z_all,
    sigma8_final,
    Omega_m_final,
    f_final,
    fsigma8_final
))

np.savetxt(
    "CAMB-GammaPrime0_results.txt",
    #"CAMB-GammaPrime_Alens_results.txt",
    data,
    header="z sigma8 Omega_m f fs8"
)


np.savetxt('pklin_CAMB-GammaPrime0_z0p0.txt',np.c_[kh,pk_lin[0,:]])
np.savetxt('pklin_CAMB-GammaPrime0_z0p51.txt',np.c_[kh,pk_lin[2,:]])
np.savetxt('pklin_CAMB-GammaPrime0_z1p491.txt',np.c_[kh,pk_lin[6,:]])

# np.savetxt('pklin_CAMB-GammaPrime_Alens_z0p0.txt',np.c_[kh,pk_lin[0,:]])
# np.savetxt('pklin_CAMB-GammaPrime_Alens_z0p51.txt',np.c_[kh,pk_lin[2,:]])
# np.savetxt('pklin_CAMB-GammaPrime_Alens_z1p491.txt',np.c_[kh,pk_lin[6,:]])
