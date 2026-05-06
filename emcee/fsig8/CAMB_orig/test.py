import numpy as np
import os, glob, sys
import multiprocessing as mp
from getdist import loadMCSamples
from tqdm import tqdm
import camb


print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

H0, Ombh2, Ocdmh2 = 70, 0.022, 0.12
As, ns = 2e-9, 0.96
Ok = 0.0

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Ocdmh2, omk=Ok)
pars.InitPower.set_params(As=As, ns=ns)

z_all = np.concatenate([[0.0], [0.295,0.510,0.706,0.930,1.317,1.491]])
sort_idx = np.argsort(z_all)[::-1]
z_sorted = z_all[sort_idx]
pars.set_matter_power(redshifts=z_sorted, kmax=10.0)

# Run CAMB
results = camb.get_results(pars)
kh, z, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)

# Get fσ₈
fs8_sorted = results.get_fsigma8()
# Undo sorting
fs8_unsorted = np.empty_like(fs8_sorted)
fs8_unsorted[sort_idx] = fs8_sorted

    # Get sigma8(z)
sigma8 = results.get_sigma8()  # array of 𝜎8 values, in order of increasing time (decreasing redshift)
f_camb = results.get_fsigma8() / sigma8  # array of f(z) values, in order of increasing time (decreasing redshift)

print("sigma8:", sigma8[sort_idx])
print("f(z) from CAMB:", f_camb[sort_idx])

Omz = np.array([results.get_Omega('baryon',z) for z in z_all]) + np.array([results.get_Omega('cdm',z) for z in z_all])
print("Ωₘ(z):", Omz)
f = Omz**0.55
print("f(z):", f)
fs8 = np.array(f*sigma8[sort_idx])
fs8_camb = results.get_fsigma8()[sort_idx]
print("fσ₈(z):", fs8)

data = np.column_stack((
    z_all,
    sigma8[sort_idx],
    Omz,
    f_camb[sort_idx],
    fs8_camb
))

np.savetxt(
    "CAMB-original_z0p0_results.txt",
    data,
    header="z sigma8 Omega_m f fs8"
)

np.savetxt('pklin_CAMB-original_z0p0.txt',np.c_[kh,pk_lin[0,:]])
np.savetxt('pklin_CAMB-original_z0p51.txt',np.c_[kh,pk_lin[2,:]])
np.savetxt('pklin_CAMB-original_z1p491.txt',np.c_[kh,pk_lin[6,:]])
np.savez('pklin_all.npz', kh=kh, z=z, pk=pk_lin)