import numpy as np
import os, glob, sys, argparse
import multiprocessing as mp
from getdist import loadMCSamples
from tqdm import tqdm
from scipy.interpolate import interp1d
camb_mod_path = "/home/camilo/cocoa/Cocoa/external_modules/code/CAMB_GammaPrime_Growth"
sys.path.insert(0, camb_mod_path)
import camb
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Compute chi2 from MCMC chains")

parser.add_argument("--mcmc", type=str, default="MCMC",
                    help="MCMC tag (e.g. MCMC21, MCMC20, ...)")

parser.add_argument("--ntasks", type=int, default=100,
                    help="Number of tasks")

args = parser.parse_args()

# Assign variables
mcmc_tag = args.mcmc
ncpu = args.ntasks
print(f'Number of tasks: {ncpu}')
print(f'MCMC tag: {mcmc_tag}')

# Find all chain files
ruta_base = '/share/storage3/bets/camilo_storage3/cmb_gamma/chains'

archivos_chain = sorted(glob.glob(os.path.join(ruta_base, f'{mcmc_tag}.*.txt')))
print(f"Found {len(archivos_chain)} chain files")

# Define data
fs8_data = {'z':[0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS','LRG1','LRG2','LRG3','ELG2','QSO']}

# --- Read and stack all chains ---
chains = []

for file in archivos_chain:
    data = np.loadtxt(file)
    chains.append(data)

# Combine all chains into one array
chains = np.vstack(chains)

print("Total samples:", chains.shape[0])

# --- Extract columns ---
weights = chains[:, 0]
minus_loglike = chains[:, 1]
params = chains[:, 2:]   # all parameters

samples = loadMCSamples(
    os.path.join(ruta_base, mcmc_tag),
    settings={'ignore_rows': 0.5}
)
param_names = [p.name for p in samples.getParamNames().names]
params_chains = {}
for name in ['H0', 'omegabh2', 'omegach2', 'As', 'ns', 'omegak','gamma0']:
    params_chains[name] = params[:, param_names.index(name)]

def compute_fsigma8_camb(theta_fs8, redshifts=fs8_data['z']):
    H0, Ombh2, Ocdmh2, As, ns, Ok, gamma = theta_fs8
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    
    # Set basic cosmology
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Ocdmh2, omk=Ok, gamma0=gamma, gamma1=0.0)
    pars.InitPower.set_params(As=As, ns=ns)

    z_all = np.concatenate([[0.0], redshifts])
    sort_idx = np.argsort(z_all)[::-1]
    z_sorted = z_all[sort_idx]
    pars.set_matter_power(redshifts=z_sorted, kmax=10.0)

    # Run CAMB
    results = camb.get_results(pars)
    
    # --- Compute growth factor D(z) for f ---
    # --- 1. Create a z grid ---
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

    return fs8

def chi2_fs8(theta_fs8):
    fs8_theory = compute_fsigma8_camb(theta_fs8)
    chi2 = np.sum(((fs8_data['fsig8'] - fs8_theory) / fs8_data['fsig8_err'])**2)
    return chi2

def process_chunk(indices):
    results = []
    for i in indices:
        #H0, Ombh2, Ocdmh2, As, ns, Ok, gamma
        theta_fs8 = [
            params_chains['H0'][i],
            params_chains['omegabh2'][i],
            params_chains['omegach2'][i],
            params_chains['As'][i],
            params_chains['ns'][i],
            params_chains['omegak'][i],
            params_chains['gamma0'][i]
        ]
        chi2 = chi2_fs8(theta_fs8)
        results.append(chi2)
    return results

def chunkify(n, n_chunks):
    return np.array_split(np.arange(n), n_chunks)

if __name__ == "__main__":
    #mp.cpu_count()
    n_chunks = ncpu * 10
    chunks = chunkify(len(params), n_chunks)
    
    with mp.Pool(ncpu) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    chi2_values = np.concatenate(results)

header = " ".join(param_names) + " chi2_fs8"
output = np.column_stack([params, chi2_values])

output_filename = f"chi2_fs8_{mcmc_tag}.txt"
np.savetxt(output_filename, output, header=header)