import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import camb, os, h5py
from astropy.io import fits
c = 299792.458 ; ns=0.96

fs8_data = {'z':[0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS','LRG1','LRG2','LRG3','ELG2','QSO']}

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
    # --- Create a z grid ---
    z_dense = np.linspace(0, max(z_all), 500)

    # --- Omega_m(z) from CAMB ---
    H_dense = results.hubble_parameter(z_dense)

    Omega_m0 = (Ombh2 + Ocdmh2) / (H0/100.0)**2
    Ez2_dense = (H_dense / H0)**2
    Omega_m_dense = Omega_m0 * (1 + z_dense)**3 / Ez2_dense

    # --- f(z) ---
    f_dense = Omega_m_dense**gamma

    # --- Cumulative integral ---
    integrand = f_dense / (1 + z_dense)
    dz = np.diff(z_dense)
    I = np.zeros_like(z_dense)
    I[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)

    # --- Growth factor ---
    D_dense = np.exp(-I)

    # --- sigma8 ---
    sigma8_0 = results.get_sigma8()[-1]  # sigma8 at z=0 from CAMB
    sigma8_dense = sigma8_0 * D_dense

    # --- fsigma8 ---
    fsigma8_dense = f_dense * sigma8_dense

    # --- Interpolate ---
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

def DESY5_data():
    data_dir = '/share/storage3/bets/camilo_storage3/DES'
    file_name = 'DES-SN5YR_HD+MetaData.csv'
    file_path = os.path.join(data_dir, file_name)
    data = pd.read_csv(file_path)
    mB_DES = data['mB_corr'].to_numpy()
    zCMB_DES = data['zCMB'].to_numpy()
    zHEL_DES = data['zHEL'].to_numpy()
    MUERR_DES = data['MUERR_FINAL'].to_numpy()

    RA_DES = data['HOST_RA'].to_numpy()
    DEC_DES = data['HOST_DEC'].to_numpy()

    diagonal_matrix = np.diag(MUERR_DES)

    cov_name = 'STAT+SYS.txt.gz'
    cov_path = os.path.join(data_dir, cov_name)
    covvec = pd.read_csv(cov_path)
    mat_size = int(np.sqrt(len(covvec)))
    cov_mat = np.reshape(covvec.to_numpy(), (mat_size, mat_size))
    cov_mat_DES = cov_mat + diagonal_matrix**2

    # Abre el archivo
    with fits.open(os.path.join(data_dir, "DES-SN5YR_LOWZ_HEAD.FITS.gz")) as hdul:
        #hdul.info()  # Revisa las extensiones disponibles
        table_data = hdul[1].data  # Generalmente, la tabla estÃ¡ en la extensiÃ³n 1
    df_lowz = pd.DataFrame(table_data)

    with fits.open(os.path.join(data_dir, "DES-SN5YR_Foundation_HEAD.FITS.gz")) as hdul:
        table_data = hdul[1].data  
    df_found = pd.DataFrame(table_data)

    z_found = df_found['REDSHIFT_FINAL'].to_numpy()
    RA_found = df_found['RA'].to_numpy()
    DEC_found = df_found['DEC'].to_numpy()
    vpec_found = df_found['VPEC'].to_numpy()

    z_lowz = df_lowz['REDSHIFT_FINAL'].to_numpy()
    RA_lowz = df_lowz['RA'].to_numpy()
    DEC_lowz = df_lowz['DEC'].to_numpy()
    VPEC_lowz = df_lowz['VPEC'].to_numpy()


    RA_test = np.zeros(len(zCMB_DES)) ; DEC_test = np.zeros(len(zCMB_DES))
    vtest1 = np.zeros(len(zCMB_DES))  ; ztest1 = np.zeros(len(zCMB_DES))

    prec_z = 1e-5 ; prec_vpec = 1
    for i in range(len(zCMB_DES)):
        for j in range(len(z_found)):
            if (np.abs(zCMB_DES[i] - z_found[j])<prec_z and np.abs(data['VPEC'][i]-vpec_found[j])<prec_vpec):
                ztest1[i] = zCMB_DES[i]
                RA_test[i] = RA_found[j]
                DEC_test[i] = DEC_found[j]

        for j in range(len(z_lowz)):
            if (np.abs(zCMB_DES[i] - z_lowz[j])<prec_z and np.abs(data['VPEC'][i]-VPEC_lowz[j])<prec_vpec):
                ztest1[i] = zCMB_DES[i]
                RA_test[i] = RA_lowz[j]
                DEC_test[i] = DEC_lowz[j]

    RA_rec = RA_DES             ;  DEC_rec = DEC_DES
    RA_rec[RA_rec == -999] = 0  ;  DEC_rec[DEC_rec == -999] = 0
    RA_rec += RA_test           ;  DEC_rec += DEC_test
    # Get sorting indices
    sorted_indices = np.argsort(zCMB_DES)

    # Apply sorting to redshifts, magnitudes, and covariance matrix
    sorted_zCMB_vec = zCMB_DES[sorted_indices]
    sorted_mB_vec = mB_DES[sorted_indices]
    sorted_zHEL_vec = zHEL_DES[sorted_indices]
    sorted_RA_vec = RA_rec[sorted_indices]
    sorted_DEC_vec = DEC_rec[sorted_indices]
    sorted_cov_matrix = cov_mat_DES[np.ix_(sorted_indices, sorted_indices)]

    RA_rad = sorted_RA_vec*np.pi/180
    DEC_rad = sorted_DEC_vec*np.pi/180
    rdirs = []
    for i in range(len(sorted_RA_vec)):
        RAi = RA_rad[i] ; DECi = DEC_rad[i]
        diri = np.array([np.sin(RAi)*np.cos(DECi),np.sin(RAi)*np.sin(DECi),np.cos(RAi)])
        rdirs.append(diri)

    return sorted_mB_vec, sorted_zCMB_vec, sorted_zHEL_vec, rdirs, sorted_cov_matrix

def Pantheon_data():
    dtype = {
    'names': ('CID', 'IDSURVEY', 'zHD', 'zHDERR', 'zCMB', 'zCMBERR', 'zHEL',
              'zHELERR', 'm_b_corr', 'm_b_corr_err_DIAG', 'MU_SH0ES',
              'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
              'USED_IN_SH0ES_HF', 'c', 'cERR', 'x1', 'x1ERR', 'mB', 'mBERR',
              'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'RA', 'DEC',
              'HOST_RA', 'HOST_DEC', 'HOST_ANGSEP', 'VPEC', 'VPECERR', 'MWEBV',
              'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 'PKMJD', 'PKMJDERR', 'NDOF',
              'FITCHI2', 'FITPROB', 'm_b_corr_err_RAW', 'm_b_corr_err_VPEC',
              'biasCor_m_b', 'biasCorErr_m_b', 'biasCor_m_b_COVSCALE',
              'biasCor_m_b_COVADD'),
    'formats': ['U10'] + ['f8']*46  # 'U10' for the first 'string' column, 'f8' for the numeric columns
    }

    file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES.dat"
    data = np.genfromtxt(file_path, skip_header=1, dtype=dtype)

    file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES_STAT+SYS.cov"
    cov_vec = np.loadtxt(file_path, delimiter=' ') 
    matrix_size = int(np.sqrt(len(cov_vec[1:])))
    cov_matrix = np.reshape(cov_vec[1:], (matrix_size, matrix_size))

    zvec = data['zCMB'].T
    zHEL = data['zHEL'].T
    mvec = data['m_b_corr'].T
    RA = data['RA'].T
    DEC = data['DEC'].T

    indxs = np.where(zvec>=0.01)
    zvec_CMB = zvec[indxs]
    mvecc = mvec[indxs]
    zvec_HEL = zHEL[indxs]

    RA_vecc = RA[indxs]
    DEC_vecc = DEC[indxs]

    cov_matrix_fil = cov_matrix[indxs[0],:]
    cov_matrix_fil = cov_matrix_fil[:,indxs[0]]  # Eliminar columnas
    cov_pan_fil = cov_matrix_fil

    RA_rad = RA_vecc*np.pi/180
    DEC_rad = DEC_vecc*np.pi/180
    rdirs = []
    for i in range(len(zvec_CMB)):
        RAi = RA_rad[i] ; DECi = DEC_rad[i]
        diri = np.array([np.sin(RAi)*np.cos(DECi),np.sin(RAi)*np.sin(DECi),np.cos(RAi)])
        rdirs.append(diri)
    return mvecc, zvec_CMB, zvec_HEL, rdirs, cov_pan_fil

def Hub(h0,Om, OL,z):
    Ok = 1 - Om - OL
    return h0*100*np.sqrt( Om*(1+z)**3 + OL + Ok*(1+z)*(1+z) )/c

def K_par(x):
    return np.sin(x)/x - 2 * (np.sin(x)/x/x - np.cos(x)/x) /x
def K_per(x):
    return (np.sin(x)/x/x - np.cos(x)/x)/x

def f(h0,Om, OL,gamma,z):
    Omz = Om*(1+z)**3 *(h0*100/c/Hub(h0,Om, OL,z))**2
    return ( Omz )**(gamma)

# Define the function to calculate the growth factor and its derivative
# def get_growth_function_derivative(results, z_values):
#     # Extract the growth factor G(z) from CAMB
#     growth_factor = results.get_redshift_evolution(1, z_values, ['delta_tot'])[:, 0]
#     # Normalize the growth factor (G(z=0) = 1)
#     growth_factor /= growth_factor[0]
#     G_interp = InterpolatedUnivariateSpline(z_values, growth_factor, k=3)
#     return G_interp#, G_der_tau

def cov_PV_mod_8(theta,zvecc, rdirs, lumdis,nmod):
    Nsns = len(lumdis)
    OL, Om, Ob, sig8, h0, Mag, gamma, sigv = theta
    Oc = Om - Ob
    Ok = 1-Om-OL
    #################### 
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h0, ombh2=Ob*h0*h0, omch2=Oc*h0*h0, omk=Ok)
    pars.InitPower.set_params(As=2e-9, ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=10.0)
    results = camb.get_results(pars)

    # Extract the growth factor G(z) from CAMB and calculate G'
    redshifts = np.linspace(0,3,100)
    growth_factor = results.get_redshift_evolution(1, redshifts, ['delta_tot'])[:, 0]
    growth_factor /= growth_factor[0]
    growth_function_interp = interp1d(redshifts, growth_factor, kind='cubic', fill_value="extrapolate")
    #print(f"growth_function: {growth_function_interp(zvecc)}")
    ####################

    ############
    # Get sigma_8 and calculate the right pk later
    sigma8 = results.get_sigma8()[0]
    As= 2e-9*(sig8/sigma8)**2

    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)
    pk = pk*As/2e-9
    #print(f"Pk (shape={np.shape(pk)}) = {pk}")
    #print(f"trapezoid(pk, kh) = {trapezoid(pk, kh)}")
    ############

    ### implementacao da matriz
    ctest = np.zeros((Nsns,Nsns), dtype=np.float64)
    constant = 5/np.log(10)
    ludis = lumdis/h0/100*c
    twopi2 = 2*np.pi**2
    nonlinpart = (sigv/c)**2 - (240/c)**2
    rs = results.comoving_radial_distance(zvecc)
    #print(f"rs = {rs}")
    Gfactor = - f(h0,Om,OL,gamma,zvecc)*growth_function_interp(zvecc)/(1+zvecc)
    #print(f"Gfactor = {Gfactor}")

        ############## Here goes the factor that considers curvature
    if Ok>0 : # k = -1  OPEN universe
        R0 = c/(100*h0*np.sqrt(Ok))
        ckvec = np.cosh(rs/R0)
    elif Ok<0 : # k = +1 CLOSED universe
        R0 = c/(100*h0*np.sqrt(-Ok))
        ckvec = np.cos(rs/R0)
    else : ckvec = np.ones(Nsns) # flat universe
        ##############

    for i in range(nmod):
        zi = zvecc[i]# ; RAi = RA_fil_rad[i] ; DECi = DEC_fil_rad[i]
        ri = rs[i] ; ri_dir = rdirs[i]

        for j in range(i, nmod):
            zj = zvecc[j]# ; RAj = RA_fil_rad[j] ; DECj = DEC_fil_rad[j]
            rj = rs[j] ; rj_dir = rdirs[j]

            rij = ri*ri_dir-rj*rj_dir
            r = np.sqrt(rij@rij) # in Mpc too

            ci = constant*(1-(1+zi)**2 /Hub(h0,Om,OL,zi)/ludis[i]*ckvec[i])
            cj = constant*(1-(1+zj)**2 /Hub(h0,Om,OL,zj)/ludis[j]*ckvec[j])

            G_der_tau_i = Gfactor[i]*Hub(h0,Om,OL,zi)
            G_der_tau_j = Gfactor[j]*Hub(h0,Om,OL,zj)

            #################
            if r == 0:
                integrand = pk[0, :] /3 / twopi2
                e_corrij = trapezoid(integrand, kh)
                nonlin = nonlinpart
            else:
                cosi = ri_dir@rij/r ; sini = np.sqrt(1-cosi**2)
                cosj = rj_dir@rij/r ; sinj = np.sqrt(1-cosj**2)

                integrand_par = pk[0, :] * K_par(kh*r/h0) / twopi2
                integrand_per = pk[0, :] * K_per(kh*r/h0) / twopi2
                epar = trapezoid(integrand_par, kh)
                eper = trapezoid(integrand_per, kh)
                e_corrij = sini*sinj*eper + cosi*cosj*epar
                nonlin = 0            

            ctest[i][j] = ci*cj*e_corrij*G_der_tau_i*G_der_tau_j + nonlin*ci*cj
            if i != j:
                ctest[j][i] = ctest[i][j]
    return ctest

def save_to_h5(filename, sampler, recent_steps):
    with h5py.File(filename, "a") as f:
        # Get the recent steps
        chain = sampler.chain[:, :, -recent_steps:, :]  # (ntemps, nwalkers, steps, ndim)
        loglikelihood = sampler.loglikelihood[:, :, -recent_steps:]  # (ntemps, nwalkers, steps)
        logprobability = sampler.logprobability[:, :, -recent_steps:]  # (ntemps, nwalkers, steps)

        ntemps, nwalkers, nsteps, ndim = chain.shape

        # Save or append data to the HDF5 file
        if "chain" in f:
            # Resize existing datasets
            f["chain"].resize((f["chain"].shape[2] + nsteps), axis=2)
            f["chain"][:, :, -nsteps:, :] = chain

            f["loglikelihood"].resize((f["loglikelihood"].shape[2] + nsteps), axis=2)
            f["loglikelihood"][:, :, -nsteps:] = loglikelihood

            f["logprobability"].resize((f["logprobability"].shape[2] + nsteps), axis=2)
            f["logprobability"][:, :, -nsteps:] = logprobability
        else:
            # Create datasets for the first time
            maxshape = (ntemps, nwalkers, None, ndim)
            f.create_dataset("chain", data=chain, maxshape=maxshape, chunks=True)

            maxshape_ln = (ntemps , nwalkers, None)
            f.create_dataset("loglikelihood", data=loglikelihood, maxshape=maxshape_ln, chunks=True)
            f.create_dataset("logprobability", data=logprobability, maxshape=maxshape_ln, chunks=True)

def save_to_h5_emcee(filename, sampler, recent_steps):
    with h5py.File(filename, "a") as f:
        # Get the recent steps
        chain = sampler.get_chain()[-recent_steps:, :, :]  # (nsteps, nwalkers, ndim)
        nsteps, nwalkers, ndim = chain.shape

        # Save or append data to the HDF5 file
        if "chain" in f:
            f["chain"].resize((f["chain"].shape[0] + nsteps), axis=0)
            f["chain"][-nsteps:, :, :] = chain

        else:
            # Create datasets for the first time
            maxshape = (None, nwalkers, ndim)
            f.create_dataset("chain", data=chain, maxshape=maxshape, chunks=True)
