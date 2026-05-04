"""

:Synopsis: DESI fsigma8 measurements

"""
# Global
import numpy as np
# Local
from cobaya.likelihood import Likelihood
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

# SNe Functions
c = 299792.458 # [km/s]

class DESI_fsig8(Likelihood):
    file_base_name = 'DESI_fsig8'

    def initialize(self):
        self.log.info("Initializing")
        super().initialize()
        #self.mvecc, self.zvec_CMB, self.zvec_HEL, self.rdirs, self.cov_pan_fil = Pantheon_data()
        self._input_params_order = self.input_params
        self.provides = []
        self.output_params = []
   
    def get_requirements(self):
        return {
            "H0": None,
            "omegam": None,
            "omegak": None,
            "sigma8": None,
            "comoving_radial_distance": {"z": self.zvec_CMB},
            "Pk_interpolator": {
                "z": np.linspace(0, 3, 150),"k_max": 10, "nonlinear": False,   #NOTE: CHANGED the number of z points to 150 because thats the max CAMB_Gamma gives??
                "vars_pairs": [("delta_tot", "delta_tot")]},
            "Cl": {"tt": 0}
        }

    def initialize_with_params(self):
        self.log.info("Initializing")
        self._input_params_order = self._input_params_order or self.input_params


    def logp(self, **params_values):
        h = self.provider.get_param('H0')/100
        M = self.provider.get_param('M')
        
        lum_dis = self.provider.get_comoving_radial_distance(self.zvec_CMB)*h*100/c*(1+self.zvec_HEL)
        vecdif = self.mvecc - 5*np.log10(lum_dis) - M

        ####################
        mahalanobis = np.linalg.multi_dot([vecdif,inv_L_tot_fil,inv_L_tot_fil.T,vecdif])
        chi2 = mahalanobis + log_det_tot
        return -0.5 * chi2


def DESI_data():
    """
        Extracts information from Pantheon data
    """
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

    file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES.dat" # TODO: maybe point to pp.dat
    # file_path = "/home/joaoreboucas/cocoa/Cocoa/cobaya/cobaya/likelihoods/mylike/data/pp.dat"
    data = np.genfromtxt(file_path, skip_header=1, dtype=dtype)

    file_path = "/share/storage3/bets/camilo_storage3/Pantheon/Pantheon+SH0ES_STAT+SYS.cov" # TODO: maybe point to pp.cov
    # file_path = "/home/joaoreboucas/cocoa/Cocoa/cobaya/cobaya/likelihoods/mylike/data/pp.cov"
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


fs8_data = {'z':[0.295,0.510,0.706,0.930,1.317,1.491],
        'fsig8':[0.378,0.516,0.484,0.422,0.375,0.435], 
        'fsig8_err':[0.094,0.061,0.055,0.048,0.043,0.045],
        'tracer':['BGS','LRG1','LRG2','LRG3','ELG2','QSO']}

def logp(self, **params_values):
    sigma8 = self.provider.get_param('sigma8')
    gamma = self.provider.get_param('gamma0')
    Om = self.provider.get_param('omegam')
    h = self.provider.get_param('H0')/100

    kh = np.logspace(-4, 1, 200)
    PK = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),extrap_kmin=h*kh[0], nonlinear=False)
    #pk = PK.P(0, kh*h)*h*h*h # P(k) in Mpc^3 to P(k) in (Mpc/h)^3
    
    redshifts = np.concatenate([[0.0], fs8_data['z']])
    growth_factor = np.sqrt(PK.P(redshifts, 0.0005)/PK.P(0,0.0005))
    growth_factor /= growth_factor[0] # D(z=0) = 1
    
    sigma8_z = sigma8 * growth_factor
 
    H_z = results.hubble_parameter(z_dense)

    Ez2_dense = (H_dense / H0)**2
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

    fs8_theory = compute_fsigma8_camb(theta_fs8)
    chi2 = np.sum(((fs8_data['fsig8'] - fs8_theory) / fs8_data['fsig8_err'])**2)
    return -0.5*chi2

def Hub(h0,Om, OL,z):
    Ok = 1 - Om - OL
    return h0*100*np.sqrt( Om*(1+z)**3 + OL + Ok*(1+z)*(1+z) )/c

def f(h0,Om, OL,gamma,z):
    Omz = Om*(1+z)**3 *(h0*100/c/Hub(h0,Om, OL,z))**2
    return ( Omz )**(gamma)
