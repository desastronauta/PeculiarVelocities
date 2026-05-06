"""
.. module:: likelihoods.mylike

:Synopsis: SNe Ia Pantheon+ likelihood with peculiar velocity covariance matrix
:Author: Joao Reboucas & Camilo Crisman

"""
# Global
import numpy as np
# Local
from cobaya.likelihood import Likelihood
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

# SNe Functions
c = 299792.458 # [km/s]
nmod = 628

class PP_PV(Likelihood):
    file_base_name = 'PP_PV'

    def initialize(self):
        self.log.info("Initializing")
        super().initialize()
        self.mvecc, self.zvec_CMB, self.zvec_HEL, self.rdirs, self.cov_pan_fil = Pantheon_data()
        self._input_params_order = self.input_params
        self.provides = []
        self.output_params = []
   
    def get_requirements(self):
        return {
            "H0": None,
            "omegam": None,
            "omegak": None,
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
        cov_tot = self.cov_pan_fil + self.cov_PV_mod_8(lum_dis, nmod)
        
        if np.any(np.linalg.eigvals(cov_tot) < 0): return -np.inf
        
        inv_cov_tot = np.linalg.inv(cov_tot)
        signo, log_det_tot = np.linalg.slogdet(cov_tot)
        inv_L_tot_fil = np.linalg.cholesky(inv_cov_tot)
        ####################
        mahalanobis = np.linalg.multi_dot([vecdif,inv_L_tot_fil,inv_L_tot_fil.T,vecdif])
        chi2 = mahalanobis + log_det_tot
        return -0.5 * chi2

    def cov_PV_mod_8(self,lumdis, nmod):
        sigv = self.provider.get_param("sigv")
        Ok = self.provider.get_param('omegak')
        Om = self.provider.get_param('omegam')
        h = self.provider.get_param('H0')/100  
        gamma = self.provider.get_param('gamma0')
        
        Nsns = len(lumdis)
        kh = np.logspace(-4, 1, 200)
        PK = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),extrap_kmin=h*kh[0], nonlinear=False)
        pk = PK.P(0, kh*h)*h*h*h # P(k) in Mpc^3 to P(k) in (Mpc/h)^3
        
        redshifts = np.linspace(0, 3, 100)
        growth_factor = np.sqrt(PK.P(redshifts, 0.0005)/PK.P(0,0.0005))
        growth_factor /= growth_factor[0]
        growth_function_interp = interp1d(redshifts, growth_factor, kind='cubic', fill_value="extrapolate")
        # TODO: CAMB_GammaPrime_Growth should give the corrected growth rate considering gamma
        
        ### implementacao da matriz
        OL = 1 - Om - Ok
        ctest = np.zeros((Nsns,Nsns), dtype=np.float64)
        constant = 5/np.log(10)
        ludis = lumdis/h/100*c
        twopi2 = 2*np.pi**2
        nonlinpart = (sigv/c)**2 - (240/c)**2
        rs = self.provider.get_comoving_radial_distance(self.zvec_CMB)
        Gfactor = - f(h,Om,OL,gamma,self.zvec_CMB)*growth_function_interp(self.zvec_CMB)/(1+self.zvec_CMB) # D(z) ou G(z)
        # TODO: check if this is correct
        
        ############## Here goes the factor that considers curvature
        if Ok>0 : # k = -1  OPEN universe
            R0 = c/(100*h*np.sqrt(Ok))
            ckvec = np.cosh(rs/R0)
        elif Ok<0 : # k = +1 CLOSED universe
            R0 = c/(100*h*np.sqrt(-Ok))
            ckvec = np.cos(rs/R0)
        else : ckvec = np.ones(Nsns) # flat universe
            ##############

        for i in range(nmod):
            zi = self.zvec_CMB[i]# ; RAi = RA_fil_rad[i] ; DECi = DEC_fil_rad[i]
            ri = rs[i] ; ri_dir = self.rdirs[i]

            for j in range(i, nmod):
                zj = self.zvec_CMB[j]# ; RAj = RA_fil_rad[j] ; DECj = DEC_fil_rad[j]
                rj = rs[j] ; rj_dir = self.rdirs[j]

                rij = ri*ri_dir-rj*rj_dir
                r = np.sqrt(rij@rij) # in Mpc too

                ci = constant*(1-(1+zi)**2 /Hub(h,Om,OL,zi)/ludis[i]*ckvec[i])
                cj = constant*(1-(1+zj)**2 /Hub(h,Om,OL,zj)/ludis[j]*ckvec[j])

                G_der_tau_i = Gfactor[i]*Hub(h,Om,OL,zi)
                G_der_tau_j = Gfactor[j]*Hub(h,Om,OL,zj)

                #################
                if r == 0:
                    integrand = pk /3 / twopi2
                    e_corrij = trapezoid(integrand, kh)
                    nonlin = nonlinpart
                else:
                    cosi = ri_dir@rij/r ; sini = np.sqrt(1-cosi**2)
                    cosj = rj_dir@rij/r ; sinj = np.sqrt(1-cosj**2)

                    integrand_par = pk * K_par(kh*r/h) / twopi2
                    integrand_per = pk * K_per(kh*r/h) / twopi2
                    epar = trapezoid(integrand_par, kh)
                    eper = trapezoid(integrand_per, kh)
                    e_corrij = sini*sinj*eper + cosi*cosj*epar
                    nonlin = 0            

                ctest[i][j] = ci*cj*e_corrij*G_der_tau_i*G_der_tau_j + nonlin*ci*cj
                if i != j:
                    ctest[j][i] = ctest[i][j]
        return ctest


def Hub(,Om, OL,z):
    Ok = 1 - Om - OL
    return h*100*np.sqrt( Om*(1+z)**3 + OL + Ok*(1+z)*(1+z) )/c

def K_par(x):
    return np.sin(x)/x - 2 * (np.sin(x)/x/x - np.cos(x)/x) /x
def K_per(x):
    return (np.sin(x)/x/x - np.cos(x)/x)/x

def f(h,Om, OL,gamma,z):
    Omz = Om*(1+z)**3 *(h*100/c/Hub(h,Om, OL,z))**2
    return ( Omz )**(gamma)

# Define the function to calculate the growth factor and its derivative
def get_growth_function_derivative(results, z_values):
    # Extract the growth factor G(z) from CAMB
    growth_factor = results.get_redshift_evolution(1, z_values, ['delta_tot'])[:, 0]
    # Normalize the growth factor (G(z=0) = 1)
    growth_factor /= growth_factor[0]
    G_interp = InterpolatedUnivariateSpline(z_values, growth_factor, k=3)
    return G_interp#, G_der_tau



def Pantheon_data():
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

####################