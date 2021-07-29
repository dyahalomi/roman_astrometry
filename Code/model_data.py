import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
from astropy import units as u
from astropy.constants import constants
import aesara_theano_fallback.tensor as tt

import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


def model_rv(periods, Ks, x_rv, y_rv, y_rv_err):

# make a fine grid that spans the observation window for plotting purposes
t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)	



with pm.Model() as model:
		
		##  wide uniform prior on t_periastron
		tperi = pm.Uniform("tperi", lower=0, upper=5000, shape=2)
		
		#log normal prior on period around estimates

		logP = pm.Uniform(
			"logP",
			lower=0,
			upper=9,
			shape=2,
			testval=np.log(periods),
		)
		


		P = pm.Deterministic("P", tt.exp(logP))
		
		
		# Wide normal prior for semi-amplitude
		logK = pm.Uniform("logK", lower=-4, upper=3, shape=2, testval=np.log(Ks))
		
		K = pm.Deterministic("K", tt.exp(logK))
		
		
		# Eccentricity & argument of periasteron
		ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
		ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
		omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
		#xo.eccentricity.vaneylen19(
		#	"ecc_prior", multi=True, shape=2, fixed=True, observed=ecc
		#)
		
		# Jitter & a quadratic RV trend
		logs = pm.Normal("logs", mu=np.log(np.median(y_rv_err)), sd=0.01)

		# Then we define the orbit
		orbit = xo.orbits.KeplerianOrbit(period=P, t_periastron=tperi, ecc=ecc, omega=omega)

		# And a function for computing the full RV model
		def get_rv_model(t, name=""):
			# First the RVs induced by the planets
			vrad = orbit.get_radial_velocity(t, K=K)
			pm.Deterministic("vrad" + name, vrad)

			# Sum over planets and add the background to get the full model
			return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))

		# Define the RVs at the observed times
		rv_model = get_rv_model(x_rv)

		# Also define the model on a fine grid as computed above (for plotting)
		rv_model_pred = get_rv_model(t_rv, name="_pred")

		# Finally add in the observation model. This next line adds a new contribution
		# to the log probability of the PyMC3 model
		err = tt.sqrt(y_rv_err ** 2 + tt.exp(2 * logs))
		pm.Normal("obs", mu=rv_model, sd=err, observed=y_rv)


		map_soln = model.test_point
		map_soln = pmx.optimize(start=map_soln, vars=[ecs, K])
		map_soln = pmx.optimize(start=map_soln, vars=[tperi, ecs, K])
		map_soln = pmx.optimize(start=map_soln)



	#return the max a-posteriori solution
	return map_soln

def a_from_Kepler3(period, M_tot):
    period = period*86400 #days to seconds
    

    M_tot = M_tot*M_sun.value #solar masses to kg
    
    a3 = ( ((constants.G.value)*M_tot) / (4*np.pi**2) ) * (period)**2.
    
    a = a3**(1/3)
    
    
    
    a = a * 6.68459*10**(-12.) # meters to AU
    
    return(a) #in AUs


def semi_amplitude(m_planet, a, ecc, inclination):
    K = \
    np.sqrt(G / (1-(ecc**2.))) * ((m_planet*M_sun)*np.sin(inclination)) * \
    ((M_sun+(m_planet*M_sun))** (-(1./2.))) * \
    (a*u.AU.to(u.m))  ** ((-1./2.))
    
    return K.value



def min_mass(K, period, ecc):
	m_sun = 333030 #earth masses
    m_planet = K/((333030*m_jup)*28.4329/(np.sqrt(1-ecc**2.)) \
                  *(m_sun)**(-2/3) * (period / 365.256) ** (-1/3))

    return m_planet/m_sun

def determine_phase(P, t_periastron):
    phase = (2 * np.pi * t_periastron) / P
    return phase





def model_both(rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, rho, rho_err, theta, theta_err):
	# make a fine grid that spans the observation window for plotting purposes
	t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)	

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)


	P_RV = np.array(rv_map_soln['P'])
	K_RV = np.array(rv_map_soln['K'])
	tperi_RV = np.array(rv_map_soln['tperi'])
	ecc_RV = np.array(rv_map_soln['ecc'])
	omega_RV = np.array(rv_map_soln['omega'])
	min_masses_RV = min_mass(K_RV, P_RV, ecc_RV)
	phase_RV = determine_phase(P_RV, tperi_RV)




	


	def get_model():
	    with pm.Model() as model:


	        # Below we will run a version of this model where a measurement of parallax is provided
	        # The measurement is in milliarcsec
	        m_plx = pm.Bound(pm.Normal, lower=0, upper=200)(
	            "m_plx", mu=parallax*1000, sd=10, testval=parallax*1000
	        )
	        plx = pm.Deterministic("plx", 1e-3 * m_plx)


	        # We expect the period to be around that found from just the RVs,
	        # so we'll set a broad prior on logP
	        
	        logP = pm.Uniform(
	            "logP", lower=0, upper=np.log(10000.), testval=np.log(P_RV), shape=2
	        )
	        P = pm.Deterministic("P", tt.exp(logP))
	        
	        # Eccentricity & argument of periasteron
	        ecs = pmx.UnitDisk("ecs", shape=(2, 2), 
	                           testval=np.array([ecc_RV*np.cos(omega_RV), 
	                                             ecc_RV*np.sin(omega_RV)]))
	        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
	        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
	        
	        # Omegas are co-dependent, so sample them with variables Omega_plus
	        # and Omegas_minus. Omega_plus is (Omega_0 + Omega_1)/2 and 
	        # Omega_minus is (Omega_0 - Omega_1)/2
	        
	     
	        
	        Omega_plus = pmx.Angle("Omega_plus", shape=1)
	        Omega_minus = pmx.Angle("Omega_minus", shape=1)
	        
	        
	        Omega = tt.concatenate( [(Omega_plus + Omega_minus),
	                                 (Omega_plus - Omega_minus)] )
	        

	        Omega = pm.Deterministic("Omega", Omega) 
	        Omega_sum = pm.Deterministic("Omega_sum", ((Omega_plus)*2)% np.pi)
	        Omega_diff = pm.Deterministic("Omega_diff", ((Omega_minus)*2)% np.pi)
	        

	        
	        # uniform prior on t0, with testval from RV fit
	        #t0 = pm.Uniform("t0", lower=0, upper=10000., shape=2, testval = t0_RV)
	    
	        # For these orbits, it can also be better to fit for a phase angle
	        # (relative to a reference time) instead of the time of periasteron
	        # passage directly
	        phase = pmx.Angle("phase", testval=phase_RV, shape=2)
	        tperi = pm.Deterministic("tperi", P * phase / (2 * np.pi))
	        
	        # uniform prior on sqrtm_sini and sqrtm_cosi
	        sqrtm_sini = pm.Uniform("sqrtm_sini", lower=0, upper=500, 
	                                testval = min_masses_RV[0]*m_sun, shape=2)
	        
	        sqrtm_cosi = pm.Uniform("sqrtm_cosi", lower=0, upper=500, 
	                                testval = min_masses_RV[0]*m_sun, shape=2)
	        
	        
	        m_planet = pm.Deterministic("m_planet", sqrtm_sini**2. + sqrtm_cosi**2.)
	        m_planet_fit = pm.Deterministic("m_planet_fit", m_planet/m_sun)
	        
	        incl = pm.Deterministic("incl", tt.arctan2(sqrtm_sini, sqrtm_cosi))
	        
	        
	        
	        # add keplers 3 law function
	        a = pm.Deterministic("a", a_from_Kepler3(P, 1.0+m_planet_fit))
	        
	        # Set up the orbit
	        orbit = xo.orbits.KeplerianOrbit(
	            t_periastron=tperi,
	            period=P,
	            incl=incl,
	            ecc=ecc,
	            omega=omega,
	            Omega=Omega,
	            m_planet = m_planet_fit,
	            plx=plx
	        )


	        
	        
	        # Add a function for computing the full astrometry model
	        def get_astrometry_model(t, name=""):
	            # First the RVs induced by the planets
	            rhos, thetas = orbit.get_relative_angles(t, plx)
	            pm.Deterministic("rhos" + name, rhos)
	            pm.Deterministic("thetas" + name, thetas)
	            
	            # Sum over planets and add the background to get the full model
	            rho_model = pm.Deterministic("rho_model" + name, tt.sum(rhos, axis=-1))
	            
	            
	            # when summing over theta, position angle, we have to careful because position
	            # angle has the range -pi to pi. So for only summing 2 thetas, we can subtract 
	            # 2pi whenever theta_sum > pi and add 2pi whenever theta_sum < -pi to get back
	            # in the correct range. Be careful though if modeling more than 2 planets, this
	            # doesn't completely solve the problem!
	            thetas_sum = tt.sum(thetas, axis=-1)
	            thetas_sum = tt.switch(tt.lt(thetas_sum,-np.pi), thetas_sum+2*np.pi, thetas_sum)
	            thetas_sum = tt.switch(tt.gt(thetas_sum, np.pi), thetas_sum-2*np.pi, thetas_sum)

	            theta_model = pm.Deterministic("theta_model" + name, thetas_sum)
	            

	            
	            return rho_model, theta_model

	        
	        # Define the astrometry model at the observed times
	        rho_model, theta_model = get_astrometry_model(x_astrometry)

	        # Also define the model on a fine grid as computed above (for plotting)
	        rho_model_pred, theta_model_pred = get_astrometry_model(t_fine, name="_pred")

	        

	        # Add jitter terms to both separation and position angle
	        log_rho_s = pm.Normal(
	            "log_rho_s", mu=np.log(np.median(rho_err)), sd=0.01
	        )
	        log_theta_s = pm.Normal(
	            "log_theta_s", mu=np.log(np.median(theta_err)), sd=2.0
	        )
	        rho_tot_err = tt.sqrt(rho_err ** 2 + tt.exp(2 * log_rho_s))
	        theta_tot_err = tt.sqrt(theta_err ** 2 + tt.exp(2 * log_theta_s))

	        # define the likelihood function, e.g., a Gaussian on both rho and theta
	        pm.Normal("rho_obs", mu=rho_model, sd=rho_tot_err, observed=rho_data)

	        # We want to be cognizant of the fact that theta wraps so the following is equivalent to
	        # pm.Normal("obs_theta", mu=theta_model, observed=theta_data, sd=theta_tot_err)
	        # but takes into account the wrapping. Thanks to Rob de Rosa for the tip.
	        theta_diff = tt.arctan2(
	            tt.sin(theta_model - theta_data), tt.cos(theta_model - theta_data)
	        )
	        pm.Normal("theta_obs", mu=theta_diff, sd=theta_tot_err, observed=0.0)

	        
	        
	        # ADD RV MODEL
	        # Jitter & a quadratic RV trend
	        log_rv = pm.Normal("log_rv", mu=np.log(np.median(y_rv_err)), sd=0.01)


	        # And a function for computing the full RV model
	        def get_rv_model(t, name=""):
	            # First the RVs induced by the planets
	            vrad = orbit.get_radial_velocity(t)
	            pm.Deterministic("vrad" + name, vrad)

	            # Sum over planets and add the background to get the full model
	            return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))

	        # Define the RVs at the observed times
	        rv_model = get_rv_model(x_rv)

	        # Also define the model on a fine grid as computed above (for plotting)
	        rv_model_pred = get_rv_model(t, name="_pred")

	        # Finally add in the observation model. This next line adds a new contribution
	        # to the log probability of the PyMC3 model
	        rv_err = tt.sqrt(y_rv_err ** 2 + tt.exp(2 * log_rv))
	        pm.Normal("obs_RV", mu=rv_model, sd=rv_err, observed=y_rv)

	        # Optimize to find the initial parameters
	        map_soln = model.test_point
	        map_soln = pmx.optimize(map_soln, vars=[sqrtm_cosi, sqrtm_sini])
	        map_soln = pmx.optimize(map_soln, vars=[phase])
	        map_soln = pmx.optimize(map_soln, vars=[Omega, ecs])
	        map_soln = pmx.optimize(map_soln, vars=[P, a, phase])
	        map_soln = pmx.optimize(map_soln)


	    return model, map_soln


	model, map_soln = get_model()

	return model, map_soln









#########
#########
#########
#test code
import sys
sys.version

import exoplanet
print(f"exoplanet.__version__ = '{exoplanet.__version__}'")






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
from astropy import units as u
from astropy.constants import M_earth, M_sun
from simulate_data import *

import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

T_subtract = 2454000
# orbital parameters from https://www.princeton.edu/~willman/planetary_systems/Sol/
# BJD determined by converting values above using https://ssd.jpl.nasa.gov/tc.cgi#top

P_earth = 365.256
e_earth = 0.0167
Tper_earth= 2454115.5208333 - T_subtract
omega_earth = np.radians(102.9)
Omega_earth = np.radians(0.0)
inclination_earth = np.radians(70.0)
m_earth = 1*3.00273e-6 #units m_sun



P_jup = 4327.631
e_jup = 0.0484
Tper_jup = 2455633.7215278 - T_subtract
omega_jup = np.radians(274.3) - 2*np.pi
Omega_jup = np.radians(100.4)
inclination_jup = np.radians(1.31) + inclination_earth
m_jup = 317.83*3.00273e-6 #units m_sun


orbit_params_earth = [P_earth, e_earth, Tper_earth, omega_earth, Omega_earth, inclination_earth, m_earth]
orbit_params_jup = [P_jup, e_jup, Tper_jup, omega_jup, Omega_jup, inclination_jup, m_jup]


orbit_params = [orbit_params_earth, orbit_params_jup]

times_observed_astrometry = []
t_0 = int(Tper_earth)
for ii in range(t_0, t_0+1800):
    if ii % 50 == 0:
        times_observed_astrometry.append(ii)

n_planets = 2
t_dur_rv = 3650
n_obs_rv = 3000
sigma_rv = 0.3

sigma_theta = 0.05
sigma_rho = 0.001
parallax = 0.1


times, rv_results, theta_results, rho_results = simulate_data(
    n_planets, 
    sigma_rv, 
    sigma_theta,
    sigma_rho,
    parallax,
    orbit_params,
    t_dur_rv = t_dur_rv,
    n_obs_rv = n_obs_rv,
    times_observed_astrometry = times_observed_astrometry
    )


[[times_rv, times_observed_rv, times_astrometry, times_observed_astrometry],
     [rv_orbit, rv_orbit_sum, rv_sim, rv_sim_sum],
     [theta_orbit, theta_orbit_sum, theta_sim, theta_sim_sum],
     [rho_orbit, rho_orbit_sum, rho_sim, rho_sim_sum]]  = times, rv_results, theta_results, rho_results


x_rv = times_observed_rv
y_rv = rv_sim_sum
y_rv_err = np.full(np.shape(y_rv), sigma_rv)

x_astrometry = np.array(times_observed_astrometry)
theta_data = theta_sim_sum
theta_err = np.full(np.shape(theta_data), sigma_theta)
rho_data = rho_sim_sum
rho_err = np.full(np.shape(rho_data), sigma_rho)



periods_guess = [360, 4330]
Ks_guess = xo.estimate_semi_amplitude(periods, x_rv, y_rv, y_rv_err)



rv_map_soln = model_rv(periods_guess, Ks_guess, x_rv, y_rv, y_rv_err)
model, map_soln = model_both(rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, rho_data, rho_err, theta_data, theta_err):














