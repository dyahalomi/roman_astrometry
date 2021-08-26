import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
from astropy import units as u
from astropy.constants import M_earth, M_sun
from astropy import constants
import aesara_theano_fallback.tensor as tt
from aesara_theano_fallback import aesara as theano


import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)



def model_rv(periods, Ks, x_rv, y_rv, y_rv_err):
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

	with pm.Model() as model:
	
		
		#log normal prior on period around estimates
		logP = pm.Uniform(
			"logP",
			lower=0,
			upper=9,
			shape=2,
			testval=np.log(periods),
		)
		


		P = pm.Deterministic("P", tt.exp(logP))


		##  wide uniform prior on t_periastron
		tperi = pm.Uniform("tperi", lower=x_rv.min(), upper=x_rv.max(), shape=2)
		
		
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
		logs = pm.Normal("logs", mu=np.log(np.median(y_rv_err)), sd=y_rv_err)

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
		map_soln = pmx.optimize(start=map_soln, vars=[tperi])
		map_soln = pmx.optimize(start=map_soln, vars=[P])
		map_soln = pmx.optimize(start=map_soln, vars=[ecs])
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
	from astropy.constants import G

	K = \
	np.sqrt(G / (1-(ecc**2.))) * ((m_planet*M_sun)*np.sin(inclination)) * \
	((M_sun+(m_planet*M_sun))** (-(1./2.))) * \
	(a*u.AU.to(u.m))  ** ((-1./2.))
	
	return K.value



def min_mass(K, period, ecc):
	#from http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf
	m_jup = 317.83*3.00273e-6 #units m_sun
	m_sun = 333030 #earth masses

	m_planet = K/((m_sun*m_jup)*28.4329/(np.sqrt(1-ecc**2.)) \
		*(m_sun)**(-2/3) * (period / 365.256) ** (-1/3))

	return m_planet/m_sun

def determine_phase(P, t_periastron):
	phase = (2 * np.pi * t_periastron) / P
	return phase





def model_both(rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, ra_data, ra_err, dec_data, dec_err, parallax):
	m_sun = 333030 #earth masses
	
	P_RV = np.array(rv_map_soln['P'])
	K_RV = np.array(rv_map_soln['K'])
	tperi_RV = np.array(rv_map_soln['tperi'])
	ecc_RV = np.array(rv_map_soln['ecc'])
	omega_RV = np.array(rv_map_soln['omega'])
	#min_masses_RV = min_mass(K_RV, P_RV, ecc_RV)
	min_masses_RV = xo.estimate_minimum_mass(P_RV, x_rv, y_rv, y_rv_err)/m_sun #in m_earth
	phase_RV = determine_phase(P_RV, tperi_RV)
	
	
	# make a fine grid that spans the observation window for plotting purposes
	t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)


	print("RV Solutions")
	print("------------")
	print("P: ", P_RV)
	print("K: ", K_RV)
	print("T_peri: ", tperi_RV)
	print("eccentricity: ", ecc_RV)
	print("omega: ", omega_RV)


	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)

	#inc_test_vals = np.array(np.radians([0.01, 10., 20., 30., 40., 50., 60., 70., 80., 89.9]))
	inc_test_vals = np.array(np.radians([90.]))
	model, map_soln = [], []
	for inc in inc_test_vals:
		mass_test_vals = min_masses_RV/np.sin(inc)
		print(np.sin(inc))
		print(min_masses_RV*m_sun)
		print(mass_test_vals*m_sun)


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
								   testval=np.array([np.sqrt(ecc_RV)*np.cos(omega_RV), 
													 np.sqrt(ecc_RV)*np.sin(omega_RV)]))
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
				


			
				# For these orbits, it can also be better to fit for a phase angle
				# (relative to a reference time) instead of the time of periasteron
				# passage directly
				phase = pmx.Angle("phase", testval=phase_RV, shape=2)
				tperi = pm.Deterministic("tperi", P * phase / (2 * np.pi))
				

				
				# uniform prior on sqrtm_sini and sqrtm_cosi (upper 10* min mass to stop planet flipping)
				sqrtm_sini = pm.Uniform(
					"sqrtm_sini_1", lower=0, upper=100*min_masses_RV*m_sun, 
					testval = np.sqrt(mass_test_vals*m_sun)*np.sin(inc), shape=2)
				
				sqrtm_cosi = pm.Uniform(
					"sqrtm_cosi_1", lower=0, upper=100*min_masses_RV*m_sun, 
					testval = np.sqrt(mass_test_vals*m_sun)*np.cos(inc), shape=2)

			
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
					# First the astrometry induced by the planets

					# determine and print the star position at desired times
					pos = orbit.get_star_position(t, plx)

					x,y,z = pos


					# calculate rho and theta
					rhos = tt.squeeze(tt.sqrt(x ** 2 + y ** 2))  # arcsec
					thetas = tt.squeeze(tt.arctan2(y, x))  # radians between [-pi, pi]
									
					
					#rhos, thetas = get_star_relative_angles(t, plx)
					
					
					dec = pm.Deterministic("dec" + name, rhos * np.cos(thetas)) # X is north
					ra = pm.Deterministic("ra" + name, rhos * np.sin(thetas)) # Y is east
					
					# Sum over planets to get the full model
					dec_model = pm.Deterministic("dec_model" + name, tt.sum(dec, axis=-1))
					ra_model = pm.Deterministic("ra_model" + name, tt.sum(ra, axis=-1))
					

					
					return dec_model, ra_model

				
				# Define the astrometry model at the observed times
				dec_model, ra_model = get_astrometry_model(x_astrometry)

				# Also define the model on a fine grid as computed above (for plotting)
				dec_model_fine, ra_model_fine = get_astrometry_model(t_fine, name="_fine")

				

				# Add jitter terms to both separation and position angle
				log_dec_s = pm.Normal(
					"log_dec_s", mu=np.log(np.median(dec_err)), sd=dec_err
				)
				log_ra_s = pm.Normal(
					"log_ra_s", mu=np.log(np.median(ra_err)), sd=ra_err
				)
				dec_tot_err = tt.sqrt(dec_err ** 2 + tt.exp(2 * log_dec_s))
				ra_tot_err = tt.sqrt(ra_err ** 2 + tt.exp(2 * log_ra_s))

				# define the likelihood function, e.g., a Gaussian on both ra and dec		
				pm.Normal("dec_obs", mu=dec_model, sd=dec_tot_err, observed=dec_data)
				pm.Normal("ra_obs", mu=ra_model, sd=ra_tot_err, observed=ra_data)


				
				
				# ADD RV MODEL
				# Jitter & a quadratic RV trend
				log_rv = pm.Normal("log_rv", mu=np.log(np.median(y_rv_err)), sd=y_rv_err)


				# And a function for computing the full RV model
				def get_rv_model(t, name=""):
					# First the RVs induced by the planets
					vrad = orbit.get_radial_velocity(t)
					pm.Deterministic("vrad" + name, vrad)

					# Sum over planets to get the full model
					return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))

				# Define the RVs at the observed times
				rv_model = get_rv_model(x_rv)

				# Also define the model on a fine grid as computed above (for plotting)
				rv_model_pred = get_rv_model(t_rv, name="_pred")

				# Finally add in the observation model. This next line adds a new contribution
				# to the log probability of the PyMC3 model
				rv_err = tt.sqrt(y_rv_err ** 2 + tt.exp(2 * log_rv))
				pm.Normal("obs_RV", mu=rv_model, sd=rv_err, observed=y_rv)

				# Optimize to find the initial parameters
				map_soln = model.test_point
				map_soln = pmx.optimize(map_soln, vars=[Omega, ecs])
				map_soln = pmx.optimize(map_soln)


			return model, map_soln



		a_model, a_map_soln = get_model()
		model.append(a_model)
		map_soln.append(a_map_soln)

	return model, map_soln








