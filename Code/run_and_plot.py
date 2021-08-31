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
from simulate import *
from model import *
from astropy.timeseries import LombScargle
import pickle


import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


def run(inc, roman_err):



	##################
	##################
	##################
	##################
	#begin simulate data
	##################
	##################
	##################
	##################
	##################

	T_subtract = 2454000
	# orbital parameters from https://www.princeton.edu/~willman/planetary_systems/Sol/
	# BJD determined by converting values above using https://ssd.jpl.nasa.gov/tc.cgi#top

	P_earth = 365.256
	e_earth = 0.0167
	Tper_earth= 2454115.5208333 - T_subtract
	omega_earth = np.radians(102.9)
	Omega_earth = np.radians(0.0)
	inclination_earth = np.radians(inc)
	m_earth = 1*3.00273e-6 #units m_sun



	P_jup = 4327.631
	e_jup = 0.0484
	Tper_jup = 2455633.7215278 - T_subtract
	omega_jup = np.radians(274.3) - 2*np.pi
	Omega_jup = np.radians(100.4)
	inclination_jup = np.radians(1.31) + inclination_earth
	m_jup = 317.83*3.00273e-6 #units m_sun


	m_sun = 333030 #earth masses


	times_observed_astrometry_gaia = []
	t_0 = int(Tper_earth)
	for ii in range(t_0, t_0+3600):
		if ii % 90 == 0:
			times_observed_astrometry_gaia.append(ii)

	t_1 =  times_observed_astrometry_gaia[-1]+1800
	times_observed_astrometry_roman = []
	for ii in range(t_1, t_1+1800):
		if ii % 90 == 0:
			times_observed_astrometry_roman.append(ii)	

			
	times_observed_rv = []
	t_0 = int(Tper_earth)
	add_data = True
	for ii in range(t_0, t_0+3600):
		
		if ii % 180 == 0:
			if add_data:
				add_data = False
			else:
				add_data = True
		   
		if add_data:
			times_observed_rv.append(ii)
			

	orbit_params_earth = [P_earth, e_earth, Tper_earth, omega_earth, Omega_earth, inclination_earth, m_earth]
	orbit_params_jup = [P_jup, e_jup, Tper_jup, omega_jup, Omega_jup, inclination_jup, m_jup]

	n_planets = 2
	orbit_params = [orbit_params_earth, orbit_params_jup]


	sigma_rv = 0.3

	sigma_ra_gaia = 1e-5
	sigma_dec_gaia = 1e-5
	parallax = 0.1



	times, rv_results, ra_results, dec_results = simulate_data(
		n_planets, 
		sigma_rv, 
		sigma_ra_gaia,
		sigma_dec_gaia,
		parallax,
		orbit_params,
		times_observed_rv = times_observed_rv,
		times_observed_astrometry = times_observed_astrometry_gaia
		)


	[[times_rv, times_observed_rv, times_astrometry, times_observed_astrometry],
	[rv_orbit, rv_orbit_sum, rv_sim, rv_sim_sum],
	[ra_orbit, ra_orbit_sum, ra_sim, ra_sim_sum],
	[dec_orbit, dec_orbit_sum, dec_sim, dec_sim_sum]]  = times, rv_results, ra_results, dec_results



	sigma_ra_roman = roman_err
	sigma_dec_roman = roman_err



	times, rv_results, ra_results, dec_results = simulate_data(
		n_planets, 
		sigma_rv, 
		sigma_ra_roman,
		sigma_dec_roman,
		parallax,
		orbit_params,
		times_observed_rv = times_observed_rv,
		times_observed_astrometry = times_observed_astrometry_roman
		)

	#mask_start = len(times_astrometry)
	times_astrometry = np.append(times_astrometry, times[2], axis=0)
	#times_astrometry[mask_start] = np.ma.masked

	times_observed_astrometry = np.append(times_observed_astrometry, times[3], axis=0)

	ra_orbit = np.append(ra_orbit, ra_results[0], axis=0)
	ra_orbit_sum = np.append(ra_orbit_sum, ra_results[1], axis=0)
	ra_sim = np.append(ra_sim, ra_results[2], axis=0)
	ra_sim_sum = np.append(ra_sim_sum, ra_results[3], axis=0)

	dec_orbit = np.append(dec_orbit, dec_results[0], axis=0)
	dec_orbit_sum = np.append(dec_orbit_sum, dec_results[1], axis=0)
	dec_sim = np.append(dec_sim, dec_results[2], axis=0)
	dec_sim_sum = np.append(dec_sim_sum, dec_results[3], axis=0)

	
	plot_rv_signal(
	2, 
	rv_orbit, 
	rv_orbit_sum, 
	rv_sim, 
	rv_sim_sum,
	times_rv,
	times_observed_rv,
	['#366537', '#db372b', '#00257c'],
	['Earth', 'Jupiter'])


	plot_astrometry_signal(
	2, 
	ra_orbit, 
	ra_orbit_sum, 
	ra_sim, 
	ra_sim_sum, 
	dec_orbit, 
	dec_orbit_sum, 
	dec_sim, 
	dec_sim_sum,
	times_astrometry,
	times_observed_astrometry,
	['#366537', '#db372b', '#00257c'],
	['Earth', 'Jupiter'])

	plt.show()
	

	##################
	##################
	##################
	##################
	#end simulate data
	##################
	##################
	##################
	##################
	##################



	##################
	##################
	##################
	##################
	#begin model data
	##################
	##################
	##################
	##################
	##################

	################
	################
	#rename variables in more consistent way for modeling
	x_rv = np.array(times_observed_rv)
	y_rv = rv_sim_sum
	y_rv_err = np.full(np.shape(y_rv), sigma_rv)

	x_astrometry = np.array(times_observed_astrometry)
	ra_data = ra_sim_sum
	ra_err = np.full(np.shape(ra_data), roman_err)
	dec_data = dec_sim_sum
	dec_err = np.full(np.shape(dec_data), roman_err)


	# make a fine grid that spans the observation window for plotting purposes
	t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)





	################
	################
	#Lombs Scargle Periodogram on RV data
	frequency, power = LombScargle(x_rv, y_rv).autopower()
	period = 1/frequency


	period_cut1 = period[period > 10]
	power_cut1 = power[period > 10]



	indices = power_cut1.argsort()[-1:][::-1]
	period1 = np.array(period_cut1[indices][0])
	print('LS period 1: ' + str(period1))

	period1_min_cut = 2000
	#period_cut1 > period1_min_cut so we don't double count

	period_cut2 = period_cut1[period_cut1 < period1_min_cut]

	power_cut2 = power_cut1[period_cut1 < period1_min_cut]


	indices = power_cut2.argsort()[-1:][::-1]
	period2 = period_cut2[indices][0]
	print('LS period 2: ' + str(period2))



	
	################
	################
	#minimize on RV data
	periods_guess = [period2, period1]

	Ks_guess = xo.estimate_semi_amplitude(periods_guess, x_rv, y_rv, y_rv_err)


	rv_map_soln = minimize_rv(periods_guess, Ks_guess, x_rv, y_rv, y_rv_err)


	################
	################
	
	#plot rv minimization results
	ekw = dict(fmt=".k", lw=0.5)
	fig, ax = plt.subplots(nrows=2, sharex=True, figsize = [15,13])

	tot_rv_err = np.sqrt(y_rv_err ** 2 + np.exp(2 * rv_map_soln["logs"]))


	ax[0].errorbar(x_rv, y_rv, yerr=tot_rv_err, fmt=".k", alpha = 0.05, label='data', zorder=1)
	ax[0].plot(t_rv, rv_map_soln["rv_model_pred"], "b", label="combined model", zorder=2, color = '#773f6a')
	ax[0].plot(t_rv, rv_map_soln["vrad_pred"], "--k", alpha=0.5, label="individual models")

	ax[0].legend(fontsize=10)
	ax[0].set_xlim(t_rv.min(), t_rv.max())
	ax[0].set_xlabel("time [days]")
	ax[0].set_ylabel("radial velocity [m/s]")
	ax[0].set_title("RV Minimization from Joint Model")


	ax[1].axhline(0.0, color="0.5")
	ax[1].errorbar(
		x_rv, y_rv - rv_map_soln["rv_model"], yerr=tot_rv_err, **ekw
	)
	ax[1].set_xlabel("time [days]")
	ax[1].set_ylabel("RV residuals [m/s]")


	plt.show()
	


	################
	################
	#minimize on joint model
	parallax = 0.1 # arcsec
	model, map_soln, logp = minimize_both(
		rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, 
		ra_data, ra_err, dec_data, dec_err, parallax
	)

	
	################
	################
	
	#plot ra and dec minimizations vs time
	fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(6, 8))
	ax[0].set_ylabel(r"$\Delta \alpha \cos \delta$ ['']")
	ax[1].set_ylabel(r"$\Delta \alpha \cos \delta$ [''] residuals")
	ax[2].set_ylabel(r"$\Delta \delta$ ['']")
	ax[3].set_ylabel(r"$\Delta \delta$ [''] residuals")


	tot_ra_err = np.sqrt(ra_err ** 2 + np.exp(2 * map_soln["log_ra_s"]))
	tot_dec_err = np.sqrt(dec_err ** 2 + np.exp(2 * map_soln["log_dec_s"]))

	ax[0].errorbar(x_astrometry, ra_data, yerr=tot_ra_err, **ekw)
	ax[0].plot(t_fine, map_soln["ra_model_fine"], color="#773f6a")

	ax[1].axhline(0.0, color="0.5")
	ax[1].errorbar(
		x_astrometry, ra_data - map_soln["ra_model"], yerr=tot_ra_err, **ekw
	)


	ax[2].plot(t_fine, map_soln["dec_model_fine"], color="#773f6a")
	ax[2].errorbar(x_astrometry, dec_data, yerr=tot_dec_err, **ekw)

	ax[3].axhline(0.0, color="0.5")
	ax[3].errorbar(
		x_astrometry, dec_data - map_soln["dec_model"], yerr=tot_dec_err, **ekw
	)

	ax[3].set_xlim(t_fine[0], t_fine[-1])
	_ = ax[0].set_title("map orbit")


	################
	################
	#plot ra vs. dec
	fig, ax = plt.subplots(1, figsize = [9,9])
	ax.plot(map_soln["ra_model_fine"], map_soln["dec_model_fine"], 
			color="#773f6a", lw=1, label = "RV + astrometry model")


	ax.plot(ra_data, dec_data, ".k", label = "data")

	ax.set_ylabel(r"$\Delta \delta$ ['']")
	ax.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
	ax.invert_xaxis()
	ax.plot(0, 0, "k*")
	ax.set_aspect("equal", "datalim")
	ax.set_title("initial orbit")
	ax.legend()
	plt.show()



	################
	################
	#plot rv vs time for joint minimization
	fig, ax = plt.subplots(nrows=2, sharex=True, figsize = [15,13])

	tot_rv_err = np.sqrt(y_rv_err ** 2 + np.exp(2 * map_soln["log_rv"]))

	ax[0].errorbar(x_rv, y_rv, yerr=tot_rv_err, fmt=".k", alpha = 0.05, label='data', zorder=1)
	ax[0].plot(t_rv, map_soln["rv_model_pred"], "b", label="combined model", zorder=2, color = '#773f6a')
	ax[0].plot(t_rv, map_soln["vrad_pred"], "--k", alpha=0.5, label="individual models")

	ax[0].legend(fontsize=10)
	ax[0].set_xlim(t_rv.min(), t_rv.max())
	ax[0].set_xlabel("time [days]")
	ax[0].set_ylabel("radial velocity [m/s]")
	ax[0].set_title("RV Minimization from Joint Model")


	ax[1].axhline(0.0, color="0.5")
	ax[1].errorbar(
		x_rv, y_rv - map_soln["rv_model"], yerr=tot_rv_err, **ekw
	)
	ax[1].set_xlabel("time [days]")
	ax[1].set_ylabel("RV residuals [m/s]")


	plt.show()
	

	################
	################
	#run full MCMC
	np.random.seed(1234)
	with model:
		trace = pmx.sample(
			tune=1000,
			draws=1000,
			start=map_soln,
			cores=2,
			chains=2,
			target_accept=0.95,
			return_inferencedata=True,
		)

	##################
	##################
	##################
	##################
	#end model data
	##################
	##################
	##################
	##################
	##################


	################
	################
	#save trace and model
	with open('./traces/inc' + str(int(inc)) + '_gaia10_roman5_err' + str(int(1e6*roman_err)) + '.pkl', 'wb') as buff:
		pickle.dump({'model': model, 'trace': trace}, buff)


	return model, trace













