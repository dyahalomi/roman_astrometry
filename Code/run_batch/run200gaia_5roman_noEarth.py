from simulate_and_model_noEarth import *




incs_earth = [45.] #, 10., 80.] #degrees
periods_jup = [4327] #, 1000, 10000] #days
roman_errs = [5e-6, 10e-6, 20e-6, None] #micro-as
roman_durations = [5]#, 10] #years
gaia_obs = [200]#, 100] #number of observations with Gaia


for inc in incs_earth:
	for period in periods_jup:
		for roman_err in roman_errs:
			for roman_duration in roman_durations:
				for gaia_ob in gaia_obs:
					if __name__ == "__main__":
						print('start')
						print('--------')
						print('Jupiter period: ' + str(int(period)))
						print('Earth inclination: ' + str(int(inc)))

						if roman_err is not None:
							print('Roman precision: ' + str(int(1e6*roman_err)))
							print('Roman duration: ' + str(int(roman_duration)))
						else:
							print('Roman precision: N/A')
							print('Roman duration: N/A')

						print('Gaia Obs: ' + str(int(gaia_ob)))


						simulate_and_model_data(inc, period, roman_err, roman_duration, gaia_ob)
						print('end')
						print('')
						print('')
						print('')
						print('')
						print('')
						print('')
						print('')
						print('')
						print('')
						print('')
