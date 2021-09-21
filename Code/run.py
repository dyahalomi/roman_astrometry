from simulate_and_model import *

incs_earth = [45.] #, 10., 80.] #degrees
periods_jup = [4327.631] #, 1000, 10000] #days
roman_errs = [5e-6] #, 10e-6, 20e-6, None] #micro-as
roman_durations = [10] #, 5] #years
gaia_obs = [200] #, 100] #number of observations with Gaia


for inc in incs_earth:
    for period in periods_jup:
        for roman_err in roman_errs:
            for roman_duration in roman_durations:
                for gaia_ob in gaia_obs:
                    #print('start... inc: ' + str(int(inc)) + ', roman_err: ' + str(int(1e6*roman_err)))
                    print('start')
                    print('--------')
                    print('Jupiter period: ' + str(int(period)))
                    print('Earth inclination: ' + str(int(inc)))

                    if roman_err is not None:
                        print('Roman precision: ' + str(int(1e6*roman_err)))
                    else:
                        print('Roman precision: N/A')
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
