from simulate_and_model import *

incs = [10., 45.]
errs = [5e-6, 10e-6, 20e-6]


for inc in incs:
    for roman_err in errs:
        print('start... inc: ' + str(int(inc)) + ', roman_err: ' + str(int(1e6*roman_err)))
        simulate_and_model_data(inc, roman_err)
