from scipy.io import netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4
import dill

file2read = netCDF4.Dataset('datasets/KIS___OPER_P___OBS_____L2.nc','r')

time_data = file2read.variables['time'][:]
wind_data = file2read.variables['FHVEC'][:].data[0]
mean_temperatrue = file2read.variables['TG'][:].data[0]
file2read.close()

valid_indexes = []
for i, (w, t) in enumerate(zip(wind_data, mean_temperatrue)):
    if w == -9999 or t == -9999:
        continue

    valid_indexes += [i]

dill.dump(np.stack((time_data[valid_indexes], wind_data[valid_indexes], mean_temperatrue[valid_indexes])).T, open("datasets/meteo.pkl", 'wb'))
####
# Data points look like x[0] - time, x[1] - wind, x[2] - temperature 