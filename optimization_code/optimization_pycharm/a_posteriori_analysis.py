# compare optimal locations to places of high average concentration map
# compare optimal locations to places of high variance according to all possible states (32*144) when stacks emit average emissions
# my solution should probably be identical to the variance...
# However, if i add the cost with different sensor types, then it is different.

# generate all possible states (32*144) when stacks emit average emissions
# compare PEDs of optimal Vs. non-optimal solutions (randomly generated)

# generate weather states according to frequency and stacks emit all kinds of emissions (+-50% from average)
# compare PEDs of optimal Vs. non-optimal solutions (randomly generated)


import numpy as np
import csv
import timeit
# np.set_printoptions(precision=10)
import pandas as pd
import sys
sys.path.append('/Users/iditbela/Documents/Borg_python/Borg_downloaded_code/serial-borg-moea/')
from plugins.Python.borg import Borg
from matplotlib import pyplot as plt

sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm')
sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_notebooks')
import conf_class_func
import data_preparation_functions
import objective_function

# (1) initialize the simulation
emissionRates = np.array([0.47,0.51,0.38,0.9,0.19])
sourceLoc = np.array([[200,300],[300,700],[650,400],[450,200],[200,500]])
num_S = np.shape(emissionRates)[0]
distanceBetweenSensors = 50
distanceFromSource = 50

Q_source, sensorArray, sensors_to_exclude = \
    data_preparation_functions.initializeSimulation(num_S, sourceLoc, emissionRates,
                                                    distanceBetweenSensors, distanceFromSource)


# (2) generate all possible states/maps (number of maps = 32*144) when stacks emit average emissionss
# df = pd.read_pickle("/Users/iditbela/Documents/Borg_python/optimization_code/optimization_notebooks/WF_2004_2018_Hadera")
# # df is sorted
# # 144 states have certain probability different than zero
# numStates = df.loc[df.percent != 0,'s'].count()
# totalTotalField = np.zeros((441,32,numStates))
# for state in range(numStates):
# #     wf = df.iloc[state].percent
#     WD = df.iloc[state].WD_to_apply
#     WS = df.iloc[state].WS_to_apply
#     ASC = df.iloc[state].SC_to_apply
#     totalField, total_active = data_preparation_functions.calcSensorsReadings(Q_source, sensorArray, WD, WS, ASC)
#     totalTotalField[:,:,state] = totalField
#
# np.save('totalTotalField', totalTotalField)
# np.save('total_active', total_active)

# (2) compare PEDs of optimal Vs. non-optimal solutions (randomly generated) for all 144 weather states
totalTotalField = np.load('totalTotalField.npy')
total_active = np.load('total_active.npy')



# ** remove the nan rows
# nan_idx = np.ravel(np.argwhere(np.isnan(totalField[:,0])), order = 'F')
# NumOfSens_reduced = np.shape(totalField)[0] - np.shape(nan_idx)[0]
# totalField = np.delete(totalField, nan_idx, axis=0)
# np.shape(totalField)

# ** where is the sensorIdx
thr, dyR = 1,1
sensorIdx = np.argwhere(x).ravel()

# ** calculate PED
PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalTotalField[:,:,0], total_active, sensorIdx, thr, dyR)
[c, idx] = np.unique(np.sort(scenario_pairs[:, 2:4], axis=1), axis=0, return_inverse=True)
min_PED = PEDs.groupby(idx).min()
mean_PED = np.mean(min_PED)