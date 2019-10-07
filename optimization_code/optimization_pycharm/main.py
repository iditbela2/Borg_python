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
import conf_class_func
import data_preparation_functions
import objective_function

# (1) initialize the simulation
emissionRates = np.array([0.47, 0.51,0.38,0.9,0.19])
sourceLoc = np.array([[200,300],[300,700],[650,400],[450,200],[200,500]])
num_S = np.shape(emissionRates)[0]
distanceBetweenSensors = 50
distanceFromSource = 50

Q_source, sensorArray, sensors_to_exclude = \
    data_preparation_functions.initializeSimulation(num_S, sourceLoc, emissionRates,
                                                    distanceBetweenSensors, distanceFromSource)

# (2) calculate readings (totalField)
WD, WS, ASC = 270, 4, 2

totalField, total_active = data_preparation_functions.calcSensorsReadings(Q_source, sensorArray, WD, WS, ASC)

# # a try
# sensorIdx = np.array([100,200,300])
# thr, dyR = 1,1
# PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)

# (3) run optimization
# which indices are NaNs?
nan_idx = np.ravel(np.argwhere(np.isnan(totalField[:,0])), order = 'F')
NumOfSens_reduced = np.shape(totalField)[0] - np.shape(nan_idx)[0]
totalField = np.delete(totalField, nan_idx, axis=0)
np.shape(totalField)

# run borg
NumOfVars = NumOfSens_reduced  # Number of sensors
NumOfObj = 2 # Number of Objectives
NumOfCons = 2
NFE=10e5 # the number of objective function evaluations, defines how many times
#the Borg MOEA can invoke objectiveFcn.  Once the NFE limit is reached, the
#algorithm terminates and returns the result.

borg = Borg(NumOfVars, NumOfObj, NumOfCons,
            objective_function.objective_func_factory(totalField, total_active))
borg.setBounds(*[[0, 1]]*NumOfVars)
borg.setEpsilons(1, 0.5e-9) #maybe - borg.setEpsilons(*[1, 0.5e-7])

tic=timeit.default_timer()

# SOLVE - where is the runtime saved???
runtime_frequency = 1000
output_loc = '/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm/'
result = borg.solve({"maxEvaluations":NFE,
                     "runtime":output_loc + 'firstTry.runtime',
                     "frequency":runtime_frequency})

toc=timeit.default_timer()
print("elapsed time [sec]" + str(toc - tic))

# extract data
# f = open(output_loc + 'results.csv', 'w')
# f.write('#Borg Optimization Results\n')
# f.write('#First ' + str(NumOfVars) + ' are the decision variables, ' + 'last ' + str(NumOfObj) +
#         ' are the ' + 'objective values\n')
# for solution in result:
#
#     line = ''
#     for i in range(len(solution.getVariables())):
#         line = line + (str(solution.getVariables()[i])) + ' '
#
#     for i in range(len(solution.getObjectives())):
#         line = line + (str(solution.getObjectives()[i])) + ' '
#
#     f.write(line[0:-1] + '\n')
# # f.write("#")
# f.close()

# print data in console:
for solution in result:
    print(solution.getObjectives())


# extract vars and objs
objs = []
vars = []
for solution in result:
    objs.append(solution.getObjectives())
    vars.append(solution.getVariables())

# turn to df
objs = pd.DataFrame(data = objs)
vars = pd.DataFrame(data = vars)

# save them
objs.to_csv('objs.csv')
vars.to_csv('vars.csv')