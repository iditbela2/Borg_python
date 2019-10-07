'''This is the objective function that is activated by the Borg
It accepts a vector (x) that contains the decision variables and returns the objective values'''
import numpy as np
import sys
import data_preparation_functions
sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm')

def objective_func_factory(totalField, total_active):

    def objective_func(*x):
        objs = np.zeros((2,))
        constrs = np.zeros((2,))

        # OBJECTIVE - 1
        # round the decision variables to 0 - no sensor and 1 - sensor is placed
        x = np.round(x)
        # objective 1 - minimize number of active sensors. LATER - minimize cost
        objs[0] = np.sum(x)

        # CONSTRAINTS
        # constrain of minimum two sensors (more realistic) and maximum? 50/100/300
        cons1 = 2
        cons2 = 50
        if objs[0] < cons1:
            constrs[0] = 1
        if objs[0] > cons2:
            constrs[1] = 1

        # OBJECTIVE - 2
        sensorIdx = np.argwhere(x).ravel()
        thr, dyR = 1, 1
        PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)

        # maximize min PEDs, given locations of active sensors.
        # scenario_pairs
        [c, idx] = np.unique(np.sort(scenario_pairs[:,2:4],axis=1), axis=0, return_inverse=True)
        min_PED = PEDs.groupby(idx).min()
        mean_PED = np.mean(min_PED)

        objs[1] = -mean_PED
        return objs, constrs

    return objective_func