'''This is the objective function that is activated by the Borg
It accepts a vector (x) that contains the decision variables and returns the objective values'''
import numpy as np
import sys
import data_preparation_functions
sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm')

def objective_func_factory(totalField, total_active):

    def objective_func(*x):

        # HETEROGENOUS
        objs = np.zeros((2,))
        constrs = np.zeros((2,))

        # OBJECTIVE - 1
        lenSensArray = len(x)/2
        x1 = np.round(x[:int(lenSensArray)]) # sensor/no-sensor
        x2 = x[int(lenSensArray):] # type of sensor
        # round the decision variables to:
        # 0-1/3 - sensor type 1
        # 1/3-2/3 - sensor type 2
        # 2/3-1 - sensor type 3

        binedges = np.arange(0, 1, 1/3)
        x2 = np.digitize(x2, binedges)

        # data dictionary (multiply in 1e-9 since I looked at the plot in microgram/m^3 to decide these values)
        THR = dict([(1,1e-9*10**-1), (2,1e-9*10**4), (3,1e-9*10**-9)])
        DYR = dict([(1,10**5), (2,10**3), (3,10**16)])
        COST = dict([(1, 50), (2, 500), (3, 5000)])

        # insert THR,DYR to each sensor
        thr = np.vectorize(THR.get)(x2)
        dyR = np.vectorize(DYR.get)(x2)

        # multiply x1 and x2
        cost = np.vectorize(COST.get)(x2)*x1
        # minimize cost of network.
        objs[0] = np.sum(cost)

        # CONSTRAINTS
        # constrain of minimum two sensors (more realistic) and maximum? 50/100/300
        cons1 = 2
        cons2 = 100
        if np.sum(x1) < cons1:
            constrs[0] = 1
        if np.sum(x1) > cons2:
            constrs[1] = 1

        # OBJECTIVE - 2
        sensorIdx = np.argwhere(x1).ravel()
        PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr[sensorIdx], dyR[sensorIdx])

        # maximize min PEDs, given locations of active sensors.
        # scenario_pairs
        [c, idx] = np.unique(np.sort(scenario_pairs[:, 2:4], axis=1), axis=0, return_inverse=True)
        min_PED = PEDs.groupby(idx).min()
        mean_PED = np.mean(min_PED)

        objs[1] = -mean_PED
        return objs, constrs

        ## HOMOGENOUS
        # objs = np.zeros((2,))
        # constrs = np.zeros((2,))
        #
        # # OBJECTIVE - 1
        # # round the decision variables to 0 - no sensor and 1 - sensor is placed
        # x = np.round(x)
        # # objective 1 - minimize number of active sensors. LATER - minimize cost
        # objs[0] = np.sum(x)
        #
        # # CONSTRAINTS
        # # constrain of minimum two sensors (more realistic) and maximum? 50/100/300
        # cons1 = 2
        # cons2 = 100
        # if objs[0] < cons1:
        #     constrs[0] = 1
        # if objs[0] > cons2:
        #     constrs[1] = 1
        #
        # # OBJECTIVE - 2
        # sensorIdx = np.argwhere(x).ravel()
        # thr, dyR = 1, 1
        # PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)
        #
        # # maximize min PEDs, given locations of active sensors.
        # # scenario_pairs
        # [c, idx] = np.unique(np.sort(scenario_pairs[:,2:4],axis=1), axis=0, return_inverse=True)
        # min_PED = PEDs.groupby(idx).min()
        # mean_PED = np.mean(min_PED)
        #
        # objs[1] = -mean_PED
        # return objs, constrs

    return objective_func