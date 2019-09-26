'''This is the objective function that is activated by the Borg
It accepts a vector (x) that contains the decision variables and returns the objective values'''
import numpy as np

def objective_func(x):
    objs = np.zeros((2,))
    constrs = np.zeros((2,))

    # OBJECTIVE - 1
    # round the decision variables to 0 - no sensor and 1 - sensor is placed
    x = np.round(x)
    # objective 1 - minimize number of active sensors
    objs[0] = np.sum(x)

    # CONSTRAINTS
    # constrain of minimum two sensors (more realistic) and maximum? 50/100/300
    cons1 = 2
    cons2 = 100
    if objs[0] < cons1:
        constrs[0] = 1
    if objs[0] > cons2:
        constrs[1] = 1

    # OBJECTIVE - 2
    # sum the sensors to calculate the final PED
    total_summed_PED = np.sqrt(np.sum(total_PED.iloc[:, x == 1], axis=1))

    # maximize min PEDs, given locations of active sensors.
    [c, idx] = np.unique(total_variations[:, 0:2], axis=0, return_inverse=True)
    min_PED = total_summed_PED.groupby(idx).min()
    mean_PED = np.mean(min_PED)

    objs[1] = -mean_PED

    return objs, constrs