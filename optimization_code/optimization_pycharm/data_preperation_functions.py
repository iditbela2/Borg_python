import numpy as np
from itertools import combinations
import conf_class_func

def initializeSimulation(numOfSources, emissionRates, distanceBetweenSensors, distanceFromSource):
    '''Initialize Q_source and sensorArray. sensorArray is given Nan in places to exclude
    numOfSources = number of sources defined in the configuration file,
    emissionRates = an array of emission rates,
    distanceBetweenSensors = how many meters appart to place sensors,
    distanceFromSource = radious [meters] around the source where I can't place my sensors at'''

    configFile = conf_class_func.Config()
    boundery = configFile.get_property('GRID_SIZE')

    # get source locations
    sourceLocations = configFile.get_property('SOURCE_LOC')

    # set source emission rates
    Q_source = np.zeros((numOfSources))
    for i in range(1, numOfSources + 1):
        Q_source[i - 1] = configFile.set_property('Q_SOURCE_' + str(i), emissionRates[i - 1])

    # place potential sensors all over the area, distanceBetweenSensors meter appart
    [X, Y] = np.meshgrid(np.arange(0, boundery + distanceBetweenSensors, distanceBetweenSensors),
                         np.arange(0, boundery + distanceBetweenSensors, distanceBetweenSensors))

    # exclude sensors near sources (distanceFromSource to each direction)
    [exclude_x, exclude_y] = np.meshgrid(np.arange(-distanceFromSource, 2 * distanceFromSource, distanceFromSource),
                                         np.arange(-distanceFromSource, 2 * distanceFromSource, distanceFromSource))
    exclude = np.zeros((np.size(exclude_x), 2))
    exclude[:, 0] = np.reshape(exclude_x, (np.size(exclude_x, 0) * np.size(exclude_x, 1),), order='F')
    exclude[:, 1] = np.reshape(exclude_y, (np.size(exclude_y, 0) * np.size(exclude_y, 1),), order='F')

    sensors_to_exclude = []  # list
    for i in range(numOfSources):
        sensors_to_exclude.append(np.matlib.repmat(sourceLocations[i, :], np.size(exclude, 0), 1) + exclude)
    # convert to numpy array
    sensors_to_exclude = np.reshape(sensors_to_exclude, (np.size(exclude, 0) * numOfSources, 2))

    # all original sensors
    NumOfSensors = np.size(X)
    sensorArray = np.zeros((NumOfSensors, 3))
    sensorArray[:, 0] = np.reshape(X, (np.size(X, 0) * np.size(X, 1),), order='F')
    sensorArray[:, 1] = np.reshape(Y, (np.size(Y, 0) * np.size(Y, 1),), order='F')
    #     original_sensor_array = sensorArray

    # these are the sensors to exclude
    a = [(row == sensorArray[:, 0:2]) for row in sensors_to_exclude]
    ind_to_exclude = []
    for i in range(np.shape(a)[0]):
        temp = a[:][i][:]
        for idx, j in enumerate(temp):
            if np.sum(j) == 2:
                ind_to_exclude.append(idx)

    sensorArray[ind_to_exclude, 2] = np.nan

    return Q_source, sensorArray, sensors_to_exclude


def calcSensorsReadings(Q_source, sensorArray, WD, WS, ASC):
    '''The function calculates the readings in places specified in sensorArray,
    according to Q_source, WD, WS and ASC, FOR ALL POSSIBLE COMBINATIONS OF ACTIVE/INACTIVE Q.
    totalField is nXm matrix with n=number of sensors and m=all scenarios (2^(number of sources))
    total_s is all possible scenarios (0 = source inactive, 1=source active)'''

    numOfSensors = np.size(sensorArray,0)
    numOfSources = np.size(Q_source)
    # Run all 32 options of working/not working stacks (for 5 sources) and calculate readings
    # ((including the change in readings from zeros))
    totalField = np.zeros((numOfSensors, 2**numOfSources))

    No_of_Bits = np.size(Q_source)
    a = []
    for bits in np.arange(0, 2 ** No_of_Bits):
        a.append(np.binary_repr(bits, width=No_of_Bits))
    total_s = []
    for row in a:
        s = []
        for binary in row:
            s.append(float(binary))
        total_s.append(s)

    for scenario, active in enumerate(total_s):
        sensorArray, _, _ = conf_class_func.calculateDisp(Q_source * active, sensorArray, WD, WS, ASC)
        total_s[scenario] = np.sum(active)
        totalField[:, scenario] = sensorArray[:, 2]
    #     total_cr = totalField
    return totalField, total_s


def calcSensorsPED(totalField, total_s, sensorIdx, thr, dyR):
    '''WF???not sure what to do with it..., The function calculates all PEDs only for situations of different number of active sources,
    given an array of sensor locations (location index(row) of sensorArray) and an array of sensors sensitivity
    (thr) and dynamic range (accordingly). The function should be used during the optimization process.
    PEDs is nX1 matrix where n=number of situations with different number of active sources.
    Scenario pairs specifies each pair of scenarios corresponding to the PED row in PEDs and the
    number of active/inactive sources in those scenarios (3,4) and number of scenario combination(1,2)'''

    # choose the relevant rows(=sensors) according to sensorIdx
    chosen_readings = totalField[sensorIdx, :]
    numOfSensors = np.size(sensorIdx)
    numOfSources = np.size(total_s, 1)
    total_active = np.sum(total_s, axis=1)

    # (comment these lines if thr and dyR not applied)
    # for each row (sensor), apply thr and dyR
    for i in range(numOfSensors):
        chosen_readings[i, chosen_readings[i, :] < thr[i]] = 0
        chosen_readings[i, chosen_readings[i, :] > thr[i] * dyR[i]] = thr[i] * dyR[i]

    # calculate PEDs for the chosen sensors
    PEDs = []
    comb = list(combinations(range(len(total_active)), 2))
    scenario_pairs = []
    for i, j in comb:
        if total_active[i] != total_active[j]:
            scenario_pairs.append((i, j, total_active[i], total_active[j]))
    for i, j, m, n in scenario_pairs:
        PEDs.append(np.sqrt(np.sum((chosen_readings[:, i] - chosen_readings[:, j]) ** 2)))

    return PEDs, scenario_pairs
