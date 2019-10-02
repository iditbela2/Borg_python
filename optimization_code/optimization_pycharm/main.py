import numpy as np
import sys
sys.path.append('/Users/iditbela/Documents/Borg_python/Borg_downloaded_code/serial-borg-moea')
from plugins.Python.borg import Borg

sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm')
import conf_class_func
import data_preperation_functions
import objective_function


# (1) initialize the simulation
num_S = 5
emissionRates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
sourceLoc = np.array([[200,300],[300,700],[650,400],[450,200],[200,500]])
distanceBetweenSensors = 50
distanceFromSource = 50

Q_source, sensorArray, sensors_to_exclude = \
    data_preperation_functions.initializeSimulation(num_S,sourceLoc,emissionRates,
                                                    distanceBetweenSensors, distanceFromSource)

# (2) calculate readings
WD, WS, ASC = 270, 4, 2
totalField, total_scenarios = data_preperation_functions.calcSensorsReadings(Q_source, sensorArray, WD, WS, ASC)

# (3) run optimization




