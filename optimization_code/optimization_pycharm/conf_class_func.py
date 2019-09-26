import numpy as np

# configurations to load
conf = {
    # general
    'GRID_SIZE': 1000, #meters
    'GRID_SCALE': 1,
    'EFFECTIVE_HEIGHT': 10,
    'NOISE_RANGE': 1e-16,

    # meteorological properties
    'WIND_SPEED': [],
    'WIND_DIRECTION': [],
    'STABILITY_PARAMETER_a': [],
    'STABILITY_PARAMETER_c': [],
    'STABILITY_PARAMETER_d': [],
    'STABILITY_PARAMETER_f': [],

    # source locations (should be configured according to number of sources)
    'SOURCE_LOC': np.array([[200,300],[300,700],[650,400],[450,200],[200,500]]),

    # leak rates[kg/sec]
    'Q_SOURCE': np.array([])
}

class Config(object):
    def __init__(self):
        self._config = conf # set it to conf

    def get_property(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    def set_property(self, property_name, value):
        self._config[property_name] = value

class API:
    # properties
    configFile = Config()

    # methods
    def distance(*arg, x1_cor, y1_cor, x2_cor, y2_cor):
        d = np.sqrt((x1_cor - x2_cor) ** 2 + (y1_cor - y2_cor) ** 2)
        return d

    def getCurveByWindDirection(*arg, windDirection):
        '''the function returns the slope of the vector from wind direction [degrees].
        alpha is the angle between the vector of the wind and the positive direction of X axis'''
        omega = windDirection
        alpha = 270 - omega
        # convert to radians
        alpha = np.deg2rad(alpha)
        m = np.tan(alpha)
        return m

        # return the equation of a line by the slope and known point
        # the line is represented by m(the slope) and n (cut-off point with Y axis)
        # the line equation is : y-y1 = m*(x-x1) =>
        # y = m*(x-x1) + y1 =>
        # n = -x1*m + y1

    def getLineEquation(*arg, x1_cor, y1_cor, m):
        '''the function returns the equation of a line by the slope=m and known point (x1,y1).
        n is the intercept with Y axis. the line equation is: y-y1 = m*(x-x1) => y = m*(x-x1)+y1,
        and when x=0 we get n:'''
        n = y1_cor - m * x1_cor
        return m, n

    #       get distance between line and point. the line represented by m
    #       and n, the point is x and y
    #       the equation is: d = (abs (-mx0 + y0 - n))/(sqrt(m^2 +1))

    def getDistanceFromPointToLine(*arg, m, n, x0, y0):
        '''get distance between a line and a point, the line is represented by m (slope) and n (equation),
        the point is x and y. the equation is d'''
        d = (np.abs(-m * x0 + y0 - n)) / (np.sqrt(m ** 2 + 1))  # 1 is the coefficient of y in the line equation
        return d

    def calculateSensorCon(obj, x_sen, y_sen, x_source, y_source, sourceCon):
        '''calculate the sensor concentration level based on given emission from source'''
        Q = sourceCon
        U = obj.configFile.get_property('WIND_SPEED')
        wind_direction = obj.configFile.get_property('WIND_DIRECTION')
        # effective stack height, aribtrrary
        He = obj.configFile.get_property('EFFECTIVE_HEIGHT')
        grid_scale = obj.configFile.get_property('GRID_SCALE')

        # first: calculate the distance between the source and sensor
        # deal with all winds directions
        if (np.mod(180 - wind_direction - np.rad2deg(np.arctan2(y_sen - y_source, x_sen - x_source)), 360) < 180):
            return 0

        # X distance: In meters
        x_distance_meters = obj.distance(x_sen, y_sen, x_source, y_source) * grid_scale
        x_distance_kilometers = (x_distance_meters / 1000)

        # the slope of the wind direction refere to X axis
        mWD = obj.getCurveByWindDirection(wind_direction)

        # get the line equation of the wind, based on two points (the source) and the slope
        [mWD, nWD] = obj.getLineEquation(x_source, y_source, mWD)

        # calculate the distance between the sensor and the line
        # equation of the wind direction
        y_distance = (obj.getDistanceFromPointToLine(mWD, nWD, x_sen, y_sen))

        # in meters
        y_distance = y_distance * grid_scale

        # calculate sigma y and sigma z
        a = obj.configFile.get_property('STABILITY_PARAMETER_a')
        c = obj.configFile.get_property('STABILITY_PARAMETER_c')
        d = obj.configFile.get_property('STABILITY_PARAMETER_d')
        f = obj.configFile.get_property('STABILITY_PARAMETER_f')

        sigma_y = a * (x_distance_kilometers ** 0.894)
        sigma_z = c * (x_distance_kilometers ** d) + f

        C = (Q / (U * sigma_y * sigma_z * np.pi)) * (
                    np.exp(-(y_distance ** 2) / (2 * (sigma_y ** 2))) * np.exp(-(He ** 2) / (2 * (sigma_z ** 2))))
        return C

    def addGausienNoise(obj, oldSignal):
        # noise level
        noiseRange = obj.configFile.get_property('NOISE_RANGE')
        noiseLevelA = -(noiseRange)
        noiseLevelB = noiseRange
        noise = noiseLevelA + (noiseLevelB - noiseLevelA) * rand

        newSignal = oldSignal + noise * oldSignal

        return newSignal


def calculateDisp(Q_source, sensorArray, WD, WS, SC):  # in matlab called idit_CD1

    svAPI = API()
    svAPI.configFile.set_property('WIND_DIRECTION', WD)
    svAPI.configFile.set_property('WIND_SPEED',WS)

    # stability parameters according to Martin 1976
    if SC == 1:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',213)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',440.8)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',1.941)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',9.27)

    if SC == 2:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',156)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',106.6)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',1.149)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',3.3)

    if SC == 3:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',104)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',61)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',0.911)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',0)

    if SC == 4:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',68)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',33.2)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',0.725)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',-1.7)

    if SC == 5:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',50.5)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',22.8)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',0.678)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',-1.3)

    if SC == 6:
        svAPI.configFile.set_property('STABILITY_PARAMETER_a',34)
        svAPI.configFile.set_property('STABILITY_PARAMETER_c',14.35)
        svAPI.configFile.set_property('STABILITY_PARAMETER_d',0.740)
        svAPI.configFile.set_property('STABILITY_PARAMETER_f',-0.035)

    configFile = Config()
    sizeOfStudyArea = configFile.get_property('GRID_SIZE')

    # Source locations (!!! maybe try to enable flexibility in number of sources and locations!!!)
    sourceLocations = configFile.get_property('SOURCE_LOC')

    sourceArray = np.zeros((np.size(Q_source), 3))
    for i in range(np.size(Q_source)):
        sourceArray[i, 1] = sourceLocations[i,0]  # x coord
        sourceArray[i, 2] = sourceLocations[i,1] # Y coord
        sourceArray[i, 3] = Q_source[i]

    for i in range(np.size(sensorArray, 0)):
        totalAmbientDataOfSensor = 0
        for j in range(np.size(sourceArray, 0)):
            ambientDataFromOneSource = svAPI.calculateSensorCon(sensorArray[i, 1], sensorArray[i, 2], sourceArray[j, 1],
                                                                sourceArray[j, 2], sourceArray[j, 3])
            ambientDataFromOneSource = svAPI.addGausienNoise(ambientDataFromOneSource)
            totalAmbientDataOfSensor = totalAmbientDataOfSensor + ambientDataFromOneSource
        sensorArray[i, 3] = totalAmbientDataOfSensor

    return sensorArray, sourceArray, sizeOfStudyArea

