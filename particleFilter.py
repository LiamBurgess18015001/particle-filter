import random
import math as math
import numpy as np
import matplotlib.pyplot as plt

gateways = [[-25.75307, 28.248],
           [-25.75271, 28.24802],
           [-25.75277, 28.24867]]

sensorSet = [[-25.752857574259657, 28.24808120727539],
             [-25.752857574259657, 28.24814021587372],
             [-25.752852742653914, 28.24819654226303],
             [-25.752847911047994, 28.248259574174877],
             [-25.75285153475245, 28.24831187725067],
             [-25.752880524384196, 28.248364180326462],
             [-25.752920385116287, 28.248408436775208],
             [-25.752951790532148, 28.24845001101494],
             [-25.75295058263169, 28.248509019613266],
             [-25.75290951400886, 28.248547911643982]]

base_factor = -61.1
env_factor = 2


def distanceBetweenCoords(p1, p2):
    p = 0.017453292519943295
    c = math.cos
    a = 0.5 - c((p2[0] - p1[0]) * p) / 2 + c(p1[0] * p) * c(p2[0] * p) * (1 - c((p2[1] - p1[1]) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))*1000


def rssiInfoFromSensor(sensorLocation):
    x = []
    for i in range(0, 3):
        x.append(distanceBetweenCoords(gateways[i], sensorLocation))
    return x


def metersToRSSI(meters):
    if meters <= 1:
        return base_factor
    else:
        return base_factor -10 * env_factor * math.log10(meters)


################################################
# credit: https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain
def point_on_triangle2(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    x, y = random.random(), random.random()
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
        pt1[2] + pt2[2] + pt3[2] / 3
    )


def weightsMeasureRelativetoSensor(sensor, measurePoints):
    n = len(measurePoints)
    weights = []
    for i in range(0, n):
        weights.append(weightDistanceEuclidean(sensor, measurePoints[i]))
    return weights


def weightDistanceEuclidean(sensor, point):
    divider = np.sqrt((sensor[0] - point[0])**2 + (sensor[1] - point[1])**2)
    if divider == 0:
        return 0
    else:
        return 1 / divider


def normalizeWeights(weights):
    n = len(weights)
    sum = 0
    for i in range(0, n):
        sum += weights[i]
    for i in range(0, n):
        weights[i] = weights[i] / sum
    return weights


def sampleByWeight(weights, points, howMany):
    return np.random.choice(points, howMany, p=weights)


def getSamplesFromIndexes(indexes, points):
    n = len(indexes)
    np.sort(indexes)
    newSamples = []
    for i in range(0, n):
        newSamples.append(points[indexes[i]])
    return newSamples


def computeDegeneracy(weights):
    nEff = 0
    for i in range(0, len(weights)):
        nEff += weights[i]**2
    return 1 / nEff


def completeSystematicResampling(samples, weights):
    n = len(weights)
    cumulativeDist = []
    cumulativeDist.append(weights[0])
    for i in range(1, n):
        cumulativeDist.append(weights[i] + cumulativeDist[i-1])

    unif = np.random.uniform(0, 1/n)

    i = 1
    newSamples = []
    for j in range(0, n):
        while unif > cumulativeDist[i]:
            i += 1
        newSamples.append(samples[i])
        unif = unif + 1 / n
    return newSamples


# credit: https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def get_cartesian(point):
    lat = point[0]
    lon = point[1]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return [x,y,z]


# credit: https://stackoverflow.com/questions/56945401/converting-xyz-coordinates-to-longitutde-latitude-in-python
def get_latlon(x,y,z):
    R = 6371
    lat = np.degrees(np.arcsin(z / R))
    lon = np.degrees(np.arctan2(y, x))


def particle_filter_v1():
    pt1 = (28.248, -25.75307)
    pt1 = get_cartesian(pt1)
    pt2 = (28.24802, -25.75271)
    pt2 = get_cartesian(pt2)
    pt3 = (28.24867, -25.75277)
    pt3 = get_cartesian(pt3)
    points = [point_on_triangle2(pt1, pt2, pt3) for _ in range(1000)]
    x, y, z = zip(*points)
    plt.scatter(x, y, s=0.1)
    plt.show()
    for i in range(0, len(sensorSet)):
        for j in range(0, 3):
            weights = weightsMeasureRelativetoSensor(sensorSet[i], points)
            weights = normalizeWeights(weights)
            weightedSample = sampleByWeight(weights, list(range(0, 1000)), 1000)
            points = getSamplesFromIndexes(weightedSample, points)
            x, y, z = zip(*points)
            plt.scatter(x, y, s=0.1)
            plt.show()


# with degeneracy protection
def particle_filter_v2(sampleSize):
    pt1 = (28.248, -25.75307)
    pt1 = get_cartesian(pt1)
    pt2 = (28.24802, -25.75271)
    pt2 = get_cartesian(pt2)
    pt3 = (28.24867, -25.75277)
    pt3 = get_cartesian(pt3)
    points = [point_on_triangle2(pt1, pt2, pt3) for _ in range(sampleSize)]
    x, y, z = zip(*points)
    #plt.scatter(x, y, s=0.1)
    plt.hist2d(x,y)
    plt.show()
    for i in range(0, len(sensorSet)):
        for j in range(0, 1000):
            weights = weightsMeasureRelativetoSensor(sensorSet[i], points)
            weights = normalizeWeights(weights)
            nEff = computeDegeneracy(weights)
            if (sampleSize - nEff) / sampleSize > 0.8:
                print('regen\n')
                points = completeSystematicResampling(points, weights)
                for k in range(0, len(weights)):
                    weights[k] = 1 / sampleSize
            else:
                weightedSample = sampleByWeight(weights, list(range(0, sampleSize)), sampleSize)
                points = getSamplesFromIndexes(weightedSample, points)
        x, y, z = zip(*points)
        plt.hist2d(x, y)
        plt.show()




################################################

particle_filter_v2(10000)

#print(rssiInfoFromSensor([-25.75285636635823,28.24830651283264]))