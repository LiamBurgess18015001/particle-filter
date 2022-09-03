import random
import math as math
import numpy as np
import matplotlib.pyplot as plt

gateways = [[-25.75307, 28.248],
           [-25.75271, 28.24802],
           [-25.75277, 28.24867]]

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
    )


def weightsMeasureRelativetoSensor(sensor, measurePoints):
    n = len(measurePoints)
    weights = []
    for i in range(0, n):
        weights.append(weightDistanceEuclidean(sensor, measurePoints[i]))
    return weights


def weightDistanceEuclidean(sensor, point):
    divider = np.abs((sensor[0]) - np.abs(point[0])) + np.abs(np.abs(sensor[1]) - np.abs(point[1]))
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


def particle_filter():
    pt1 = (28.248, -25.75307)
    pt2 = (28.24802, -25.75271)
    pt3 = (28.24867, -25.75277)
    points = [point_on_triangle2(pt1, pt2, pt3) for _ in range(1000)]
    x, y = zip(*points)
    plt.scatter(x, y, s=0.1)
    plt.show()
    for i in range(0,5):
        weights = weightsMeasureRelativetoSensor([28.24830651283264, -25.75285636635823], points)
        weightedSample = sampleByWeight(normalizeWeights(weights), list(range(0,1000)), 1000)
        points = getSamplesFromIndexes(weightedSample, points)
        x, y = zip(*points)
        plt.scatter(x, y, s=0.1)
        plt.show()




################################################


particle_filter()


#print(rssiInfoFromSensor([-25.75285636635823,28.24830651283264]))