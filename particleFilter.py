import random
import math as math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

################################################
# gateways = [[-25.75307, 28.248],
#            [-25.75271, 28.24802],
#            [-25.75277, 28.24867]]

# sensorSet = [[-25.752857574259657, 28.24808120727539],
#              [-25.752857574259657, 28.24814021587372],
#              [-25.752852742653914, 28.24819654226303],
#              [-25.752847911047994, 28.248259574174877],
#              [-25.75285153475245, 28.24831187725067],
#              [-25.752880524384196, 28.248364180326462],
#              [-25.752920385116287, 28.248408436775208],
#              [-25.752951790532148, 28.24845001101494],
#              [-25.75295058263169, 28.248509019613266],
#              [-25.75290951400886, 28.248547911643982]]

################################################

# gateways = [[28.251326680183407,
#              -25.750026219611552],
#             [28.250766098499298,
#              -25.7485911898637],
#             [28.249990940093994,
#              -25.74900189038381]]

# sensorSet = [[28.25106382369995,
#               -25.74964209718626],
#              [28.25086534023285,
#               -25.749335281526573],
#              [28.250790238380432,
#               -25.748929413924614],
#              [28.25038254261017,
#               -25.748861769189467],
#              [28.25036108493805,
#               -25.74919032897078]]

################################################

gateways = [
            [
              28.250527381896973,
              -25.748791708530312
            ],
            [
              28.249475955963135,
              -25.748482474782456
            ],
            [
              28.249014616012573,
              -25.74928454815217
            ],
            [
              28.250334262847897,
              -25.749487481519804
            ],
            [
              28.250527381896973,
              -25.748791708530312
            ]
          ]

sensorSet = [
          [
            28.250248432159424,
            -25.749390846626067
          ],
          [
            28.24955105781555,
            -25.749313538654487
          ],
          [
            28.248800039291382,
            -25.749265221146686
          ],
          [
            28.24913263320923,
            -25.749004306265018
          ],
          [
            28.249422311782837,
            -25.748704736620095
          ],
          [
            28.249701261520386,
            -25.74850180191528
          ],
          [
            28.25006604194641,
            -25.748695073070593
          ],
          [
            28.250291347503662,
            -25.74901396978938
          ],
          [
            28.24986219406128,
            -25.74901396978938
          ],
          [
            28.249958753585815,
            -25.749158922560373
          ],
          [
            28.249980211257935,
            -25.749487481519804
          ]
        ]

reservePerimeter = [
            [
              28.25060248374939,
              -25.749709741477048
            ],
            [
              28.25075268745422,
              -25.7486177646462
            ],
            [
              28.249357938766476,
              -25.748182903821597
            ],
            [
              28.248413801193237,
              -25.74940051011898
            ],
            [
              28.25060248374939,
              -25.749709741477048
            ]
          ]

minMax = [[],[]]


base_factor = -61.1
env_factor = 2


def distanceBetweenCoords(p1, p2):
    p = 0.017453292519943295
    c = math.cos
    a = 0.5 - c((p2[0] - p1[0]) * p) / 2 + c(p1[0] * p) * c(p2[0] * p) * (1 - c((p2[1] - p1[1]) * p)) / 2
    return 12742 * math.asin(math.sqrt(a)) * 1000


def rssiInfoFromSensor(sensorLocation):
    x = []
    for i in range(0, 3):
        x.append(distanceBetweenCoords(gateways[i], sensorLocation))
    return x


def metersToRSSI(meters):
    if meters <= 1:
        return base_factor
    else:
        return base_factor - 10 * env_factor * math.log10(meters)


def RSSItoMeters(RSSI):
    return 10 ^ ((-69 - RSSI) / (10 * 2))


# credit: https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def get_cartesian(point):
    lat = point[0]
    lon = point[1]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371  # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return [x, y, z]


# credit: https://stackoverflow.com/questions/56945401/converting-xyz-coordinates-to-longitutde-latitude-in-python
def get_latlon(x, y, z):
    R = 6371
    lat = np.degrees(np.arcsin(z / R))
    lon = np.degrees(np.arctan2(y, x))



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
        # pt1[2] + pt2[2] + pt3[2] / 3
    )

# credit: https://www.matecdev.com/posts/random-points-in-polygon.html
def points_in_polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt.coords[:][0])
    return points[:]


def minMaxLatLong():
    # small lat
    minMax[0][0] = gateways[0][1]
    # large lat
    minMax[0][1] = gateways[0][1]
    # small long
    minMax[1][0] = gateways[0][0]
    # large long
    minMax[1][1] = gateways[0][0]
    for i  in range(1, len(gateways)):
        if gateways[i][1] < minMax[0][0]:
            minMax[0][0] = gateways[i][1]
        elif gateways[i][1] > minMax[0][1]:
            minMax[0][1] = gateways[i][1]
        if gateways[i][0] < minMax[1][0]:
            minMax[1][0] = gateways[i][0]
        elif gateways[i][0] > minMax[1][1]:
            minMax[1][1] = gateways[i][0]


def weightsMeasureRelativetoSensor(OriginalPoint, RandomPoints):
    n = len(RandomPoints)
    weights = []
    for i in range(0, n):
        # Get Dists of Particle
        RandomParticleToCompare = []
        for j in range(0, len(gateways)):
            RandomParticleToCompare.append(distanceBetweenCoords(RandomPoints[i], gateways[j]))
        # Weight against original
        weights.append(weightDistanceEuclidean(OriginalPoint, RandomParticleToCompare))
    return weights


def weightDistanceEuclidean(sensor, point):
    sums = 0
    # for i in range(0, len(sensor)):
    #     abs(sensor[i] - point[i])
    for i in range(0, len(sensor)):
        sums += (sensor[i] - point[i])**2
    divider = np.sqrt(sums)
    # bound the upper limit
    if divider < 0.000001:
        return 100000
    else:
        return 1 / divider


def normalizeWeights(weights):
    n = len(weights)
    sum = np.sum(weights)
    # for i in range(0, n):
    #    sum += weights[i]
    for i in range(0, n):
        weights[i] = weights[i] / sum
    return weights


def sampleByWeight(weights, points, howMany):
    return np.random.choice(points, howMany, p=weights)

#######################################################


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
        nEff += weights[i] ** 2
    return 1 / nEff

##########################################################


def completeSystematicResampling(samples, weights):
    n = len(weights)
    cumulativeDist = []
    cumulativeDist.append(weights[0])
    for i in range(1, n):
        cumulativeDist.append(weights[i] + cumulativeDist[i - 1])

    unif = np.random.uniform(0, 1 / n)

    i = 1
    newSamples = []
    for j in range(0, n):
        while unif > cumulativeDist[i]:
            i += 1
        newSamples.append(samples[i])
        unif = unif + 1 / n
    return newSamples


def multinomialResampling(points, weights):
    newPoints = []
    N_samples = len(points)
    n = 0

    cumulativeDist = []
    cumulativeDist.append(weights[0])
    for i in range(1, N_samples):
        cumulativeDist.append(weights[i] + cumulativeDist[i - 1])

    while n < N_samples:
        u = np.random.uniform(0, 1)
        m = 0

        while cumulativeDist[m] < u:
            m += 1
        n += 1
        newPoints.append(points[m])

    for i in range(0, N_samples):
        weights[i] = 1 / N_samples
    return newPoints, weights


def stratifiedResampling(points, weights):
    newPoints = []
    N_samples = len(points)
    n = 0

    cumulativeDist = []
    cumulativeDist.append(weights[0])
    for i in range(1, N_samples):
        cumulativeDist.append(weights[i] + cumulativeDist[i - 1])

    m = 1

    while n < N_samples:
        u0 = np.random.uniform(0,1/N_samples)
        u = u0 + n / N_samples

        while cumulativeDist[m] < u:
            m += 1
        n += 1
        newPoints.append(points[m])

    for i in range(0, N_samples):
        weights[i] = 1 / N_samples
    return newPoints, weights


####################################################################

def randomWalk(points):
    N = len(points)
    newPoint = []
    for i in range(0, N):
        rand1 = np.random.uniform(0.0001, 0.001)
        rand2 = np.random.uniform(0.0001, 0.001)
        choice = np.random.randint(0,4)
        if choice == 0:
            newPoint.append([points[i][0] + rand1, points[i][1] + rand2])
        elif choice == 1:
            newPoint.append([points[i][0] + rand1, points[i][1] - rand2])
        elif choice == 2:
            newPoint.append([points[i][0] - rand1, points[i][1] + rand2])
        else:
            newPoint.append([points[i][0] - rand1, points[i][1] - rand2])
    #     if np.random.randint(0, 1):
    #         newPoint.append([points[i][0] + rand, points[i][1] - rand])
    #     else:
    #         newPoint.append([points[i][0] + rand, points[i][1] - rand])
    return newPoint


####################################################################

def resampleProportion(points, howMany):
    N = len(points)
    newPoints = [point_on_triangle2(gateways[0], gateways[1], gateways[2]) for _ in range(N - howMany)]
    oldPoints = points[:howMany]
    newPoints = newPoints + oldPoints
    return newPoints


def relativeError(point, estimate):
    error = []
    for i in range(0, len(point)):
        error.append((abs(point[i] - estimate[i]) / abs(point[i]))*1000000)
    return error


def printGeoJson(point):
    print('{"type": "Feature","properties": {"marker-color": "#d62929","marker-size": "medium","marker-symbol": ""},"geometry": {"type": "Point","coordinates": ['+str(point[0])+','+str(point[1])+']}},')


def genGeoJsonFile(points, iteration):
    with open("Geojson/"+str(iteration)+".geojson", "w") as file:
        for i in range(0, len(points)):
            file.write('{"type": "Feature","properties": {"marker-color": "#d62929","marker-size": "medium","marker-symbol": ""},"geometry": {"type": "Point","coordinates": ['+str(points[i][0])+','+str(points[i][1])+']}},')
    file.close()

####################################################################

# def particle_filter_v1():
#     pt1 = (28.248, -25.75307)
#     pt1 = get_cartesian(pt1)
#     pt2 = (28.24802, -25.75271)
#     pt2 = get_cartesian(pt2)
#     pt3 = (28.24867, -25.75277)
#     pt3 = get_cartesian(pt3)
#     points = [point_on_triangle2(pt1, pt2, pt3) for _ in range(1000)]
#     x, y, z = zip(*points)
#     plt.scatter(x, y, s=0.1)
#     plt.show()
#     for i in range(0, len(sensorSet)):
#         for j in range(0, 3):
#             weights = weightsMeasureRelativetoSensor(sensorSet[i], points)
#             weights = normalizeWeights(weights)
#             weightedSample = sampleByWeight(weights, list(range(0, 1000)), 1000)
#             points = getSamplesFromIndexes(weightedSample, points)
#             x, y, z = zip(*points)
#             plt.scatter(x, y, s=0.1)
#             plt.show()


####################################################
# with degeneracy adjustments
# def particle_filter_v2(sampleSize):
#     # sample uniform particles
#     points = [point_on_triangle2(gateways[0], gateways[1], gateways[2]) for _ in range(sampleSize)]
#     lats, longs = zip(*points)
#     plt.scatter(longs, lats, s=0.1)
#     # plt.hist2d(longs, lats)
#     plt.show()
#
#     pointIter = []
#
#     # change reading
#     for i in range(4, len(sensorSet)):
#         # get RSSI of reading, here I use the diff of lat long for simulation
#         readingMeasurement = []
#         readingMeasurement.append(distanceBetweenCoords(sensorSet[i], gateways[0]))
#         readingMeasurement.append(distanceBetweenCoords(sensorSet[i], gateways[1]))
#         readingMeasurement.append(distanceBetweenCoords(sensorSet[i], gateways[2]))
#
#         # random walk
#         points = randomWalk(points)
#
#         # train to reading
#         for j in range(0, 100):
#
#             # get weights based on closeness to original reading
#             weights = weightsMeasureRelativetoSensor(readingMeasurement, points)
#             # Normalize to a distribution
#             weights = normalizeWeights(weights)
#
#             # Check for Degeneracy
#             nEff = computeDegeneracy(weights)
#             # print("Neff: ", nEff)
#             # if (sampleSize - nEff) / sampleSize > 0.8:
#
#             if nEff < sampleSize / 2:
#                 # points = completeSystematicResampling(points, weights)
#                 points, weights = stratifiedResampling(points, weights)
#             else:
#                 # Generate Indexes of new Sample
#                 weightedSample = sampleByWeight(weights, list(range(0, sampleSize)), sampleSize)
#
#                 # Create new sample from indexes
#                 points = getSamplesFromIndexes(weightedSample, points)
#
#             #points = resampleProportion(points, int(np.floor(sampleSize / 8)))
#
#         #points = randomWalk(points)
#         #print("iter: ", i, "\n")#, points[0:5])
#         printGeoJson(points[1])
#         lats, longs = zip(*points)
#         plt.scatter(longs, lats, s=0.8)
#         # plt.hist2d(longs, lats)
#         plt.show()

################################################
# Random walk, polygon sampling and more than three gateway measurements
def particle_filter_v3(sampleSize):
    errors = []
    # sample uniform particles
    points = points_in_polygon(Polygon(reservePerimeter), sampleSize)
    lats, longs = zip(*points)
    # plt.scatter(longs, lats, s=0.1)
    plt.hist2d(longs, lats)
    plt.show()

    pointIter = []

    # change reading
    for i in range(0, len(sensorSet)):
        iterationError = []
        readingMeasurement = []

        # get RSSI of reading, here I use the diff of lat long for simulation
        for j in range(0, len(gateways)):
            readingMeasurement.append(distanceBetweenCoords(sensorSet[i], gateways[j]))

        # random walk
        points = randomWalk(points)

        # train to reading
        for j in range(0, 50):

            # get weights based on closeness to original reading
            weights = weightsMeasureRelativetoSensor(readingMeasurement, points)
            # Normalize to a distribution
            weights = normalizeWeights(weights)

            # Check for Degeneracy
            nEff = computeDegeneracy(weights)
            if nEff < sampleSize / 4:
                #print("regen")
                points, weights = stratifiedResampling(points, weights)
            else:
                # Generate Indexes of new Sample
                weightedSample = sampleByWeight(weights, list(range(0, sampleSize)), sampleSize)

                # Create new sample from indexes
                points = getSamplesFromIndexes(weightedSample, points)

            max = np.argmax(weights)
            iterationError.append(relativeError(sensorSet[i], points[max]))

            #points = resampleProportion(points, int(np.floor(sampleSize / 8)))

        #genGeoJsonFile(points, i)

        # error graphing
        # referenceX = [sensorSet[i][1]] * 8
        # referenceY = [sensorSet[i][0]] * 8
        # y, x = zip(*iterationError)
        # plt.plot(range(0, 10), x, color='r', label='')
        # plt.plot(range(0, 10), y, color='g', label='')
        # plt.show()

        # printGeoJson(points[1])
        max = np.argmax(weights)
        #printGeoJson(points[max])
        lats, longs = zip(*points)
        plt.imshow((lats, longs), cmap="hot", interpolation="nearest")
        #plt.scatter(longs, lats, s=0.8)
        #plt.hist2d(longs, lats)

        plt.show()
        #errors.append(relativeError(sensorSet[i], points[max]))
    #print(errors)



################################################


random.seed(1000)
np.random.seed(1000)
particle_filter_v3(10000)

# print(rssiInfoFromSensor([-25.75285636635823,28.24830651283264]))

