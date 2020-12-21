import csv

import numpy as np


def getKeypointAngles(keypoints):
    print(" getting angles ")
    allPoints = keypoints
    nose = allPoints[0]
    neck = allPoints[1]
    rShoulder = allPoints[2]
    rElbow = allPoints[3]
    rWrist = allPoints[4]
    lShoulder = allPoints[5]
    lElbow = allPoints[6]
    lWrist = allPoints[7]
    midHip = allPoints[8]
    rHip = allPoints[9]
    rKnee = allPoints[10]
    rAnkle = allPoints[11]
    lHip = allPoints[12]
    lKnee = allPoints[13]
    lAnkle = allPoints[14]
    rEye = allPoints[15]
    lEye = allPoints[16]
    rEar = allPoints[17]
    lEar = allPoints[18]
    lBigToe = allPoints[19]
    lSmallToe = allPoints[20]
    lHeel = allPoints[21]
    rBigToe = allPoints[22]
    rSmallToe = allPoints[23]
    rHeel = allPoints[24]

    neckToRightShoulder = getAngle(nose, neck, rShoulder)
    print("neck -> right shoulder = " + str(neckToRightShoulder))

    neckToLeftShoulder = getAngle(nose, neck, lShoulder)
    print("neck -> left shoulder = " + str(neckToLeftShoulder))

    neckToRightElbow = getAngle(neck, rShoulder, rElbow)
    print("neck -> right elbow = " + str(neckToRightElbow))

    neckToLeftElbow = getAngle(neck, lShoulder, lElbow)
    print("neck -> left elbow = " + str(neckToLeftElbow))


def getAngle(pointA, pointB, pointC):
    print(pointA, pointB, pointC)
    ba = pointA - pointB
    bc = pointC - pointB
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def writeData(entryArray):  #todo: test and perfect

    with open('dataFormatted.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(entryArray)

    # data = asarray(reformatedKeys)
    # savetxt('data.csv', data, delimiter=',')
    currentData = np.genfromtxt('dataFormatted.csv', delimiter=',')
    # currentData.append(reformatedKeys)
    # savetxt('dataFormatted.csv', currentData, delimiter=',')
