import csv

import numpy as np


def get_keypoint_angles(keypoints):
    print(" getting angles ")
    result_angles = []
    all_points = keypoints
    nose = all_points[0]
    neck = all_points[1]
    rShoulder = all_points[2]
    rElbow = all_points[3]
    rWrist = all_points[4]
    lShoulder = all_points[5]
    lElbow = all_points[6]
    lWrist = all_points[7]
    midHip = all_points[8]
    rHip = all_points[9]
    rKnee = all_points[10]
    rAnkle = all_points[11]
    lHip = all_points[12]
    lKnee = all_points[13]
    lAnkle = all_points[14]
    rEye = all_points[15]
    lEye = all_points[16]
    rEar = all_points[17]
    lEar = all_points[18]
    lBigToe = all_points[19]
    lSmallToe = all_points[20]
    lHeel = all_points[21]
    rBigToe = all_points[22]
    rSmallToe = all_points[23]
    rHeel = all_points[24]

    neck_to_right_shoulder = get_angle(nose, neck, rShoulder)
    #print("neck -> right shoulder = " + str(neck_to_right_shoulder))
    result_angles.append(neck_to_right_shoulder)

    neck_to_left_shoulder = get_angle(nose, neck, lShoulder)
    #print("neck -> left shoulder = " + str(neck_to_left_shoulder))
    result_angles.append(neck_to_left_shoulder)

    neck_to_right_elbow = get_angle(neck, rShoulder, rElbow)
    #print("neck -> right elbow = " + str(neck_to_right_elbow))
    result_angles.append(neck_to_right_elbow)

    neck_to_left_elbow = get_angle(neck, lShoulder, lElbow)
    #print("neck -> left elbow = " + str(neck_to_left_elbow))
    result_angles.append(neck_to_left_elbow)

    r_shoulder_to_r_wrist = get_angle(rShoulder, rElbow, rWrist)
    #print("right shoulder -> right wrist = " + str(r_shoulder_to_r_wrist))
    result_angles.append(r_shoulder_to_r_wrist)

    l_shoulder_to_l_wrist = get_angle(lShoulder, lElbow, lWrist)
    #print("left shoulder -> left wrist = " + str(l_shoulder_to_l_wrist))
    result_angles.append(l_shoulder_to_l_wrist)

    nose_to_hip = get_angle(nose, neck, midHip)
    #print("nose -> hip = " + str(nose_to_hip))
    result_angles.append(nose_to_hip)

    neck_to_r_knee = get_angle(neck, midHip, rKnee)
    #print("neck -> right knee = " + str(neck_to_r_knee))
    result_angles.append(neck_to_r_knee)

    neck_to_l_knee = get_angle(neck, midHip, lKnee)
    #print("neck -> left knee = " + str(neck_to_l_knee))
    result_angles.append(neck_to_l_knee)

    return result_angles

def get_angle(point_a, point_b, point_c):
    #print(point_a, point_b, point_c)
    if point_a.all() == 0 or point_b.all() == 0 or point_c.all() == 0:
        return 0.0
    ba = point_a - point_b
    bc = point_c - point_b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def write_data(entry_array):  # todo: test and perfect

    with open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/data_with_angles_new.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(entry_array)

    # data = asarray(reformatedKeys)
    # savetxt('data.csv', data, delimiter=',')
    #currentData = np.genfromtxt('../../dataFormatted.csv', delimiter=',')
    # currentData.append(reformatedKeys)
    # savetxt('dataFormatted.csv', currentData, delimiter=',')
