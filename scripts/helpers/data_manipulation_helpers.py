import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_keypoint_angles(keypoints):
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
    result_angles.append(neck_to_right_shoulder)

    neck_to_left_shoulder = get_angle(nose, neck, lShoulder)
    result_angles.append(neck_to_left_shoulder)

    neck_to_right_elbow = get_angle(neck, rShoulder, rElbow)
    result_angles.append(neck_to_right_elbow)

    neck_to_left_elbow = get_angle(neck, lShoulder, lElbow)
    result_angles.append(neck_to_left_elbow)

    r_shoulder_to_r_wrist = get_angle(rShoulder, rElbow, rWrist)
    result_angles.append(r_shoulder_to_r_wrist)

    l_shoulder_to_l_wrist = get_angle(lShoulder, lElbow, lWrist)
    result_angles.append(l_shoulder_to_l_wrist)

    nose_to_hip = get_angle(nose, neck, midHip)
    result_angles.append(nose_to_hip)

    neck_to_r_knee = get_angle(neck, midHip, rKnee)
    result_angles.append(neck_to_r_knee)

    neck_to_l_knee = get_angle(neck, midHip, lKnee)
    result_angles.append(neck_to_l_knee)

    r_hip_to_r_foot = get_angle(rHip, rKnee, rAnkle)
    result_angles.append(r_hip_to_r_foot)

    l_hip_to_l_foot = get_angle(lHip, lKnee, lAnkle)
    result_angles.append(l_hip_to_l_foot)

    return result_angles


def get_angle(point_a, point_b, point_c):
    # print(point_a, point_b, point_c)
    if point_a.all() == 0 or point_b.all() == 0 or point_c.all() == 0:
        return 0.0
    ba = point_a - point_b
    bc = point_c - point_b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def write_data(entry_array, file_name):
    with open(file_name, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(entry_array)


def split_data(data, classes):
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.33, random_state=42)

    print(y_train)
    # df = pd.read_csv(file_path)
    # df['split'] = np.random.randn(df.shape[0], 1)
    #
    # msk = np.random.rand(len(df)) <= 0.7
    #
    # train = df[msk]
    # test = df[~msk]
    #
    # print(test)
