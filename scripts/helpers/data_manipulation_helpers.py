import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.op_skeleton import OpSkeleton


def get_keypoint_angles(keypoints):
    result_angles = []
    all_points = keypoints
    skelton = OpSkeleton(all_points)

    neck_to_right_shoulder = get_angle(skelton.nose, skelton.neck, skelton.rShoulder)
    result_angles.append(neck_to_right_shoulder)

    neck_to_left_shoulder = get_angle(skelton.nose, skelton.neck, skelton.lShoulder)
    result_angles.append(neck_to_left_shoulder)

    neck_to_right_elbow = get_angle(skelton.neck, skelton.rShoulder, skelton.rElbow)
    result_angles.append(neck_to_right_elbow)

    neck_to_left_elbow = get_angle(skelton.neck, skelton.lShoulder, skelton.lElbow)
    result_angles.append(neck_to_left_elbow)

    r_shoulder_to_r_wrist = get_angle(skelton.rShoulder, skelton.rElbow, skelton.rWrist)
    result_angles.append(r_shoulder_to_r_wrist)

    l_shoulder_to_l_wrist = get_angle(skelton.lShoulder, skelton.lElbow, skelton.lWrist)
    result_angles.append(l_shoulder_to_l_wrist)

    nose_to_hip = get_angle(skelton.nose, skelton.neck, skelton.midHip)
    result_angles.append(nose_to_hip)

    neck_to_r_knee = get_angle(skelton.neck, skelton.midHip, skelton.rKnee)
    result_angles.append(neck_to_r_knee)

    neck_to_l_knee = get_angle(skelton.neck, skelton.midHip, skelton.lKnee)
    result_angles.append(neck_to_l_knee)

    r_hip_to_r_foot = get_angle(skelton.rHip, skelton.rKnee, skelton.rAnkle)
    result_angles.append(r_hip_to_r_foot)

    l_hip_to_l_foot = get_angle(skelton.lHip, skelton.lKnee, skelton.lAnkle)
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

