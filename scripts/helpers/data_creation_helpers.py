import os

from scripts.helpers.angle_calculation import get_keypoint_angles
from scripts.helpers.data_manipulation_helpers import write_data
from scripts.helpers.relative_distance_calculation import get_relative_distances
from scripts.openpose_algorithm import run_openpose_algorithm


def run_openpose_and_angle_calc(input_file_path, result_file_path, net_res_width, net_res_height, class_number=None):
    result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, input_file_path)
    for item in result_from_openpose:
        angles = get_keypoint_angles(item.keypoints)
        if class_number is not None:
            angles.append(class_number)
        write_data(angles, result_file_path)


def run_openpose_and_distance_calc(input_file_path, result_file_path, net_res_width, net_res_height, class_number):
    result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, input_file_path)
    for item in result_from_openpose:
        distances = get_relative_distances(item.keypoints)
        if class_number is not None:
            distances.append(class_number)
        write_data(distances, result_file_path)


def run_openpose_with_distance_and_degree_calc(input_file_path, result_file_path, net_res_width, net_res_height,
                                               class_number):
    result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, input_file_path)
    for item in result_from_openpose:
        features = get_relative_distances(item.keypoints)
        features.extend(get_keypoint_angles(item.keypoints))
        if class_number is not None:
            features.append(class_number)
        write_data(features, result_file_path)
