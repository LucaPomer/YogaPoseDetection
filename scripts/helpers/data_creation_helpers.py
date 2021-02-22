from scripts.helpers.angle_calculation import get_keypoint_angles
from scripts.helpers.data_manipulation_helpers import write_multiple_lines
from scripts.helpers.relative_distance_calculation import get_relative_distances


def angle_calc_and_write_data(result_from_openpose, result_file_path, class_number=None):
    lines = get_angles(result_from_openpose, class_number)
    write_multiple_lines(lines, result_file_path)


def distance_calc_and_write_data(result_from_openpose, result_file_path, class_number=None):
    lines = get_distances(result_from_openpose, class_number)
    write_multiple_lines(lines, result_file_path)


def angle_and_dist_calc_and_write_data(result_from_openpose, result_file_path, class_number=None):
    lines = get_distances_and_angles(result_from_openpose, class_number)
    write_multiple_lines(lines, result_file_path)


def get_angles(result_from_openpose, class_num=None):
    lines = []
    for item in result_from_openpose:
        angles = get_keypoint_angles(item.keypoints)
        if class_num is not None:
            angles.append(class_num)
        lines.append(angles)
    return lines


def get_distances(result_from_openpose, class_num=None):
    lines = []
    for item in result_from_openpose:
        distances = get_relative_distances(item.keypoints)
        if class_num is not None:
            distances.append(class_num)
        lines.append(distances)
    return lines


def get_distances_and_angles(result_from_openpose, class_num=None):
    lines = []
    for item in result_from_openpose:
        features = get_relative_distances(item.keypoints)
        features.extend(get_keypoint_angles(item.keypoints))
        if class_num is not None:
            features.append(class_num)
        lines.append(features)
    return lines
