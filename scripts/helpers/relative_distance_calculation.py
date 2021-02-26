import math

from scripts.Openpose.op_skeleton import OpSkeleton


def get_relative_distances(keypoints):
    result_distances = []
    skeleton = OpSkeleton(keypoints)

    result_distances.append(calc_distance(skeleton.neck, skeleton.lAnkle))
    result_distances.append(calc_distance(skeleton.neck, skeleton.rAnkle))
    result_distances.append(calc_distance(skeleton.neck, skeleton.lWrist))
    result_distances.append(calc_distance(skeleton.neck, skeleton.rWrist))
    result_distances.append(calc_distance(skeleton.lHip, skeleton.lAnkle))
    result_distances.append(calc_distance(skeleton.rHip, skeleton.rAnkle))
    result_distances.append(calc_distance(skeleton.lWrist, skeleton.lAnkle))
    result_distances.append(calc_distance(skeleton.rWrist, skeleton.rAnkle))
    result_distances.append(calc_distance(skeleton.nose, skeleton.lWrist))
    result_distances.append(calc_distance(skeleton.nose, skeleton.rWrist))

    return result_distances


def calc_distance(point_a, point_b):
    return math.sqrt(((point_b[0] - point_a[0]) ** 2) + ((point_b[1] - point_a[1]) ** 2))
