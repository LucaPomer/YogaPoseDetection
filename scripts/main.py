import os

import cv2

from scripts.helpers.data_manipulation_helpers import get_keypoint_angles, write_data
from scripts.helpers.openpose_accuraccy_testing import get_accuracy, optimal_net_res
from scripts.openposeKeypoints import get_openpose_keypoints
from scripts.preprocessImages import split_file

class_number = 3
input_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/downwarddog'
result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv'

split_file(input_file_path, 15)

net_res_width = -1
net_res_height = 368

# optimal_net_res(input_file_path, 8, 128, 64)

# result_from_openpose = get_openpose_keypoints(net_res_width, net_res_height, input_file_path)

# accuracy = get_accuracy(result_from_openpose)
# print(accuracy)
#
# for item in result_from_openpose:
#     angles = get_keypoint_angles(item.keypoints)
#     angles.append(class_number)
#     write_data(angles, result_file_path)
