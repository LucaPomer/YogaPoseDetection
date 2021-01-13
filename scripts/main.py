import os

import cv2

from scripts.helpers.data_manipulation_helpers import get_keypoint_angles, write_data
from scripts.helpers.openpose_accuraccy_testing import get_accuracy, optimal_net_res
from scripts.openposeKeypoints import get_openpose_keypoints
from scripts.preprocess_image_folders import split_file, get_file_class_num


input_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/downwarddog'
result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/anglesNew.csv'

# split_file(input_file_path, 15)
class_number = get_file_class_num(input_file_path)
net_res_width = 656
net_res_height = 368


# result_from_openpose = get_openpose_keypoints(net_res_width, net_res_height, input_file_path)
optimal_net_res(input_file_path, 8, 128, 64)
for batch_folder in os.listdir(input_file_path):
    full_folder_path = input_file_path + '/' + batch_folder
    print(full_folder_path)
    result_from_openpose = await get_openpose_keypoints(net_res_width, net_res_height, full_folder_path)
    for item in result_from_openpose:
        angles = get_keypoint_angles(item.keypoints)
        angles.append(class_number)
        write_data(angles, result_file_path)

# accuracy = get_accuracy(result_from_openpose)
# print(accuracy)
#
# for item in result_from_openpose:
#     angles = get_keypoint_angles(item.keypoints)
#     angles.append(class_number)
#     write_data(angles, result_file_path)
