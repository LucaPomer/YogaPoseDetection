import os

import cv2

from scripts.helpers.data_manipulation_helpers import get_keypoint_angles, write_data, split_data
from scripts.helpers.openpose_accuraccy_testing import get_accuracy, optimal_net_res
from scripts.openposeKeypoints import get_openpose_keypoints
from scripts.preprocess_image_folders import split_file, get_file_class_num, resize_images_to_scale, sort_images_by_size

input_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/bridge'
result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/anglesNew.csv'

# split_file(input_file_path, 5)
class_number = get_file_class_num(input_file_path)
net_res_width = 512
net_res_height = 256



# optimal_net_res(input_file_path, 8, 128, 64)




