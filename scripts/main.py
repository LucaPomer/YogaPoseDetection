import cv2

from scripts.helpers.data_manipulation_helpers import get_keypoint_angles, write_data
from scripts.helpers.openpose_accuraccy_testing import get_accuracy
from scripts.openposeKeypoints import get_openpose_keypoints


class_number = 3
input_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/testTree'
result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv'

net_res_width = -1
net_res_height = 368


result_from_openpose = get_openpose_keypoints(net_res_width, net_res_height, input_file_path)

accuracy = get_accuracy(result_from_openpose)
print(accuracy)

for item in result_from_openpose:
    angles = get_keypoint_angles(item.keypoints)
    angles.append(class_number)
    write_data(angles, result_file_path)

# result = get_openpose_keypoints(-1, 368, input_folder)
# processd_images =0
# for result_obj in result:
#     print(result_obj.img_path)
#     cv2.imwrite('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/sceletons/processedImg' + str(processd_images) + '.jpg', result_obj.output_img)
#     processd_images += 1
#

