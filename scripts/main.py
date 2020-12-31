import cv2

from scripts.helpers.openpose_accuraccy_testing import get_accuracy
from scripts.openposeKeypoints import get_openpose_keypoints

input_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/bridgeSmallAmount'
accuracy = get_accuracy(input_folder, -1, 368)
print(accuracy)
# result = get_openpose_keypoints(-1, 368, input_folder)
# processd_images =0
# for result_obj in result:
#     print(result_obj.img_path)
#     cv2.imwrite('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/sceletons/processedImg' + str(processd_images) + '.jpg', result_obj.output_img)
#     processd_images += 1
#

