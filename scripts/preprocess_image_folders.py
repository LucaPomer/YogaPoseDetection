import re

import cv2
import os
import shutil


# resizing images
# for filename in os.listdir('/experiments/accuraccyTest'): img = cv2.imread(os.path.join(
# '/experiments/accuraccyTest', filename)) print('Original Dimensions : ', img.shape) cv2.imwrite(
# '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/croppedAndResized/' + str(filename) + '.jpg', cv2.resize(
# img, (256, 256), interpolation=cv2.INTER_AREA))
#
from scripts.helpers.dictionaries import pose_to_class_num


def get_file_class_num(folder_path):
    for key in pose_to_class_num.keys():
        if re.search(key, folder_path, re.IGNORECASE):
            return pose_to_class_num[key]
    return -1


def split_file(file_path, img_per_folder):
    batch_num = 1
    dst_path = file_path + '/batch' + str(batch_num)
    os.makedirs(dst_path)
    num_images_in_batch = 0
    for img_path in os.listdir(file_path):
        full_img_path = file_path + '/' + img_path
        if num_images_in_batch >= img_per_folder:
            batch_num += 1
            dst_path = file_path + '/batch' + str(batch_num)
            os.makedirs(dst_path)
            num_images_in_batch = 0
        shutil.move(full_img_path, dst_path)
        num_images_in_batch += 1



