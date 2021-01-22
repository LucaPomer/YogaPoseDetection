import re

import os
import shutil
import cv2
import os

from scripts.helpers.dictionaries import pose_to_class_num


def resize_images_to_scale(width, height, file_path, dst_file_path):
    for img_path in os.listdir(file_path):
        full_img_path = file_path + '/' + img_path
        img_elem = cv2.imread(full_img_path)
        cv2.imwrite(dst_file_path + '/' + img_path, cv2.resize(img_elem, (width, height), interpolation=cv2.INTER_AREA))


def get_file_class_num(folder_path):
    for key in pose_to_class_num.keys():
        if re.search(key, folder_path, re.IGNORECASE):
            return pose_to_class_num[key]
    return -1


def split_file(file_path, img_per_folder):
    batch_num = 1
    sorted_by_size = sort_images_by_size(file_path)
    dst_path = file_path + '/batch' + str(batch_num)
    os.makedirs(dst_path)
    num_images_in_batch = 0
    for img_path in sorted_by_size:
        if num_images_in_batch >= img_per_folder:
            batch_num += 1
            dst_path = file_path + '/batch' + str(batch_num)
            os.makedirs(dst_path)
            num_images_in_batch = 0
        shutil.move(img_path, dst_path)
        num_images_in_batch += 1


# Note: the array is in the right order but when saved the file system will use alphabetic order
def sort_images_by_size(file_path):
    files = os.listdir(file_path)
    full_path_files = []
    for file in files:
        full_path_files.append(file_path + '/' + file)
    full_path_files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
    return full_path_files
