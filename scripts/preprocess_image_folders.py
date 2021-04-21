import re
import shutil
import cv2
import os

from scripts.helpers.pose_to_class_dict import pose_to_class_num


def get_file_class_num(folder_path):
    for key in pose_to_class_num.keys():
        if re.search(key, folder_path, re.IGNORECASE):
            return pose_to_class_num[key]
    return -1


def split_and_sort_folder(folder_path, img_per_folder):
    batch_num = 1
    sorted_by_size = sort_images_by_size(folder_path)
    dst_path = folder_path + '/batch' + str(batch_num)
    os.makedirs(dst_path)
    num_images_in_batch = 0
    for img_path in sorted_by_size:
        if num_images_in_batch >= img_per_folder:
            batch_num += 1
            dst_path = folder_path + '/batch' + str(batch_num)
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


def add_flipped_images(classes_folder):
    for class_folder in os.listdir(classes_folder):
        full_class_path = classes_folder + '/' + class_folder
        for image_path in os.listdir(full_class_path):
            full_img_path = full_class_path + '/' + image_path
            flipped_img_path = full_class_path + '/' + 'flip' + image_path
            try:
                img = cv2.imread(full_img_path)
                img_flip_lr = cv2.flip(img, 1)
                cv2.imwrite(flipped_img_path, img_flip_lr)
            except:
                print(flipped_img_path)
