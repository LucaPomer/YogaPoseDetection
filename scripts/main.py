import os

from scripts.helpers.data_creation_helpers import distance_calc_and_write_data, \
    distance_and_degree_calc_and_write_data, angle_calc_and_write_data
from scripts.openpose_algorithm import run_openpose_algorithm
from scripts.preprocess_image_folders import split_file, get_file_class_num, resize_images_to_scale, sort_images_by_size

images_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/images'

result_file_path = '/angles.csv'

##  split_file(input_file_path, 5)
class_number = get_file_class_num(images_folder)
net_res_width = 512
net_res_height = 256

# for class_folder in os.listdir(images_folder):
#     try:
#         split_file(class_folder, 5)
#     except:
#         print("no split needed")

for class_folder in os.listdir(images_folder):
    full_class_path = images_folder + '/' + class_folder
    try:
        split_file(full_class_path, 5)
    except:
        print("no split needed")

    class_number = get_file_class_num(full_class_path)

    for batch_folder in os.listdir(full_class_path):
        full_folder_path = full_class_path + '/' + batch_folder
        result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, full_folder_path)
        angle_calc_and_write_data(result_from_openpose, result_file_path, class_number)

# optimal_net_res(input_file_path, 8, 128, 64)
