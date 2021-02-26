import os

from scripts.helpers.data_creation_helpers import distance_calc_and_write_data, \
    angle_and_dist_calc_and_write_data, angle_calc_and_write_data
from scripts.helpers.data_manipulation_helpers import write_data
from scripts.Openpose.openpose_algorithm import run_openpose_algorithm
from scripts.preprocess_image_folders import split_file, get_file_class_num

images_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/images/part2'

angles_result_file_path = '/csv_data_files/angles_with_flipped.csv'
dist_result_file_path = '/csv_data_files/dist_with_flipped.csv'
both_result_file_path = '/csv_data_files/both_with_flipped.csv'

##  split_file(input_file_path, 5)
class_number = get_file_class_num(images_folder)
net_res_width = 512
net_res_height = 256
batch_size = 5

# for class_folder in os.listdir(images_folder):
#     try:
#         split_file(class_folder, 5)
#     except:
#         print("no split needed")

# add_flipped_images(images_folder)


for class_folder in os.listdir(images_folder):
    full_class_path = images_folder + '/' + class_folder
    try:
        split_file(full_class_path, batch_size)
    except:
        print("no split needed")

    class_number = get_file_class_num(full_class_path)

    try:
        for batch_folder in os.listdir(full_class_path):
            full_folder_path = full_class_path + '/' + batch_folder
            result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, full_folder_path)

            angle_calc_and_write_data(result_from_openpose, angles_result_file_path, class_number)
            distance_calc_and_write_data(result_from_openpose, dist_result_file_path, class_number)
            angle_and_dist_calc_and_write_data(result_from_openpose, both_result_file_path, class_number)
            write_data([full_folder_path], '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/batchesComplete.csv')
    except:
        print("problem with folder: " + batch_folder)
# optimal_net_res(input_file_path, 8, 128, 64)
