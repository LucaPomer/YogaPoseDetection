import os
from scripts.helpers.data_creation_helpers import distance_calc_and_write_data, \
    angle_and_dist_calc_and_write_data, angle_calc_and_write_data
from scripts.helpers.data_manipulation_helpers import write_data
from scripts.Openpose.openpose_algorithm import run_openpose_algorithm
from scripts.preprocess_image_folders import split_and_sort_file, get_file_class_num


def create_classification_data(images_folder, angles_result_file_path, dist_result_file_path,
                               both_result_file_path, net_res_width, net_res_height, batch_size,
                               complete_batches_csv):
    for class_folder in os.listdir(images_folder):
        full_class_path = images_folder + '/' + class_folder
        try:
            split_and_sort_file(full_class_path, batch_size)
        except:
            print("no split needed")

        class_number = get_file_class_num(full_class_path)
        current_folder = class_folder
        try:
            for batch_folder in os.listdir(full_class_path):
                current_folder = batch_folder
                full_folder_path = full_class_path + '/' + batch_folder
                result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, full_folder_path)

                angle_calc_and_write_data(result_from_openpose, angles_result_file_path, class_number)
                distance_calc_and_write_data(result_from_openpose, dist_result_file_path, class_number)
                angle_and_dist_calc_and_write_data(result_from_openpose, both_result_file_path, class_number)
                write_data([full_folder_path],
                           complete_batches_csv)
        except:
            print("problem with folder: " + current_folder)
