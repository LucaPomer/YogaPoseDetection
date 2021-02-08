import os

from scripts.helpers.data_manipulation_helpers import write_data
from scripts.openpose_algorithm import run_openpose_algorithm


def get_num_files(folder_path):
    listOfFiles = os.listdir(folder_path)  # dir is your directory path
    number_files = len(listOfFiles)
    return number_files


def optimal_net_res(file_path, num_tries, min_height, step_size):
    net_res_height = min_height
    net_res_width = min_height
    best_accuracy = 0
    optimal_res_height = net_res_height
    optimal_res_width = net_res_width
    num_files = get_num_files(file_path)
    for i in range(num_tries):
        for j in range(num_tries):
            print(net_res_height)
            result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, file_path)
            accuracy = get_accuracy(result_from_openpose, num_files, True)
            accuracy_found_skeletons = get_accuracy(result_from_openpose, num_files, False)
            print(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_res_height = net_res_height
                optimal_res_width = net_res_width
            net_res_width += step_size
            write_data([net_res_width, net_res_height, accuracy, accuracy_found_skeletons], '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/accuracy.csv')
        net_res_height += step_size
        net_res_width = min_height
    print("optimal net res width " + str(optimal_res_width) + " optimal res height " + str(optimal_res_height) + " best accuracy: " + str(best_accuracy))
    return [optimal_res_width, optimal_res_height]


def get_accuracy(result_from_openpose, num_files_in_folder, respective_to_all_files):
    total_found_keypoints = 0
    print("num images " + str(len(result_from_openpose)))
    for result_obj in result_from_openpose:
        num_non_zero_points = result_obj.number_found_keypoints()
        total_found_keypoints += num_non_zero_points
        print('found keypoints ' + str(num_non_zero_points))

    if respective_to_all_files:
        total_possible_points = 25 * num_files_in_folder
    else:
        total_possible_points = 25 * len(result_from_openpose)

    if total_possible_points == 0:
        return 0
    return (total_found_keypoints * 100)/total_possible_points
