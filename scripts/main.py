
from scripts.helpers.useful_functions import run_openpose_and_distance_calc
from scripts.preprocess_image_folders import split_file, get_file_class_num, resize_images_to_scale, sort_images_by_size

input_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/tree'

result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/distances.csv'

# split_file(input_file_path, 5)
class_number = get_file_class_num(input_file_path)
net_res_width = 512
net_res_height = 256

run_openpose_and_distance_calc(input_file_path, result_file_path,
                               net_res_width, net_res_height, class_number)

# optimal_net_res(input_file_path, 8, 128, 64)




