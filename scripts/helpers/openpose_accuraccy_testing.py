from scripts.openposeKeypoints import get_openpose_keypoints


def get_accuracy(input_file_path, net_res_width, net_res_height):
    result_from_openpose = get_openpose_keypoints(net_res_width, net_res_height, input_file_path)
    total_found_keypoints = 0
    for result_obj in result_from_openpose:
        num_non_zero_points = result_obj.number_found_keypoints()
        total_found_keypoints += num_non_zero_points
        print('found keypoints ' + str(num_non_zero_points))

    total_possible_points = 25 * len(result_from_openpose)
    return (total_found_keypoints * 100)/total_possible_points
