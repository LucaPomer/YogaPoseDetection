

def get_accuracy(result_from_openpose):
    total_found_keypoints = 0
    for result_obj in result_from_openpose:
        num_non_zero_points = result_obj.number_found_keypoints()
        total_found_keypoints += num_non_zero_points
        print('found keypoints ' + str(num_non_zero_points))

    total_possible_points = 25 * len(result_from_openpose)
    return (total_found_keypoints * 100)/total_possible_points
