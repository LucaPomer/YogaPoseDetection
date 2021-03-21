from scripts.helpers.create_classification_data import create_classification_data


### TRAIN DATA
from scripts.preprocess_image_folders import add_flipped_images

train_images_folder_pt1 = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_images/part1'


train_images_folder_pt2 = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_images/part2'
train_angles_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_angles_with_flipped.csv'
train_dist_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_dist_with_flipped.csv'
train_both_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_both_with_flipped.csv'
train_complete_batches_csv = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/batchesComplete_train.csv'


## TEST DATA
test_images_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_images'
test_angles_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/test_data_angles.csv'
test_dist_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/test_data_dist.csv'
test_both_result_file_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/test_data_both.csv'
test_complete_batches_csv = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/batchesComplete_test.csv'


net_res_width = 512
net_res_height = 256
batch_size = 5

## Add FLipped images
add_flipped_images(train_images_folder_pt2)

##Create Data
# create_classification_data(train_images_folder_pt1,train_angles_result_file_path,
#                            train_dist_result_file_path, train_both_result_file_path,
#                            net_res_width, net_res_height, batch_size,
#                            train_complete_batches_csv )

# create_classification_data(train_images_folder_pt2, train_angles_result_file_path,
#                            train_dist_result_file_path, train_both_result_file_path,
#                            net_res_width, net_res_height, batch_size,
#                            train_complete_batches_csv)
#
create_classification_data(test_images_folder,test_angles_result_file_path,
                           test_dist_result_file_path, test_both_result_file_path,
                           net_res_width, net_res_height, batch_size,
                           test_complete_batches_csv )

