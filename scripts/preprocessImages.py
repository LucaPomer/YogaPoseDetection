import cv2
import os
import shutil


# resizing images
# for filename in os.listdir('/experiments/accuraccyTest'): img = cv2.imread(os.path.join(
# '/experiments/accuraccyTest', filename)) print('Original Dimensions : ', img.shape) cv2.imwrite(
# '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/croppedAndResized/' + str(filename) + '.jpg', cv2.resize(
# img, (256, 256), interpolation=cv2.INTER_AREA))
#

def split_file(file_path, img_per_folder):
    batch_num = 1
    dst_path = file_path + '/batch' + str(batch_num)
    os.makedirs(dst_path)
    num_images_in_batch = 0
    for img_path in os.listdir(file_path):
        print(img_path)
        full_img_path = file_path + '/' + img_path
        if num_images_in_batch >= img_per_folder:
            batch_num += 1
            dst_path = file_path + '/batch' + str(batch_num)
            os.makedirs(dst_path)
            num_images_in_batch = 0
        shutil.move(full_img_path, dst_path)
        num_images_in_batch += 1


# scale_percent = 60  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#
# print('Resized Dimensions : ', resized.shape)
#
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
cv2.destroyAllWindows()
