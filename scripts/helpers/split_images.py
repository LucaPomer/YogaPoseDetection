import os
import shutil


def split_images(classes_folder, split_ration, train_folder, test_folder):
    for class_folder in os.listdir(classes_folder):
        full_class_folder = classes_folder + "/" + class_folder
        list = os.listdir(full_class_folder)  # dir is your directory path
        number_files = len(list)
        num_test = int(number_files * split_ration)
        num_int_test =0
        print (number_files)
        print(number_files * split_ration)
        os.mkdir(test_folder + "/" + class_folder)
        os.mkdir(train_folder + "/" + class_folder)

        for image in os.listdir(full_class_folder):
            full_image_file_name = full_class_folder +"/"+image
            if num_test > num_int_test:
                new_image_folder = test_folder+"/" + class_folder + "/" +image
            else:
                new_image_folder = train_folder+"/" + class_folder + "/" +image

            shutil.move(full_image_file_name, new_image_folder)
            num_int_test +=1




split_images('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images', 0.25,
             train_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/train_images',
             test_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_images')