import os
import shutil

def combine_and_numerate(folder1, folder2, output_folder, class_name):
    file_num =1
    full_output_folder = output_folder+"/"+class_name
    try:
        os.mkdir(full_output_folder)
    except OSError:
        print("Creation of the directory already exists")

    for filename in os.listdir(folder1):
        old_file_name = folder1 + "/" + filename
        new_file_name = full_output_folder+'/file'+str(file_num)+".jpg"
        shutil.move(old_file_name, new_file_name)
        file_num +=1
    for filename in os.listdir(folder2):
        old_file_name = folder2 + "/" + filename
        new_file_name = full_output_folder + '/file' + str(file_num) + ".jpg"
        shutil.move(old_file_name, new_file_name)
        file_num += 1


combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/bridge',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/bridge',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images/bridge',
                     class_name='bridge')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/childs',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/childs',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='childs')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/downwarddog',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/downwarddog',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='downwarddog')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/mountain',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/mountain',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='mountain')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/plank',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/plank',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='plank')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/seatedforwardbend',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/seatedforwardbend',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='seatedforwardbend')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/tree',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/tree',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='tree')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/trianglepose',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/trianglepose',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='trianglepose')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/warrior1',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/warrior1',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='warrior1')

combine_and_numerate('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/test_set/warrior2',
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/training_set/warrior2',
                     output_folder='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images',
                     class_name='warrior2')



