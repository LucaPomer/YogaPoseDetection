# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
from scripts.helpers.data_manipulation_helpers import get_keypoint_angles, number_found_keypoints
from scripts.helpers.data_manipulation_helpers import write_data
from scripts.helpers.openpose_helpers import define_parser, define_params, get_keypoints_all_humans


def get_openpose_keypoints(net_width, net_height, img_dir, output_dir):
    try:
        # Import Openpose (Windows/Ubuntu/OSX) -- give the path to the openpose build
        dir_path = os.path.dirname(os.path.realpath('/Users/lucapomer/openpose_build_new/openpose/build'))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.) sys.path.append(
                # '../../python'); If you run `make install` (default path is `/usr/local/python` for Ubuntu),
                # you can also access the OpenPose/python module from there. This will install OpenPose and the python
                # library at your desired installation path. Ensure that this is in your python path in order to use it.
                sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
                'script in the right folder?')
            raise e

        parser = define_parser(img_dir)
        args = parser.parse_known_args()
        params = define_params(args, net_width, net_height)

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Read frames on directory
        image_paths = op.get_images_on_directory(args[0].image_dir)
        start = time.time()

        processed_images = 0
        class_number = 2
        # Process and display images
        for image_path in image_paths:
            datum = op.Datum()
            keypoints_all_humans = get_keypoints_all_humans(op, image_path, opWrapper, datum)
            first_human_found = keypoints_all_humans[0]
            num_found_keyppoints = number_found_keypoints(first_human_found)
            cv2.imwrite(output_dir+'/prossesedImg' + str(net_width) + 'points'+ str(num_found_keyppoints) + '.jpg', datum.cvOutputData)
            #for keypoints_human in keypoints_all_humans:

            # print(proccesedImages)

            # keypoint_angles = get_keypoint_angles(first_human_found)
            # keypoint_angles.append(class_number)
            # print(keypoint_angles)
            write_data(first_human_found)
            processed_images = processed_images + 1


            # cv2.imwrite('prossesedImg' + str(proccesedImages) + '.jpg', datum.cvOutputData)

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        print(e)
        sys.exit(-1)
