# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
from scripts.helpers.data_manipulation_helpers import get_keypoint_angles
from scripts.helpers.data_manipulation_helpers import write_data
from scripts.helpers.openpose_helpers import define_parser, define_params

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

    parser = define_parser("/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angleCalcTest")
    args = parser.parse_known_args()
    params = define_params(args)


    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    image_paths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    processed_images = 0
    # print(imagePaths)
    # Process and display images
    for image_path in image_paths:
        print(image_path)
        datum = op.Datum()
        image_to_process = cv2.imread(image_path)
        # print(imageToProcess is None)
        datum.cvInputData = image_to_process
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print(datum.poseKeypoints[0] is None or image_to_process is None or datum is None)

        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print("NumOfHumansInPicture" + str(len(datum.poseKeypoints)))
        processed_images = processed_images + 1
        # print(proccesedImages)

        keypoint_angles = get_keypoint_angles(datum.poseKeypoints[0])
        print(keypoint_angles)
        write_data(keypoint_angles)

        #cv2.imwrite('prossesedImg' + str(proccesedImages) + '.jpg', datum.cvOutputData)

        # if not args[0].no_display:
        #     cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        #     key = cv2.waitKey(15)
        #     if key == 27: break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
