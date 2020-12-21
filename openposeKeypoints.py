# From Python
# It requires OpenCV installed for Python
import csv
import sys
import cv2
import os
from sys import platform
import argparse
import time
from numpy import asarray, genfromtxt
import numpy as np
from numpy import savetxt
from helpFunctions import getKeypointAngles

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
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angleCalcTest",
                        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    # params["write_json"] = " angleCalcTest/"
    params["num_gpu_start"] = 1
    # params["net_resolution"] = "256x256"
    params["model_folder"] = "/Users/lucapomer/openpose_build_new/openpose/models"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    keypoints = []
    proccesedImages = 0
    # print(imagePaths)
    # Process and display images
    for imagePath in imagePaths:
        print(imagePath)
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        # print(imageToProcess is None)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print(datum.poseKeypoints[0] is None or imageToProcess is None or datum is None)

        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print("NumOfHumansInPicture" + str(len(datum.poseKeypoints)))
        keypoints.append(datum.poseKeypoints[0])
        proccesedImages = proccesedImages + 1
        # print(proccesedImages)

        getKeypointAngles(datum.poseKeypoints[0])

        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(33)  # press a to continue
        #cv2.imwrite('prossesedImg' + str(proccesedImages) + '.jpg', datum.cvOutputData)

        # if not args[0].no_display:
        #     cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        #     key = cv2.waitKey(15)
        #     if key == 27: break

    reformatedKeys = []
    # for keypoint in keypoints:
    #     personEntry = []
    #     for entry in keypoint:
    #         #print(entry)
    #         personEntry.append(entry[0] + entry[1])
    #     personEntry.append(2)
    #     reformatedKeys.append(personEntry)
    #     with open('dataFormatted.csv', 'a') as fd:
    #         writer = csv.writer(fd)
    #         writer.writerow(personEntry)

    # data = asarray(reformatedKeys)
    # savetxt('data.csv', data, delimiter=',')
    currentData = genfromtxt('dataFormatted.csv', delimiter=',')
    # currentData.append(reformatedKeys)
    # savetxt('dataFormatted.csv', currentData, delimiter=',')

    # print(str(keypoints))
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
