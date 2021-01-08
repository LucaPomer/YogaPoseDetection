# From Python
# It requires OpenCV installed for Python
import sys
import os
from sys import platform
import time
from scripts.helpers.openpose_helpers import define_parser, define_params, get_keypoints_first_human
from scripts.openpose_result import OpenposeResult


def get_openpose_keypoints(net_width, net_height, img_dir):
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

        return_list = []
        for image_path in image_paths:
            datum = op.Datum()
            keypoints = get_keypoints_first_human(op, image_path, opWrapper, datum)
            if keypoints is not None:
                result = OpenposeResult(keypoints, image_path, datum.cvOutputData)
                return_list.append(result)

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
        return return_list
    except Exception as e:
        print(e)
        sys.exit(-1)
