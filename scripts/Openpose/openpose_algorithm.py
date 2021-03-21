# From Python
# It requires OpenCV installed for Python
import sys
import os
from sys import platform
import time

from scripts.helpers.data_manipulation_helpers import write_data
from scripts.helpers.openpose_helpers import define_parser, define_params, get_keypoints_first_human
from scripts.Openpose.openpose_result import OpenposeResult


# @article{8765346,
#   author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
#   journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
#   year = {2019}
# }

def run_openpose_algorithm(net_width, net_height, img_dir):
    try:
        # Import Openpose (Windows/Ubuntu/OSX) -- give the path to the openpose build
        dir_path = os.path.dirname(os.path.realpath('/Users/lucapomer/openpose_build_new/openpose/build'))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH'] = os.environ[
                                         'PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
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
        print(args[0].image_dir)
        for image_path in image_paths:
            datum = op.Datum()
            key_points = get_keypoints_first_human(op, image_path, opWrapper, datum)
            if key_points is not None:
                result = OpenposeResult(key_points, image_path, datum.cvOutputData)
                return_list.append(result)
            else:
                write_data([image_path], '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/failed_images_openpose.csv')

        end = time.time()
        print("OpenPose COMPLETE Total time: " + str(end - start) + " seconds " + str(args[0].image_dir)
              )
        return return_list
    except Exception as e:
        print(e)
        sys.exit(-1)
