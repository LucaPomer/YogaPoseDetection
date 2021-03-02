import argparse

import cv2


def define_parser(images_directory):
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_people_max', type=int, default=1)
    parser.add_argument("--image_dir", default=images_directory,
                        help="Process a directory of training_images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    return parser


def define_params(args, net_width, net_height):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    # params["write_json"] = " accuraccyTest/"
    params["num_gpu_start"] = 1
    params["number_people_max"] = 1  # get only the one top scored person
    params["net_resolution"] = str(net_width) + "x" + str(net_height)
    params["scale_number"] = 4
    params["keypoint_scale"] = 3  # scale the keypoints in the range [0,1]
    params["scale_gap"] = 0.25
    params["model_folder"] = "/Users/lucapomer/openpose_build_new/openpose/models"

    # Add others in path?  #todo: figure out what this is needed for
    # print("arguments " + str(args))  # args[1] seems empty
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
    return params


def get_keypoints_first_human(op_instance, image_path, op_wrapper, datum):
    image_to_process = cv2.imread(image_path)
    datum.cvInputData = image_to_process
    op_wrapper.emplaceAndPop(op_instance.VectorDatum([datum]))
    if datum.poseKeypoints is not None:
        # print("NumOfHumansInPicture " + str(len(datum.poseKeypoints)))
        # cv2.imshow('image', datum.cvOutputData)
        # cv2.waitKey(33)
        return datum.poseKeypoints[0]  # as I defined max people 0 the array has always 1 human
    else:
        return None
