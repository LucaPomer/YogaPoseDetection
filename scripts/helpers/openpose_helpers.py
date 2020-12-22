import argparse


def define_parser(images_directory):
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=images_directory,
                        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    return parser


def define_params(args):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    # params["write_json"] = " angleCalcTest/"
    params["num_gpu_start"] = 1
    # params["net_resolution"] = "256x256"
    params["model_folder"] = "/Users/lucapomer/openpose_build_new/openpose/models"

    # Add others in path?  #todo: figure out what this is needed for
    # print("arguments " + str(args)) - args[1] seems empty
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
