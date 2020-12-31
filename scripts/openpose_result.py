import numpy as np


class OpenposeResult:
    def __init__(self, keypoints, img_path, output_img):
        self.keypoints = keypoints
        self.img_path = img_path
        self.output_img = output_img

    def number_found_keypoints(self):
        # print(keypoints)
        # print("amount of non zero keypoints" + str(np.count_nonzero(self.keypoints, axis=0)))
        return np.count_nonzero(self.keypoints, axis=0)[0]
