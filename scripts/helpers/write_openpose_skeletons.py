## TEST DATA
import cv2
from scripts.Openpose.openpose_algorithm import run_openpose_algorithm

net_res_width = 512
net_res_height = 256

result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height,
                                              '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/Pose_Skeletons')

i = 0
for item in result_from_openpose:
    cv2.imwrite('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/result_skeletons/' + str(i) + '.jpg',
                item.output_img)
    i += 1
