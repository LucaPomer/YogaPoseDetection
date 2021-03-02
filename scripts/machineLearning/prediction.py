from scripts.helpers.data_creation_helpers import get_angles
from scripts.helpers.sklearn_helpers import load_model_and_predict
from scripts.Openpose.openpose_algorithm import run_openpose_algorithm

net_res_width = 512
net_res_height = 256
images_to_classify = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/unlabled_images/set2'

result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, images_to_classify)
angles = get_angles(result_from_openpose)
classResult = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_angles.sav', angles)

print(classResult)
