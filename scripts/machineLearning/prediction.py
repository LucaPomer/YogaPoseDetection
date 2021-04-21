from keras import models

from scripts.helpers.data_creation_helpers import get_angles, get_distances_and_angles
from scripts.helpers.sklearn_helpers import load_model_and_predict
from scripts.Openpose.openpose_algorithm import run_openpose_algorithm

net_res_width = 512
net_res_height = 256
images_to_classify = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/unlabled_images/set1'

result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, images_to_classify)
angles_and_dist = get_distances_and_angles(result_from_openpose)
Gauss_Result = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/Gauss_optimal_both.sav', angles_and_dist)
SVM_Result = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/SVC_optimal_both.sav', angles_and_dist)
Tree_Result = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/Tree_optimal_both.sav', angles_and_dist)
MLP_Result = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/MlP_optimal_both.sav', angles_and_dist)
model_nn = models.load_model('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks/correct_split_both.h5')
y_prob = result = model_nn.predict(angles_and_dist)
NN_Result = y_prob.argmax(axis=-1)

print(Gauss_Result, '->', "gauss")
print(SVM_Result, '->', "SVM")
print(Tree_Result, '->', "tree")
print(MLP_Result, '->', "mlp")
print(NN_Result, '->', "nn")
