import pickle
import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.charts_helperse import get_per_data_accuacy, get_nn_per_data_accuracy

models_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/models'
nn_model_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks'

gauss_model_dist = pickle.load(open(models_folder + '/Gauss_optimal_dist.sav', 'rb'))
gauss_model_angles = pickle.load(open( models_folder +'/Gauss_optimal_angles.sav', 'rb'))
gauss_model_both = pickle.load(open(models_folder +'/Gauss_optimal_both.sav', 'rb'))

scv_model_angles = pickle.load(open(models_folder +'/SVC_optimal_angles.sav', 'rb'))
scv_model_dist = pickle.load(open(models_folder +'/SVC_optimal_dist.sav', 'rb'))
scv_model_both = pickle.load(open(models_folder +'/SVC_optimal_both.sav', 'rb'))


tree_model_angles = pickle.load(open(models_folder +'/Tree_optimal_angles.sav', 'rb'))
tree_model_dist = pickle.load(open(models_folder +'/Tree_optimal_dist.sav', 'rb'))
tree_model_both = pickle.load(open(models_folder +'/Tree_optimal_both.sav', 'rb'))

mlp_model_angles = pickle.load(open(models_folder +'/Mlp_optimal_angles.sav', 'rb'))
mlp_model_dist = pickle.load(open(models_folder +'/Mlp_optimal_dist.sav', 'rb'))
mlp_model_both = pickle.load(open(models_folder +'/Mlp_optimal_both.sav', 'rb'))

nn_angles_model = nn_model_folder + '/correct_split_angles.h5'
nn_dist_model = nn_model_folder + '/correct_split_dist.h5'
nn_both_model = nn_model_folder + '/correct_split_both.h5'

scv_accuracys = get_per_data_accuacy(scv_model_angles, scv_model_dist, scv_model_both)
gauss_accuracies = get_per_data_accuacy(gauss_model_angles, gauss_model_dist, gauss_model_both)
tree_accuracies = get_per_data_accuacy(tree_model_angles,tree_model_dist, tree_model_both)
mlp_accuracies = get_per_data_accuacy(mlp_model_angles,mlp_model_dist, mlp_model_both)
nn_accuracies = get_nn_per_data_accuracy(angle_model_path=nn_angles_model, dist_model_path=nn_dist_model, both_model_path=nn_both_model)


rounded_svm = np.around(scv_accuracys, decimals=2)
rounded_gauss = np.around(gauss_accuracies, decimals=2)
rounded_tree = np.around(tree_accuracies, decimals=2)
rounded_mlp = np.around(mlp_accuracies, decimals=2)
rounded_nn = np.around(nn_accuracies, decimals=2)


labels = ['Angles', 'Distances', 'Both']


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
svm_rects = ax.bar(x - (width *2), rounded_svm, width, label='SVM')
gauss_rects = ax.bar(x - width, rounded_gauss, width, label='Gauss')
tree_rects = ax.bar(x , rounded_tree, width, label='Tree')
mlp_rects = ax.bar(x + width, rounded_mlp, width, label='MLP')
nn_rects = ax.bar(x + (width*2), rounded_mlp, width, label='Neural_Network')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Model and Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(svm_rects)
autolabel(gauss_rects)
autolabel(tree_rects)
autolabel(mlp_rects)
autolabel(nn_rects)

plt.tight_layout()

plt.show()