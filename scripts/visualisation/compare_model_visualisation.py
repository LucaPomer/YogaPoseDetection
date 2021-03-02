import pickle
import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.charts_helperse import get_per_data_accuacy

gauss_model_dist = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_dist.sav', 'rb'))
gauss_model_angles = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_angles.sav', 'rb'))
gauss_model_both = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_both.sav', 'rb'))

scv_model_angles = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles.sav', 'rb'))
scv_model_dist = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_dist.sav', 'rb'))
scv_model_both = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_both.sav', 'rb'))


tree_model_angles = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_angles.sav', 'rb'))
tree_model_dist = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_dist.sav', 'rb'))
tree_model_both = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_both.sav', 'rb'))

mlp_model_angles = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_angles.sav', 'rb'))
mlp_model_dist = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_dist.sav', 'rb'))
mlp_model_both = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_both.sav', 'rb'))

scv_accuracys = get_per_data_accuacy(scv_model_angles, scv_model_dist, scv_model_both)
gauss_accuracies = get_per_data_accuacy(gauss_model_angles, gauss_model_dist, gauss_model_both)
tree_accuracies = get_per_data_accuacy(tree_model_angles,tree_model_dist, tree_model_both)
mlp_accuracies = get_per_data_accuacy(mlp_model_angles,mlp_model_dist, mlp_model_both)


rounded_svm = np.around(scv_accuracys, decimals=2)
rounded_gauss = np.around(gauss_accuracies, decimals=2)
rounded_tree = np.around(tree_accuracies, decimals=2)
rounded_mlp = np.around(mlp_accuracies, decimals=2)


labels = ['Angles', 'Distances', 'Both']


x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
svm_rects = ax.bar(x - width, rounded_svm, width, label='SVM')
gauss_rects = ax.bar(x, rounded_gauss, width, label='Gauss')
tree_rects = ax.bar(x + width, rounded_tree, width, label='Tree')

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

plt.tight_layout()

plt.show()