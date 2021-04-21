import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.charts_helperse import get_per_data_accuacy, get_nn_per_data_accuracy
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTesting

images_folder = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/combined_images'

labels = [0 for x in range(10)]
images_count = [0 for x in range(10)]

for class_folder in os.listdir(images_folder):
    full_class_folder = images_folder + "/" + class_folder
    list = os.listdir(full_class_folder)  # dir is your directory path
    number_files = len(list)
    name_split = class_folder.split('-')
    # print(name_split[1], '->', name_split[0])
    labels[int(name_split[0])] = name_split[1]
    images_count[int(name_split[0])] = number_files


x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
count_rects = ax.bar(x, images_count, width, label='images')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Images')
ax.set_title('Number of images per Class')
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
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(count_rects)

plt.tight_layout()

plt.show()
