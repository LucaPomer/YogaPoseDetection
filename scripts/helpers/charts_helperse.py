import numpy as np
import matplotlib.pyplot as plt


def accuracy_bar_chart(accuracy_array, labels, classifier_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, accuracy_array, width, label=None)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Precision')
    ax.set_title('Accuracy Per Class - ' + classifier_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()
