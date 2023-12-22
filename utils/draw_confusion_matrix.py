import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", save_path=None, dpi=300):
    """
    Parameters:
        label_true: True Labels, e.g.[0,1,2,7,4,5,...]
        label_pred: Pred Labels, e.g.[0,5,4,2,1,4,...]
        label_name: Labels' Name, e.g.['cat','dog','flower',...]
        title: 
        save_path: 
        dpi: 
    Return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    return save_path
