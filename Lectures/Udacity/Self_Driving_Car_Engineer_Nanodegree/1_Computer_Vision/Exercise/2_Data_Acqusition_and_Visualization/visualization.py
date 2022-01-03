from utils import get_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pathlib
from PIL import Image

plt.style.use('default')
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 12

def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION
    image_base_path = pathlib.Path().cwd() / 'data' / 'images'

    total_classes = []
    first = 1
    for gt in ground_truth:
        if first:
            total_classes = gt['classes']
            first = 0
        else:
            total_classes += gt['classes']
    total_classes = np.unique(np.array(total_classes))
    bbox_colormap = {}
    for cl in total_classes:
        if cl not in bbox_colormap.keys():
            if cl == 1:
                bbox_colormap[cl] = 'g'
            elif cl == 2:
                bbox_colormap[cl] = 'r'
    for i, gt in enumerate(ground_truth):
        ax = plt.subplot(4, 5, i + 1)
        ax.axis('off')
        filename = image_base_path / gt['filename']
        image = Image.open(filename)
        image = np.array(image)
        bboxes = gt['boxes']
        classes = gt['classes']
        for bbox, cl in zip(bboxes, classes):
            y1, x1, y2, x2 = bbox
            rect = patches.Rectangle((x1, y1), width=x2-x1, height=y2-y1,
                                     edgecolor=bbox_colormap[cl], fill=False)
            ax.add_patch(rect)
        ax.imshow(image)
    plt.tight_layout()        
    plt.show()
        


if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)