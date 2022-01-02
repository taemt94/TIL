from utils import get_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pathlib
from PIL import Image

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION
    # print(ground_truth[0])
    # print(len(ground_truth))
    # plt.figure(figsize=(10, 2))
    plt.axis('off')
    image_base_path = pathlib.Path().cwd() / 'data' / 'images'
    for i, gt in enumerate(ground_truth):
        ax = plt.subplot(2, 1, i + 1)
        ax.axis('off')
        filename = image_base_path / gt['filename']
        image = Image.open(filename)
        print(gt.keys())
        # image.show()
        image = np.array(image)
        bboxes = gt['boxes']
        classes = gt['classes']
        for bbox, cl in zip(bboxes, classes):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), width=x2-x1, height=y2-y1,
                                     edgecolor='g', fill=False)
            ax.add_patch(rect)
            ax.text((x1+x2)/2, y2+10, f"Class[{cl}]")
        
        ax.imshow(image)
        if i == 1:
            break
        
    plt.show()
        


if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)