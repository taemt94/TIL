import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # IMPLEMENT THIS FUNCTION
    img = np.array(Image.open(path))
    print(img.shape)
    img = np.array(Image.open(path).convert('RGB'))
    color_threshold = np.array(color_threshold).reshape(1, 1, 3)
    mask = np.where(img > color_threshold, True, False)
    mask = mask[..., 0] & mask[..., 1] & mask[..., 2]
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # IMPLEMENT THIS FUNCTION
    plt.figure(figsize=(10, 6))
    plt.subplot(131)
    plt.imshow(img)
    
    plt.subplot(132)
    plt.imshow(mask)
    
    masked_img = img * np.stack([mask]*3, axis=2)
    plt.subplot(133)
    plt.imshow(masked_img)
    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)