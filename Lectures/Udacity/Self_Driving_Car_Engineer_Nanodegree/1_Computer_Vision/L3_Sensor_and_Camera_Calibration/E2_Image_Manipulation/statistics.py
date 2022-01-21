import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import check_results

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    for i, path in enumerate(image_list):
        image = np.array(Image.open(path).convert('RGB'))
        if i == 0:
            images = image
        else:
            images = np.vstack([images, image])
    mean = np.array([np.mean(images[..., i]) for i in range(images.shape[-1])]).reshape(1, -1)
    std = np.array([np.std(images[..., i]) for i in range(images.shape[-1])]).reshape(1, -1)
    
    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION
    red_dict = {i: 0 for i in range(256)}
    green_dict = {i: 0 for i in range(256)}
    blue_dict = {i: 0 for i in range(256)}
    total_pixel_num = 0
    for path in image_list:
        image = np.array(Image.open(path).convert('RGB'))
        for i in range(image.shape[-1]):
            values, counts = np.unique(image[..., i], return_counts=True)
            # print(values, counts)
            if i == 0:
                for v, c in zip(values, counts):
                    red_dict[v] += c
            elif i == 1:
                for v, c in zip(values, counts):
                    green_dict[v] += c
            else:
                for v, c in zip(values, counts):
                    blue_dict[v] += c                
        total_pixel_num += image.shape[0] * image.shape[1]
    red_dict = {k: v / total_pixel_num for k, v in red_dict.items()}
    green_dict = {k: v / total_pixel_num for k, v in green_dict.items()}
    blue_dict = {k: v / total_pixel_num for k, v in blue_dict.items()}

    plt.figure(figsize=(10, 6))
    plt.plot(red_dict.keys(), red_dict.values(), color='r')
    plt.plot(green_dict.keys(), green_dict.values(), color='g')
    plt.plot(blue_dict.keys(), blue_dict.values(), color='b')
    
    plt.show()


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    # check_results(mean, std)
    channel_histogram(image_list)