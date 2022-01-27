import copy
import json
import pathlib

import numpy as np 
from PIL import Image

from utils import display_results, check_results

def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes  y1, x1, y2, x2 format
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

    width = img.size[0]
    flipped_bboxes = [[y1, width - x1, y2, width - x2] for y1, x1, y2, x2 in bboxes]

    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    resized_image = img.resize(size)
    
    origin_width, origin_height = img.size
    resize_width, resize_height = size
    
    width_ratio = resize_width / origin_width
    height_ratio = resize_height / origin_height
    resized_boxes = [[y1 * height_ratio, x1 * width_ratio, y2 * height_ratio, x2 * width_ratio] for y1, x1, y2, x2 in boxes]
    
    return resized_image, resized_boxes


def random_crop(img, boxes, classes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - classes [list]: list of classes
    - crop_size [array]: 1x2 array [width, height] ## 512, 512
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    origin_w, origin_h = img.size
    cropped_w, cropped_h = crop_size
    if cropped_w > origin_w or cropped_h > origin_h:
        print('[Cropped Image Size Error]')
        return None, None

    left = np.random.randint(0, origin_w - cropped_w)
    right = left + cropped_w
    bottom = np.random.randint(0, origin_h - cropped_h)
    top = bottom + cropped_h
    cropped_image = img.crop((left, bottom, right, top))
    cropped_box = [bottom, left, top, right]
    cropped_boxes = []
    cropped_classes = []
    for bb, cls in zip(boxes, classes):
        iou, tmp_box = calculate_iou(cropped_box, bb)
        if iou > 0:
            new_area = (tmp_box[3] - tmp_box[1]) * (tmp_box[2] - tmp_box[0])
            if new_area >= min_area:
                xmin = tmp_box[1] - left
                ymin = tmp_box[0] - bottom
                xmax = tmp_box[3] - left
                ymax = tmp_box[2] - bottom
                new_box = [ymin, xmin, ymax, xmax]
                cropped_boxes.append(new_box)
                cropped_classes.append(cls)
    
    return cropped_image, cropped_boxes, cropped_classes

if __name__ == '__main__':
    # fix seed to check results
    
    # open annotations
    
    # filter annotations and open image
    
    # check horizontal flip, resize and random crop
    # use check_results defined in utils.py for this
    
    ### TEST IMAGE : data/images/segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png
    
    gt_json_path = "./data/ground_truth.json"
    with open(gt_json_path, 'rb') as f:
        gt_json = json.load(f)
    
    img_base_path = pathlib.Path("./data/images")
    img_path = img_base_path.glob('*')
    TEST_IMG_PATH = pathlib.Path("./data/images/segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png")
    for path in img_path:
        img = Image.open(path)
        boxes = [gt['boxes'] for gt in gt_json if gt['filename'] == path.name][0]
        classes = [gt['classes'] for gt in gt_json if gt['filename'] == path.name][0]
        
        flipped_img, flipped_bboxes = hflip(img, boxes)
        resized_image, resized_boxes = resize(img, boxes, [640, 640])
        cropped_image, cropped_boxes, cropped_classes = random_crop(img, boxes, classes, [512, 512])
        
        display_results(img, boxes, flipped_img, flipped_bboxes)
        display_results(img, boxes, resized_image, resized_boxes)
        display_results(img, boxes, cropped_image, cropped_boxes)
        if path.name == TEST_IMG_PATH.name:
            check_results(img, boxes, 'hflip')
            check_results(img, boxes, 'resize')
        
        # break