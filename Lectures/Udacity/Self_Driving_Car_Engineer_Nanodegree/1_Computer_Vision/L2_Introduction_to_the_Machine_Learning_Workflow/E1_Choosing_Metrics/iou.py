import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    ## IMPLEMENT THIS FUNCTION
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox    
    iou = 0
    if max(x1_pred, x1_gt) < min(x2_pred, x2_gt):
        if max(y1_pred, y1_gt) < min(y2_pred, y2_gt):
            intersection = np.abs(min(x2_pred, x2_gt) - max(x1_pred, x1_gt)) * np.abs(min(y2_pred, y2_gt) - max(y1_pred, y1_gt))
            union = np.abs(x2_gt - x1_gt) * np.abs(y2_gt - y1_gt) + np.abs(x2_pred - x1_pred) * np.abs(y2_pred - y1_pred) - intersection
            iou = intersection / union
    iou = float(iou)
    return iou


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)