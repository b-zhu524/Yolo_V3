# import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
         boxes1 (tensor): width and height of the first bounding boxes
         boxes2 (tensor): width and height of the second bounding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )

    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )

    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """

    :param boxes_preds (tensor): Predictions of bounding boxes (batch_size, 4)
    :param boxes_labels (tensor): Correct labels of bounding boxes (batch_size, 4)
    :param box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    :return:
        tensor: intersection / union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) for the case if they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    :param bboxes (list): List lof lists containing all bboxes with each bboxes
                            specified  as [class_pred, prob_score, x1, y1, x2, y2]
    :param iou_threshold (float): threshold were predicted bboxes is correct
    :param threshold (float): threshold to remove predicted bboxes (independet of iou)
    :param box_formats (str): "midpoint" or "corners" used to specifty bboxes

    :return:
        list: bboxes after performing nms given a specific iou threshold
    """

    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """

    :param pred_boxes (list): list of lists containing all bboxes with each bbox
                                [train_idx, class_preds, prob_score, x1, x2, y1, y2]
    :param true_boxes (list): correct boxes, same form as pred_boxes
    :param iou_threshold (float): threshold where predicted bbox is corect
    :param box_format (string): midpoint or corners, specify boxes
    :param num_classes (int): number of classes
    :return:
        float: map value across all classes given specific IOU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # numerical stability
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # GO through all predictions and targets
        # add only ones that belong to current class "c"

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                true_boxes.append(true_box)

                # img 0 has 3 bboxes
                # img 1 has 5 bboxes
                # amount_bboxes = {0:3, 1:5)
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            for detection_idx, detection in enumerate(detections):
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_ground_truths = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumulative_sum = torch.cumsum(TP, dim=0)
        FP_cumulative_sum = torch.cumsum(FP, dim=0)
        recalls = TP_cumulative_sum / (total_true_bboxes + epsilon)
        precisions = torch.devide(TP_cumulative_sum, (TP_cumulative_sum + FP_cumulative_sum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions)) # cat to [1] for y axis
        recalls = torch.cat((torch.tensor([0]), recalls))   # cat to [0] for x axis
        average_precisions.append(torch.trapz)

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.dataset=="COCO" else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain [class_pred, confidence, x, y, w, h]"
        class_pred = box[0]
        box = box[2:]   # box[2:] is [x, y, w, h]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none"
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0}
        )

    plt.show()





























