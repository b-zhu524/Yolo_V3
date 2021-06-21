import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import YoloDataset


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
    stable_val = 1e-6

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
        recalls = TP_cumulative_sum / (total_true_bboxes + stable_val)
        precisions = torch.divide(TP_cumulative_sum, (TP_cumulative_sum + FP_cumulative_sum + epsilon))
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


def get_evaluation_bboxes(
        loader, model, iou_threshold, anchors,
        threshold, box_format="midpoint", device="cuda"
):
    # make sure model is in eval() before getting bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales predictions from model to be relative to the entire image
    so they can be plotted
    :param predictions: tensor of size (N, 3, S, S, num_classes + 5)
    :param anchors: the anchors used for predictions
    :param S: number of cells the image is divided in;
                num grids in width and height
    :param is_preds: whether the input is predictions or true bounding boxes
    :return:
        converted_bboxes: the converted boxes of sizes
        (N, num_anchors, S, S, 1+5) with class index,
        object score, bounding box coordinates
    """

    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1
                                 ).reshape(batch_size, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def check_class_accuracy(model, loader, threshold):
    model.eval()
    total_class_preds, correct_class = 0, 0
    total_no_obj, correct_no_obj = 0, 0
    total_obj, correct_obj = 0, 0

    device = config.device  # cuda
    stable_val = 1e-16

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            no_obj = y[i][..., 0] == 0 # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:], dim=-1) == y[i][..., 5][obj]
            )

            total_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            total_obj += torch.sum(obj)
            correct_no_obj += torch.sum(obj_preds[no_obj] == y[i][..., 0][no_obj])
            total_no_obj += torch.sum(no_obj)

    print(f"Class Accuracy is {(correct_class / (total_class_preds+stable_val))*100:2f}%")
    print(f"No obj accuracy is: {(correct_no_obj/(total_no_obj+stable_val))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(total_obj+stable_val))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[x] = e[x**2] - e[x]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_gropus:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    image_size = config.image_size

    train_dataset = YoloDataset(
        train_csv_path,
        transform=config.train_transforms,
        s=(image_size // 32, image_size // 16, image_size // 8),
        img_dir=config.img_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
        drop_last=True
    )   # drop_last ignores the last batch if not divisible

    test_dataset = YoloDataset(
        test_csv_path,
        transform=config.test_transforms,
        s=(image_size // 32, image_size // 16, image_size // 8),
        img_dir=config.img_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
        drop_last=False
    )

    train_eval_dataset = YoloDataset(
        train_csv_path,
        transform=config.test_transforms,
        s=(image_size // 32, image_size // 16, image_size // 8),
        img_dir=config.img_dir,
        label_dir=config.label_dir,
        anchors=config.anchors
    )

    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader, train_eval_loader


def plot_couple_examples(model, loader, threshold, iou_threshold, anchors):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint"
        )
        plot_image(x[i].permute(1, 2, 0).detatch().cpu(), nms_boxes)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
