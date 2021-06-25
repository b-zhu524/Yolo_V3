import config
import torch
import torch.optim as optim
from model import YoloV3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings


warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.device) # image
        y0, y1, y2 = (
            y[0].to(config.device),
            y[1].to(config.device),
            y[2].to(config.device),
        )   # tuple(targets)

        optimizer.zero_grad()
        # forward
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors)
            )

        losses.append(loss.item())

        # backward
        scaler.scale(loss).backward()

        # gradient descent
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    model = YoloV3(num_classes=config.num_classes)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.dataset + "/train.csv",
        test_csv_path=config.dataset + "/test.csv"
    )

    if config.load_model:
        load_checkpoint(config.checkpoint_file, model, optimizer, config.learning_rate)

    scaled_anchors = (
        torch.tensor(config.anchors)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.device) # double unsqueeze to add more dimensions

    for epoch in range(config.num_epochs):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.save_model:
            save_checkpoint(model, optimizer)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.conf_threshold)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.nms_iou_thresh,
                anchors=config.anchors,
                threshold=config.conf_threshold
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.map_iou_thresh,
                box_format="midpoint",
                num_classes=config.num_classes
            )
            print(f"mAP: {mapval.item()}")
            model.train()


if __name__ == '__main__':
    main()
