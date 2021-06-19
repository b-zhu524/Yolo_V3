import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything

dataset = "PASCAL_VOC"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed_everything()
num_workers = 4
batch_size = 32
image_size = 416,
num_classes = 20
learning_rate = 1e-5
weight_decay = 1e-4
num_epochs = 100
conf_threshold = 0.05
map_iou_thresh = 0.5
nms_iou_thresh = 0.45
S = [image_size // 32, image_size // 16, image_size // 8]
pin_memory = True
load_model = True
save_model = True
checkpoint_file = "checkpoint.pth.tar"
img_dir = dataset + "/images/"
label_dir = dataset + "/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
# rescaled to be between [0, 1]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(image_size * scale)),
        A.PadIfNeeded(
            min_height=int(image_size * scale),
            min_width=int(image_size * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=image_size, height=image_size),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

