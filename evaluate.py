import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ProstateDataset
from model import create_model


# Dataset (Validation)
dataset = ProstateDataset(
    "dataset/images",
    "dataset/masks"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_model()
model.load_state_dict(
    torch.load("cancer_segmentation_model.pth", map_location=device)
)

model.to(device)
model.eval()


def dice_score(pred, target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2 * intersection + 1e-6) / (
        pred.sum() + target.sum() + 1e-6
    )

    return dice.item()


def iou_score(pred, target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()


dice_values = []
iou_values = []


with torch.no_grad():

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        dice_values.append(
            dice_score(outputs, masks)
        )

        iou_values.append(
            iou_score(outputs, masks)
        )


print("Prosječni Dice:", np.mean(dice_values))
print("Prosječni IoU :", np.mean(iou_values))