import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ProstateDataset
from model import create_model


# Dataset
dataset = ProstateDataset(
    "dataset/images",
    "dataset/masks"
)


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# Uređaj
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Uređaj:", device)


# Model
model = create_model()
model.to(device)


# Loss funkcija
bce_loss = torch.nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    smooth = 1e-6

    pred = torch.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# Trening
epochs = 10


for epoch in range(epochs):

    model.train()

    running_loss = 0.0


    progress = tqdm(loader)

    for images, masks in progress:

        images = images.to(device)
        masks = masks.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        progress.set_description(
            f"Epoch {epoch+1}/{epochs}"
        )


    avg_loss = running_loss / len(loader)


    print(
        f"Epoch {epoch+1} Loss: {avg_loss:.4f}"
    )


# Spremanje modela
torch.save(
    model.state_dict(),
    "cancer_segmentation_model.pth"
)


print("Trening završen!")