import os
import random
import torch
import matplotlib.pyplot as plt

from dataset import ProstateDataset
from model import create_model


# Dataset
dataset = ProstateDataset(
    "dataset/images",
    "dataset/masks"
)


# Uređaj
device = "cuda" if torch.cuda.is_available() else "cpu"


# Učitavanje modela
model = create_model()

model.load_state_dict(
    torch.load(
        "cancer_segmentation_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# Folder za rezultate
os.makedirs("results", exist_ok=True)


# Broj nasumičnih slika
num_images = 5


# Nasumični indeksi
random_indices = random.sample(
    range(len(dataset)),
    num_images
)


for i, index in enumerate(random_indices):

    image, mask = dataset[index]


    image_input = image.unsqueeze(0).to(device)


    # Predikcija
    with torch.no_grad():
        output = model(image_input)


    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.5).float()


    # Pretvaranje u numpy
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()


    # Prikaz
    plt.figure(figsize=(12, 4))


    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Originalna slika #{index}")
    plt.axis("off")


    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Stvarna maska")
    plt.axis("off")


    plt.subplot(1, 3, 3)
    plt.imshow(prediction_np, cmap="gray")
    plt.title("Predikcija modela")
    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        f"results/random_prediction_{i}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


print("Spremljeno 5 nasumičnih predikcija!")
print("Odabrane slike:", random_indices)