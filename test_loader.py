from dataset import ProstateDataset


dataset = ProstateDataset(
    "dataset/images",
    "dataset/masks"
)

image, mask = dataset[0]

print("Slika:", image.shape)
print("Maska:", mask.shape)