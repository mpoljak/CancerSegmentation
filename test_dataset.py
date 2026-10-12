from dataset import CancerDataset


dataset = CancerDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks"
)


print("Broj slika:", len(dataset))