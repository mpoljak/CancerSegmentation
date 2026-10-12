import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ProstateDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        image_path = os.path.join(
            self.image_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[index]
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask