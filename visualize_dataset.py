import matplotlib.pyplot as plt
from dataset import CancerDataset


dataset = CancerDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks"
)


image, mask = dataset[0]


plt.figure(figsize=(10,5))


plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")


plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")


plt.show()
