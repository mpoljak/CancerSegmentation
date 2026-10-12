import matplotlib.pyplot as plt
import os


# Loss vrijednosti dobivene tijekom treninga
epochs = [
    5,
    6,
    7,
    8,
    9,
    10
]

loss_values = [
    0.7703,
    0.7590,
    0.7464,
    0.7386,
    0.7317,
    0.7263
]


# Kreiranje results foldera
os.makedirs("results", exist_ok=True)


# Crtanje grafa
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss_values,
    marker="o"
)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")


plt.grid(True)


# Spremanje
plt.savefig(
    "results/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("Graf spremljen u results/training_loss.png")