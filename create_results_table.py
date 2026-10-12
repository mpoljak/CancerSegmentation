import os
import pandas as pd
import matplotlib.pyplot as plt


# Rezultati modela
data = {
    "Model": ["U-Net"],
    "Encoder": ["ResNet34"],
    "Dataset": ["SICAPv2"],
    "Dice coefficient": [0.7492],
    "IoU": [0.6969]
}


# Kreiranje tablice
df = pd.DataFrame(data)


# Kreiranje results foldera
os.makedirs("results", exist_ok=True)


# Spremanje kao CSV
df.to_csv(
    "results/model_results.csv",
    index=False
)


# Prikaz tablice kao slike
fig, ax = plt.subplots(figsize=(8, 2))

ax.axis("off")


table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center"
)


table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)


plt.savefig(
    "results/model_results.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("Rezultati spremljeni:")
print("results/model_results.csv")
print("results/model_results.png")