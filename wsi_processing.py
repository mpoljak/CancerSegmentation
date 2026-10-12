import openslide
from PIL import Image


def extract_patch(slide_path, save_path):

    slide = openslide.OpenSlide(slide_path)

    # uzimamo mali dio velikog preparata
    patch = slide.read_region(
        (1000, 1000),
        0,
        (512, 512)
    )

    patch = patch.convert("RGB")

    patch.save(save_path)

    slide.close()


if __name__ == "__main__":

    print("WSI processing module ready")