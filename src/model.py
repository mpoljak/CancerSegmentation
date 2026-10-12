import segmentation_models_pytorch as smp


def create_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    return model


if __name__ == "__main__":

    print("Model file pokrenut")

    model = create_model()

    print(model)