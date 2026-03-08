import segmentation_models_pytorch as smp


def build_model(encoder='resnet18', encoder_weights='imagenet'):
    """
    Build a U-Net segmentation model.

    Args:
        encoder (str): encoder backbone (e.g., resnet18, resnet34)
        encoder_weights (str): pretrained weights source

    Returns:
        PyTorch model
    """

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,     # RGB input
        classes=1,         # binary mask output
        activation='sigmoid'  # output in [0,1]
    )

    return model


if __name__ == '__main__':

    # quick test
    model = build_model()

    print("Model structure:")
    print(model)

    print("\nModel created successfully")