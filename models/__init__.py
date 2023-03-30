from .UNet import UNet
def get_model(model_name):
    if model_name == 'resnet18':
        pass
    elif model_name == 'resnet34':
        pass
    elif model_name == 'resnet50':
        pass
    elif model_name == 'segformer':
        pass
    elif model_name == 'hrnet':
        pass
    elif model_name == 'unet':
        return UNet(n_channels=1, n_classes=1)
    elif model_name == 'ocrnet':
        pass
    elif model_name == 'upernet':
        pass
    elif model_name == 'swimtransformer':
        pass