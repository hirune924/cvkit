import segmentation_models_pytorch as smp

def build_model(conf):
    arch, encoder = conf.model_name.split('>')
    if arch == 'unet':
        return smp.Unet(encoder_name=encoder, in_channels=3, classes=1)