import timm

def build_model(conf):

    return timm.create_model(model_name=conf.model_name, num_classes=len(conf.classes), pretrained=True, in_chans=3)