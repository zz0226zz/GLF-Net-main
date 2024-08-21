from Regression.libs import *
from Regression.swin_model import swin_tiny_patch4_window7_224 as swin_tiny
from Regression.vit_model import vit_base_patch16_224_in21k as vit_l_16
from Regression.model import swin_tiny_patch4_window7_224 as GLF_Net

def get_model(model_name):

    if model_name == "vit_l_16":

        model = vit_l_16()
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()

    elif model_name == "swin_tiny":

        model = swin_tiny()
        num_features = model.head.in_features
        model.head = nn.Identity()

    elif model_name == "GLF_Net":

        model = GLF_Net()
        num_features = model.head.in_features
        model.head = nn.Identity()

    else:
        raise ValueError(f"Unknown model name {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    img_size = 224
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, num_features, preprocess
