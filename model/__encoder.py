from . import utils
from torch import nn, Tensor
from torchvision.models import get_model

# Testing here: https://colab.research.google.com/drive/1c7BCwacGiAbXNLDb3esJwqtSiscfYZyB?usp=sharing

ENCODER_DICT = {}


def register_encoder(name, encoder_class):
    ENCODER_DICT[name] = encoder_class


def get_encoder(name: str, **kwargs):
    for encoder_name, encoder_class in ENCODER_DICT.items():
        if name[: len(encoder_name)] == encoder_name:
            return encoder_class(name, **kwargs)
    return None


class ResNet(nn.Module):
    name = "resnet"
    sizes = ["18", "34", "50", "101", "152"]

    def __init__(self, size: str, in_channels: int = 2, **kwargs):
        super().__init__(**kwargs)
        assert size[len(self.name) :] in self.sizes, "No size in desire in torchvision"
        model = get_model(size, weights=None)
        model = list(model.children())[:-2]
        model[0] = utils.conv_do_replace(model[0], in_channels)
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor):
        return self.model(x)


class ConvNeXt(nn.Module):
    name = "convnext_"
    sizes = ["tiny", "small", "base", "large"]

    def __init__(self, size: str, in_channels: int = 2, **kwargs):
        super().__init__(**kwargs)
        assert size[len(self.name) :] in self.sizes, "No size in desire in torchvision"
        model = get_model(size, weights=None).features
        model[0][0] = utils.conv_do_replace(model[0][0], in_channels)
        self.model = nn.Sequential(model)

    def forward(self, x: Tensor):
        return self.model(x)


# class ViT(BaseEncoder):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.name = 'vit_'
#         self.sizes = ['b_16', 'b_32', 'l_16', 'l_32', 'h_14']

#     def create_model(self, size: str, in_channels: int, **kwargs):
#         model = get_model(size, weights=None)
#         old_conv = model.conv_proj


register_encoder("resnet", ResNet)
register_encoder("convnext", ConvNeXt)
