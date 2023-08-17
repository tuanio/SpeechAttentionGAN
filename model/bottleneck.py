from torch import nn, Tensor

BOTTLE_NECK_DICT = {}


def register_bottle_neck(name, _class):
    BOTTLE_NECK_DICT[name] = _class


def get_bottle_neck(name: str, **kwargs):
    for _name, _class in BOTTLE_NECK_DICT.items():
        if name[: len(_name)] == _name:
            return _class(**kwargs)
    return None


class ResNetBottleNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_blocks: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer=nn.BatchNorm2d,  # can be instance norm
    ):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers.extend(self.create_block(in_channels, kernel_size, stride, padding))
        self.out_dim = in_channels
        self.norm_layer = norm_layer
        self.model = nn.Sequential(*layers)

        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def create_block(self, in_channels, kernel_size, stride, padding):
        # padding reflect to prevent checkboard artifacts
        return [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
            ),
            self.norm_layer(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
            ),
            self.norm_layer(in_channels),
        ]

    def forward(self, x: Tensor):
        return x + self.model(x)


register_bottle_neck("resnet", ResNetBottleNeck)
