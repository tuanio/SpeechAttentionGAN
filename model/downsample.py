from torch import nn, Tensor

DOWNSAMPLE = {}


def register_downsample(name, _class):
    DOWNSAMPLE[name] = _class


def get_downsample(name: str, **kwargs):
    for _name, _class in DOWNSAMPLE.items():
        if name[: len(_name)] == _name:
            return _class(**kwargs)
    return None


class SimpleDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        channel_expand: int = 2,
        n_blocks: int = 3,
        kernel_sizes=[7, 3, 3],
        strides=[1, 1, 2],
        paddings=[0, 1, 1],
        dilations=[1, 1, 1],
        norm_layer=nn.InstanceNorm2d,  # can be instance norm
        activation=nn.ReLU,  # can be relu
        **kwargs,
    ):
        super().__init__()
        layers = []
        out_channels = hidden_channels
        self.norm_layer = norm_layer
        self.activation = activation

        for idx in range(n_blocks):
            layers.extend(
                self.create_block(
                    in_channels,
                    out_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    paddings[idx],
                    dilations[idx],
                )
            )
            in_channels = out_channels
            out_channels = out_channels * channel_expand
        self.out_dim = out_channels // channel_expand
        self.model = nn.Sequential(*layers)
        self.pad_reflect = nn.ReflectionPad2d(3)

    def create_block(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation
    ):
        return [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, dilation
            ),
            self.norm_layer(out_channels),
            self.activation(),
        ]

    def forward(self, x: Tensor):
        x = self.pad_reflect(x)
        return self.model(x)


register_downsample("simple", SimpleDownsample)
