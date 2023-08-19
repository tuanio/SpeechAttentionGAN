from torch import nn, Tensor

UPSAMPLE_DICT = {}


def register_upsample(name, _class):
    UPSAMPLE_DICT[name] = _class


def get_upsample(name: str, **kwargs):
    for _name, _class in UPSAMPLE_DICT.items():
        if name[: len(_name)] == _name:
            return _class(**kwargs)
    return None


class Print(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.size())
        return x


class SimpleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_expand: int = 2,
        n_blocks: int = 3,
        kernel_sizes=[3, 3, 7],
        strides=[2, 2, 1],
        paddings=[1, 1, 1],
        output_padding=[1, 1, 0],
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        layers = []
        in_channels = in_channels // (channel_expand * (n_blocks - 1))
        self.activation = activation
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[-1],
                    stride=strides[-1],
                    padding=paddings[-1],
                ),
                nn.ReflectionPad2d(3),
            ]
        )
        for idx in range(n_blocks - 2, -1, -1):
            out_channels = in_channels
            in_channels *= channel_expand
            layers.extend(
                self.create_block(
                    in_channels,
                    out_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    paddings[idx],
                    output_padding[idx],
                    idx == idx - 1,
                )
            )

        layers = layers[::-1]

        self.model = nn.Sequential(*layers)

    def create_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        is_output,
    ):
        blocks = []
        if not is_output:
            blocks = [self.activation(), nn.InstanceNorm2d(out_channels)]
        return blocks + [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
        ]

    def forward(self, x: Tensor):
        return self.model(x)


class AttentionMaskUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_expand: int = 2,
        n_blocks: int = 3,
        kernel_sizes=[3, 3, 1],
        strides=[1, 1, 1],
        paddings=[1, 1, 0],
        output_padding=[1, 1, 0],
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        layers = []
        in_channels = in_channels // (channel_expand * (n_blocks - 1))
        self.activation = activation
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[-1],
                    stride=strides[-1],
                    padding=paddings[-1],
                ),
                nn.ReflectionPad2d(3),
            ]
        )
        for idx in range(n_blocks - 2, -1, -1):
            out_channels = in_channels
            in_channels *= channel_expand
            layers.extend(
                self.create_block(
                    in_channels,
                    out_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    paddings[idx],
                    output_padding[idx],
                    idx == idx - 1,
                )
            )

        layers = layers[::-1]

        self.model = nn.Sequential(*layers)

    def create_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        is_output,
    ):
        blocks = []
        if not is_output:
            blocks = [self.activation(), nn.InstanceNorm2d(out_channels)]
        return blocks + [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
        ]

    def forward(self, x: Tensor):
        return self.model(x)


register_upsample("simple", SimpleUpsample)
