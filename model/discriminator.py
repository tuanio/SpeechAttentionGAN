from torch import nn, Tensor

class PatchGAN(nn.Module):
    # 70x70 patch gan from cyclegan
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 3,
        kernel_size: int = 4,  # more than n_layers 2 layers (first and last conv)
        stride: int = 2,  # quickly downsample then fine-grained information
        padding: int = 1,
        channel_expand: int = 2,
        leakyrelu_slope: float = 0.2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        # norm_layer can be BatchNorm2d or InstanceNorm2d
        # not use instance norm for first conv
        layers = [
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=use_bias,
            ),
            nn.LeakyReLU(leakyrelu_slope, True),
        ]

        out_channels = hidden_channels
        for idx in range(n_layers):
            in_channels = out_channels
            out_channels = out_channels * channel_expand
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=padding,
                        bias=use_bias,
                    ),
                    norm_layer(out_channels),
                    nn.LeakyReLU(leakyrelu_slope, True),
                ]
            )

        # can use dilated convolution

        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    1,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.model(x)