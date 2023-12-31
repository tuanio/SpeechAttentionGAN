import torch
from torch import nn, Tensor
from .upsample import get_upsample
from .downsample import get_downsample
from .bottleneck import get_bottle_neck


class AttentionGuideGenerator(nn.Module):
    def __init__(
        self,
        downsample_name: str,
        bottle_neck_name: str,
        upsample_name: str,
        downsample_params: dict,
        bottle_neck_params: dict,
        upsample_content_params: dict,
        upsample_attn_params: dict,
        **kwargs
    ):
        super().__init__()
        self.downsample = get_downsample(downsample_name, **downsample_params)
        self.bottle_neck = get_bottle_neck(
            bottle_neck_name, in_channels=self.downsample.out_dim, **bottle_neck_params
        )
        # self.upsample_attn = get_upsample(
        #     upsample_name, in_channels=self.bottle_neck.out_dim, **upsample_attn_params
        # )
        in_dim = self.bottle_neck.out_dim
        self.upsample_attn = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, 3, 2, 1, 1),
            nn.InstanceNorm2d(in_dim // 2),
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 3, 2, 1, 1),
            nn.InstanceNorm2d(in_dim // 4),
            nn.Conv2d(in_dim // 4, in_dim // 4, (3, 1), 1, 0),
            nn.InstanceNorm2d(in_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_dim // 4, 10, (2, 1), 1, 0),
        )
        self.upsample_content = get_upsample(
            upsample_name,
            in_channels=self.bottle_neck.out_dim,
            **upsample_content_params
        )
        # self.tanh = nn.Tanh()
        self.content_activation = nn.Identity()
        self.attn_activation = nn.Softmax(dim=1)

    def forward(self, x: Tensor, mask: Tensor):
        inp = torch.cat([x * mask, mask], dim=1)
        enc = self.downsample(inp)
        emb = self.bottle_neck(enc)
        attn_masks = self.attn_activation(self.upsample_attn(emb))
        contents = self.content_activation(self.upsample_content(emb))

        bg_mask = attn_masks[:, -1:, :, :]
        attn_masks = attn_masks[:, :-1, :, :]

        fake_img = (attn_masks * contents).sum(dim=1, keepdim=True) + x * bg_mask
        return fake_img
