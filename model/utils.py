from torch import nn, Tensor

conv_keep_params = ["in_channels", "out_channels", "kernel_size", "stride"]
conv_replace_params = {"in_channels": 1}


def replace_params(replace_params, keep_params, old_net, new_net):
    params = {i: j for i, j in vars(old_net).items() if i in keep_params}
    params.update(replace_params)
    return new_net(**params)


def conv_do_replace(old_net, in_channels):
    return replace_params(
        {"in_channels": in_channels}, conv_keep_params, old_net, nn.Conv2d
    )


class Identity(nn.Module):
    def forward(self, x: Tensor):
        return x

def get_criterion(name):
    # already with sigmoid
    if name == 'l2':
        return nn.MSELoss()
    if name == 'l1':
        return nn.L1Loss()
    elif name == 'bce':
        return nn.BCELoss()
    return nn.Identity()