from torch import nn


def init_resnet(
    resnet: nn.Module,
    method: str,
    zero_init_last: bool,
):
    """Initialize a ResNet model.

    It seems that both init methods yield slightly worse results than
    the default PyTorch initialization.
    `zero_init_last` will cause the performance to drop drastically.

    """
    for __m in resnet.modules():
        if isinstance(__m, nn.Conv2d) or isinstance(__m, nn.Linear):
            if method == "kaiming_normal":
                nn.init.kaiming_normal_(
                    __m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif method == "orthogonal":
                nn.init.orthogonal_(__m.weight)
            elif method == "none":
                pass
            else:
                raise ValueError(f"Invalid method: {method}")
        if hasattr(__m, "zero_init_last") and zero_init_last:
            __m.zero_init_last()
