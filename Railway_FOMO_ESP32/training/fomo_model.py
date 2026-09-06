import torch
import torch.nn as nn


# ============================================================
# DEPTHWISE-SEPARABLE CONVOLUTION
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.block = nn.Sequential(

            # ------------------------------------------------
            # Depthwise convolution
            # ------------------------------------------------

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),

            nn.BatchNorm2d(
                in_channels
            ),

            nn.ReLU6(
                inplace=True
            ),

            # ------------------------------------------------
            # Pointwise convolution
            # ------------------------------------------------

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU6(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# LIGHTWEIGHT FOMO MODEL
#
# INPUT:
#     [B, 3, 96, 96]
#
# OUTPUT:
#     [B, 5, 12, 12]
#
# CHANNELS:
#
#     0 -> Objectness
#     1 -> Cracks
#     2 -> Scars
#     3 -> breaks
#     4 -> lightbands
# ============================================================

class FOMOModel(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # 96 -> 48

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU6(
                inplace=True
            ),

            # 48 -> 48

            DepthwiseSeparableConv(
                16,
                24,
                stride=1
            ),

            # 48 -> 24

            DepthwiseSeparableConv(
                24,
                32,
                stride=2
            ),

            # 24 -> 24

            DepthwiseSeparableConv(
                32,
                32,
                stride=1
            ),

            # 24 -> 12

            DepthwiseSeparableConv(
                32,
                48,
                stride=2
            ),

            # 12 -> 12

            DepthwiseSeparableConv(
                48,
                48,
                stride=1
            )
        )

        # ----------------------------------------------------
        # 1×1 FOMO head
        #
        # 4 classes + objectness = 5
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            48,
            num_classes + 1,
            kernel_size=1
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.head(
            x
        )

        return x


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY FOMO 5-CHANNEL MODEL TEST")
    print("=" * 60)

    model = FOMOModel(
        num_classes=4
    )

    test_input = torch.randn(
        1,
        3,
        96,
        96
    )

    print()
    print(
        "Input shape :",
        tuple(test_input.shape)
    )

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        "Output shape:",
        tuple(output.shape)
    )

    print(
        "Parameters  :",
        count_parameters(model)
    )

    print()
    print(
        "Channels:"
    )

    print(
        "  0 -> Objectness"
    )

    print(
        "  1 -> Cracks"
    )

    print(
        "  2 -> Scars"
    )

    print(
        "  3 -> breaks"
    )

    print(
        "  4 -> lightbands"
    )

    expected_shape = (
        1,
        5,
        12,
        12
    )

    print()
    print(
        "Expected output:",
        expected_shape
    )

    if tuple(output.shape) == expected_shape:

        print(
            "✓ OUTPUT SHAPE CORRECT"
        )

    else:

        print(
            "✗ OUTPUT SHAPE INCORRECT"
        )

    if torch.isfinite(output).all():

        print(
            "✓ OUTPUT VALUES FINITE"
        )

    else:

        print(
            "✗ OUTPUT VALUES INVALID"
        )

    print()
    print("=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)