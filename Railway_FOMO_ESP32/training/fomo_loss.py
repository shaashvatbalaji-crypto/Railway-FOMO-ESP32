import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RAILWAY FOMO FOCAL LOSS
#
# Predictions:
#
#   [B, 5, H, W]
#
#   Channel 0 -> Objectness
#   Channel 1 -> Cracks
#   Channel 2 -> Scars
#   Channel 3 -> breaks
#   Channel 4 -> lightbands
#
#
# Targets:
#
#   [B, H, W, 2]
#
#   Channel 0 -> Objectness
#   Channel 1 -> Class index
#
#
# LOSS:
#
#   Total Loss =
#
#       Objectness Focal Loss
#       +
#       Weighted Classification Loss
#
# ============================================================


class FOMOLoss(nn.Module):

    def __init__(
        self,
        num_classes=4,
        class_weights=None,
        alpha=0.75,
        gamma=2.0,
        class_loss_weight=1.0
    ):

        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Focal loss parameters
        #
        # alpha:
        # Gives more importance to positive object cells.
        #
        # gamma:
        # Reduces the contribution of easy examples.
        # ----------------------------------------------------

        self.alpha = alpha
        self.gamma = gamma

        self.class_loss_weight = (
            class_loss_weight
        )

        # ----------------------------------------------------
        # Exact class weights obtained from the
        # 560-image training dataset.
        #
        # Cracks      = 58
        # Scars       = 189
        # breaks      = 72
        # lightbands  = 146
        # ----------------------------------------------------

        if class_weights is None:

            class_weights = [
                3.26,   # Cracks
                1.00,   # Scars
                2.62,   # breaks
                1.29    # lightbands
            ]

        if len(class_weights) != num_classes:

            raise ValueError(
                "class_weights must contain "
                f"{num_classes} values."
            )

        self.register_buffer(
            "class_weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

    # ========================================================
    # FOCAL OBJECTNESS LOSS
    # ========================================================

    def focal_objectness_loss(
        self,
        logits,
        targets
    ):

        # ----------------------------------------------------
        # Binary cross entropy without reduction
        # ----------------------------------------------------

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Convert logits → probability
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        # ----------------------------------------------------
        # p_t:
        #
        # For positive:
        #     p_t = p
        #
        # For negative:
        #     p_t = 1-p
        # ----------------------------------------------------

        p_t = (
            probabilities * targets
            +
            (1.0 - probabilities)
            * (1.0 - targets)
        )

        # ----------------------------------------------------
        # Alpha balancing
        #
        # Positive cells:
        #     alpha
        #
        # Background:
        #     1-alpha
        # ----------------------------------------------------

        alpha_t = (
            self.alpha * targets
            +
            (1.0 - self.alpha)
            * (1.0 - targets)
        )

        # ----------------------------------------------------
        # Focal modulation
        #
        # Easy examples → small contribution
        # Hard examples → large contribution
        # ----------------------------------------------------

        focal_factor = (
            1.0 - p_t
        ).pow(
            self.gamma
        )

        loss = (
            alpha_t
            * focal_factor
            * bce
        )

        return loss.mean()

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        predictions,
        targets
    ):

        # ----------------------------------------------------
        # Validate prediction shape
        # ----------------------------------------------------

        if predictions.ndim != 4:

            raise ValueError(
                "Predictions must have shape "
                "[B,5,H,W]."
            )

        if targets.ndim != 4:

            raise ValueError(
                "Targets must have shape "
                "[B,H,W,2]."
            )

        batch_size = (
            predictions.shape[0]
        )

        channels = (
            predictions.shape[1]
        )

        height = (
            predictions.shape[2]
        )

        width = (
            predictions.shape[3]
        )

        expected_channels = (
            self.num_classes + 1
        )

        if channels != expected_channels:

            raise ValueError(
                f"Expected {expected_channels} "
                f"prediction channels, "
                f"got {channels}."
            )

        if targets.shape != (
            batch_size,
            height,
            width,
            2
        ):

            raise ValueError(
                "Target shape must be "
                f"[B,H,W,2]. "
                f"Got {tuple(targets.shape)}."
            )

        # ====================================================
        # OBJECTNESS
        # ====================================================

        objectness_logits = (
            predictions[:, 0, :, :]
        )

        objectness_targets = (
            targets[:, :, :, 0]
        )

        # ----------------------------------------------------
        # Focal objectness loss
        # ----------------------------------------------------

        objectness_loss = (
            self.focal_objectness_loss(
                objectness_logits,
                objectness_targets
            )
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        class_logits = (
            predictions[:, 1:, :, :]
        )

        class_targets = (
            targets[:, :, :, 1]
            .long()
        )

        # ----------------------------------------------------
        # Only classify cells that actually contain
        # a defect.
        # ----------------------------------------------------

        positive_mask = (
            objectness_targets > 0.5
        )

        if positive_mask.any():

            # ------------------------------------------------
            # [B,4,H,W]
            #
            # →
            #
            # [B,H,W,4]
            # ------------------------------------------------

            class_logits = (
                class_logits.permute(
                    0,
                    2,
                    3,
                    1
                )
            )

            # ------------------------------------------------
            # Select only positive cells
            # ------------------------------------------------

            positive_logits = (
                class_logits[
                    positive_mask
                ]
            )

            positive_targets = (
                class_targets[
                    positive_mask
                ]
            )

            # ------------------------------------------------
            # Weighted cross entropy
            # ------------------------------------------------

            class_loss = F.cross_entropy(
                positive_logits,
                positive_targets,
                weight=self.class_weights.to(
                    predictions.device
                ),
                reduction="mean"
            )

        else:

            class_loss = torch.zeros(
                (),
                device=predictions.device,
                dtype=predictions.dtype
            )

        # ====================================================
        # TOTAL LOSS
        # ====================================================

        total_loss = (
            objectness_loss
            +
            self.class_loss_weight
            * class_loss
        )

        return total_loss


# ============================================================
# LOSS TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY FOMO FOCAL LOSS TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Fake predictions
    #
    # [B,5,H,W]
    # --------------------------------------------------------

    predictions = torch.randn(
        2,
        5,
        12,
        12,
        device=device,
        requires_grad=True
    )

    # --------------------------------------------------------
    # Fake targets
    #
    # [B,H,W,2]
    #
    # Channel 0 = objectness
    # Channel 1 = class index
    # --------------------------------------------------------

    targets = torch.zeros(
        2,
        12,
        12,
        2,
        device=device
    )

    # --------------------------------------------------------
    # Sample 1
    #
    # Crack
    # --------------------------------------------------------

    targets[
        0,
        5,
        6,
        0
    ] = 1.0

    targets[
        0,
        5,
        6,
        1
    ] = 0.0

    # --------------------------------------------------------
    # Sample 2
    #
    # breaks
    # --------------------------------------------------------

    targets[
        1,
        4,
        7,
        0
    ] = 1.0

    targets[
        1,
        4,
        7,
        1
    ] = 2.0

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    criterion = FOMOLoss(
        num_classes=4,

        # Exact class weights calculated
        # from the 560-image dataset.
        class_weights=[
            3.26,
            1.00,
            2.62,
            1.29
        ],

        # Focal loss parameters
        alpha=0.75,
        gamma=2.0,

        class_loss_weight=1.0
    ).to(device)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = criterion(
        predictions,
        targets
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    print()
    print(
        "Prediction shape :",
        tuple(predictions.shape)
    )

    print(
        "Target shape     :",
        tuple(targets.shape)
    )

    print(
        "Loss             :",
        loss.item()
    )

    print(
        "Gradient exists  :",
        predictions.grad is not None
    )

    print(
        "Gradient finite  :",
        torch.isfinite(
            predictions.grad
        ).all().item()
    )

    print(
        "Gradient mean    :",
        predictions.grad.abs().mean().item()
    )

    print()
    print(
        "Focal alpha      :",
        criterion.alpha
    )

    print(
        "Focal gamma      :",
        criterion.gamma
    )

    print(
        "Class weights    :",
        criterion.class_weights.tolist()
    )

    print()

    if (
        torch.isfinite(loss)
        and
        predictions.grad is not None
        and
        torch.isfinite(
            predictions.grad
        ).all()
    ):

        print(
            "✓ FOCAL LOSS TEST PASSED"
        )

    else:

        print(
            "✗ FOCAL LOSS TEST FAILED"
        )

    print()
    print("=" * 60)
    print("FOCAL LOSS TEST COMPLETE")
    print("=" * 60)