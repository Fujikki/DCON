import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Combines CrossEntropyLoss and Multiclass Dice Loss.
    Handles class imbalance effectively in medical / oral segmentation.
    """
    def __init__(self, n_classes=4, ce_weight=None, dice_weight=1.0, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) long tensor
        """
        ce_loss = self.ce(logits, targets)

        # Compute Dice loss over each class
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        # Compute dice for non-background classes (1, 2, 3), and background (0)
        for c in range(self.n_classes):
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets_one_hot[:, c, :, :].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_total += (1.0 - dice)

        dice_loss = dice_total / self.n_classes
        return ce_loss + self.dice_weight * dice_loss


def compute_metrics(preds, targets, n_classes=4, smooth=1e-5):
    """
    preds: (B, H, W)
    targets: (B, H, W)
    Returns: mIoU, per-class IoUs, per-class Dice
    """
    ious = []
    dices = []
    for c in range(n_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        iou = (intersection + smooth) / (union + smooth) if union > 0 else 1.0
        dice = (2.0 * intersection + smooth) / (pred_c.sum().item() + target_c.sum().item() + smooth) if (pred_c.sum().item() + target_c.sum().item()) > 0 else 1.0

        ious.append(iou)
        dices.append(dice)

    mean_iou = sum(ious) / len(ious)
    mean_dice = sum(dices) / len(dices)
    return mean_iou, mean_dice, ious, dices


if __name__ == "__main__":
    criterion = CombinedLoss(n_classes=4)
    logits = torch.randn(2, 4, 128, 128)
    targets = torch.randint(0, 4, (2, 128, 128))
    loss = criterion(logits, targets)
    preds = logits.argmax(dim=1)
    miou, mdice, ious, dices = compute_metrics(preds, targets)
    print(f"Loss: {loss.item():.4f}, mIoU: {miou:.4f}, Mean Dice: {mdice:.4f}")
    print(f"Class IoUs: {[round(x, 4) for x in ious]}")
