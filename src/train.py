import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OralDataset, get_data_splits
from model import UNet
from loss import CombinedLoss, compute_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks.cpu())

    val_loss = running_loss / total_samples
    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    miou, mdice, ious, dices = compute_metrics(cat_preds, cat_targets, n_classes=4)

    return val_loss, miou, mdice, ious, dices


def main():
    parser = argparse.ArgumentParser(description="Train Oral Healthcare Segmentation Base Model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_h", type=int, default=256, help="Image height")
    parser.add_argument("--img_w", type=int, default=384, help="Image width")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional subset limit for fast training/testing")
    scratch_default = r"C:\Users\waaam\.gemini\antigravity-cli\brain\64abefda-bc93-44e9-b26a-2b29f68585a1\scratch\checkpoints"
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, or cuda")
    parser.add_argument("--save_dir", type=str, default=scratch_default, help="Directory to save model weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        # Auto detect with active kernel test
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                t = torch.ones((2, 2), device="cuda")
                _ = t @ t
                device = torch.device("cuda")
            except Exception as e:
                print(f"[Device Notice] CUDA capability check failed (falling back to CPU): {e}")
                device = torch.device("cpu")

    print(f"==================================================")
    print(f" Training Base Segmentation Model on Oral Dataset")
    print(f" Device: {device} | Image Size: ({args.img_h}, {args.img_w})")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"==================================================")

    # Data splits
    train_pairs, val_pairs, test_pairs = get_data_splits(root_dir=args.data_dir)
    if args.sample_limit:
        train_pairs = train_pairs[:args.sample_limit]
        val_pairs = val_pairs[:max(10, args.sample_limit // 4)]
        print(f"Subsampled dataset for fast iteration: Train={len(train_pairs)}, Val={len(val_pairs)}")

    train_dataset = OralDataset(train_pairs, img_size=(args.img_h, args.img_w), is_train=True)
    val_dataset = OralDataset(val_pairs, img_size=(args.img_h, args.img_w), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model, Loss, Optimizer
    model = UNet(n_channels=3, n_classes=4, base_c=32).to(device)
    # 面積の大きい背景(0)の重みを下げ、歯肉炎(1)・歯石(2)・う蝕(3)のペナルティを重くする
    class_weights = torch.tensor([0.3, 2.5, 2.0, 2.0], dtype=torch.float32).to(device)
    criterion = CombinedLoss(n_classes=4, ce_weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou, val_mdice, ious, dices = validate(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val Dice: {val_mdice:.4f}")
        print(f"   Per-class IoU -> [BG: {ious[0]:.3f}, Gingivitis: {ious[1]:.3f}, Calculus: {ious[2]:.3f}, Caries: {ious[3]:.3f}]")

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_miou': val_miou,
            'val_loss': val_loss,
            'ious': ious,
            'dices': dices
        }, checkpoint_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_miou': val_miou,
                'ious': ious,
                'dices': dices
            }, best_path)
            print(f"   ★ Best model updated! (mIoU: {val_miou:.4f}) -> {best_path}")

    total_time = time.time() - start_time
    print(f"==================================================")
    print(f" Training Completed in {total_time/60:.2f} mins. Best Val mIoU: {best_miou:.4f}")
    print(f" Checkpoints saved at: {os.path.abspath(args.save_dir)}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
