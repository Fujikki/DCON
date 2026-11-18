import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_data_splits
from model import UNet
from loss import compute_metrics

PALETTE = {
    1: (0, 0, 230),     # Red: Gingivitis (歯肉炎)
    2: (0, 190, 255),   # Yellow-Orange: Dental Calculus (歯石)
    3: (220, 40, 120),  # Purple-Blue: Dental Caries (う蝕)
}

def overlay_mask(img_bgr, mask, alpha=0.5):
    out = img_bgr.copy()
    colored = np.zeros_like(out)
    for cls_id, color in PALETTE.items():
        colored[mask == cls_id] = color
    active = (mask > 0)
    out[active] = cv2.addWeighted(out, 1.0 - alpha, colored, alpha, 0)[active]
    return out

def add_label(img, text):
    h, w = img.shape[:2]
    bar = np.zeros((32, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, img])

def main():
    model_path = r"C:\Users\waaam\.gemini\antigravity-cli\brain\64abefda-bc93-44e9-b26a-2b29f68585a1\scratch\checkpoints\best_model.pth"
    out_dir = "baseline_rgb_eval"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=4, base_c=32)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_pairs, test_pairs = get_data_splits(root_dir="dataset")
    eval_pairs = test_pairs if len(test_pairs) > 0 else val_pairs

    # 歯肉炎(Label 1)を含むサンプルを10枚探す
    gingivitis_samples = []
    for img_p, lbl_p in eval_pairs:
        lbl = cv2.imread(lbl_p, cv2.IMREAD_UNCHANGED)
        if lbl is not None and np.any(lbl == 1):
            gingivitis_samples.append((img_p, lbl_p))
        if len(gingivitis_samples) >= 10:
            break

    print(f"Found {len(gingivitis_samples)} samples with Gingivitis (Label 1) for visual check.")

    img_size = (256, 384)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    all_ious = []

    for idx, (img_p, lbl_p) in enumerate(gingivitis_samples):
        img_bgr = cv2.imread(img_p)
        gt_mask = cv2.imread(lbl_p, cv2.IMREAD_UNCHANGED)
        gt_mask[gt_mask == 4] = 3

        h_orig, w_orig = img_bgr.shape[:2]

        # 前処理（学習時と全く同じ正規化）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - mean) / std
        tensor_in = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(tensor_in)
            logits_full = F.interpolate(logits, size=(h_orig, w_orig), mode="bilinear", align_corners=True)
            pred_mask = torch.argmax(logits_full, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # メトリクス計算
        _, _, ious, _ = compute_metrics(torch.from_numpy(pred_mask).unsqueeze(0), torch.from_numpy(gt_mask).unsqueeze(0))
        all_ious.append(ious)

        # 3面比較画像の生成
        disp_w, disp_h = 400, int(h_orig * (400 / w_orig))
        p1 = add_label(cv2.resize(img_bgr, (disp_w, disp_h)), "1. Original RGB")
        p2 = add_label(cv2.resize(overlay_mask(img_bgr, gt_mask), (disp_w, disp_h)), "2. Ground Truth (Label)")
        p3 = add_label(cv2.resize(overlay_mask(img_bgr, pred_mask), (disp_w, disp_h)), f"3. U-Net Pred (Ging IoU:{ious[1]:.2f})")

        combined = np.hstack([p1, p2, p3])
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        cv2.imwrite(os.path.join(out_dir, f"eval_{base_name}.jpg"), combined)

    mean_ious = np.mean(all_ious, axis=0)
    print("\n=== Normal RGB Baseline Class IoU (10 Gingivitis Samples) ===")
    print(f" - Class 0 (BG/Enamel): {mean_ious[0]:.4f}")
    print(f" - Class 1 (Gingivitis): {mean_ious[1]:.4f}")
    print(f" - Class 2 (Calculus):   {mean_ious[2]:.4f}")
    print(f" - Class 3 (Caries):     {mean_ious[3]:.4f}")
    print(f"\nSaved visual comparisons to: {out_dir}/")

if __name__ == "__main__":
    main()