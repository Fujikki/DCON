import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import UNet
from qlf_adapter import QLFColorAdapter

PALETTE = {
    1: (0, 0, 230),     # Red: Gingivitis (歯肉炎)
    2: (0, 190, 255),   # Yellow-Orange: Dental Calculus / Plaque (歯石・プラーク)
    3: (220, 40, 120),  # Purple-Blue: Dental Caries (う蝕)
}

CLASS_NAMES = {
    0: "Healthy Enamel / Background (正常歯面・黒塗り背景)",
    1: "Gingivitis (歯肉炎 - 歯頸部マージン)",
    2: "Dental Calculus / Plaque (歯石・成熟プラーク)",
    3: "Dental Caries (う蝕)"
}


def create_overlay(base_img_bgr, pred_mask, alpha=0.55):
    """Overlay segmentation colors on top of base image."""
    overlay = base_img_bgr.copy()
    h, w = pred_mask.shape

    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

    colored_mask = np.zeros_like(overlay)
    for cls_idx, color in PALETTE.items():
        colored_mask[pred_mask == cls_idx] = color

    has_disease = (pred_mask > 0)
    overlay[has_disease] = cv2.addWeighted(overlay, 1.0 - alpha, colored_mask, alpha, 0)[has_disease]
    return overlay


def infer_single_image(model, adapter, img_bgr, device, img_size=(256, 384), gingivitis_prob_thresh=0.25):
    """
    Run clinically sound inference combining:
    - Blackout strategy on non-fluorescent low-luma pixels
    - Active optical QLF signals (enamel green + porphyrin red)
    - Anatomical gingival margin ROI with adjusted gingivitis sensitivity
    """
    h_orig, w_orig = img_bgr.shape[:2]

    # Step 1: Optical, Anatomical, and Blackout Masks
    masks = adapter.extract_masks(img_bgr)
    enamel_mask = masks["enamel"]
    plaque_mask = masks["plaque"]
    margin_roi = masks["margin_roi"]

    # Step 2: Blackout & Color Adaptation
    adapted_bgr = adapter.adapt_image(img_bgr, mode="balanced")
    feats = adapter.extract_features(img_bgr)

    # Step 3: Neural Net Forward Pass
    adapted_rgb = cv2.cvtColor(adapted_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(adapted_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)

    norm_img = resized_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (norm_img - mean) / std

    tensor_in = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        logits_full = F.interpolate(logits, size=(h_orig, w_orig), mode="bilinear", align_corners=True)
        probs = F.softmax(logits_full, dim=1).squeeze(0).cpu().numpy()
        raw_pred = np.argmax(probs, axis=0).astype(np.uint8)

    # Step 4: Optical prior + Gingivitis probability threshold inside safe gingival margin
    pred_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    gingivitis_ai = (raw_pred == 1) | (probs[1] > gingivitis_prob_thresh)
    gingivitis_refined = gingivitis_ai & margin_roi & (~enamel_mask) & (~plaque_mask)

    pred_mask[gingivitis_refined] = 1
    pred_mask[plaque_mask] = 2

    # Overlays
    overlay_qlf = create_overlay(img_bgr, pred_mask, alpha=0.55)
    overlay_adapted = create_overlay(adapted_bgr, pred_mask, alpha=0.55)

    return {
        "adapted_bgr": adapted_bgr,
        "porphyrin_red": feats["porphyrin_red"],
        "pred_mask": pred_mask,
        "overlay_qlf": overlay_qlf,
        "overlay_adapted": overlay_adapted,
        "probs": probs
    }


def main():
    parser = argparse.ArgumentParser(description="QLF Clinical Inference Pipeline (Blackout Optimized)")
    parser.add_argument("--model_path", type=str, 
                        default=r"C:\Users\waaam\.gemini\antigravity-cli\brain\64abefda-bc93-44e9-b26a-2b29f68585a1\scratch\checkpoints\best_model.pth")
    parser.add_argument("--input_path", type=str, default="extracted_qlf_ref/fig1_row_b.png")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\Users\waaam\.gemini\antigravity-cli\brain\64abefda-bc93-44e9-b26a-2b29f68585a1\scratch\qlf_inference")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gingivitis_thresh", type=float, default=0.25, help="Probability threshold for gingivitis inside margin ROI")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading base segmentation model from: {args.model_path}")
    model = UNet(n_channels=3, n_classes=4, base_c=32)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    adapter = QLFColorAdapter(margin_dilation_kernel=23)

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    img_bgr = cv2.imread(args.input_path)
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    print(f"Processing QLF Image: {args.input_path} (Size: {img_bgr.shape[1]}x{img_bgr.shape[0]})")

    results = infer_single_image(model, adapter, img_bgr, device, gingivitis_prob_thresh=args.gingivitis_thresh)

    # Save outputs
    cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_orig_qlf.png"), img_bgr)
    cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_adapted_rgb.png"), results["adapted_bgr"])
    cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_porphyrin_red.png"), results["porphyrin_red"])
    cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_overlay_qlf.png"), results["overlay_qlf"])
    cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_overlay_adapted.png"), results["overlay_adapted"])

    # Create composite 4-panel dashboard
    h, w = img_bgr.shape[:2]
    disp_w = 480
    disp_h = int(h * (disp_w / w))

    p1 = cv2.resize(img_bgr, (disp_w, disp_h))
    p2 = cv2.resize(results["adapted_bgr"], (disp_w, disp_h))
    p3 = cv2.cvtColor(cv2.resize(results["porphyrin_red"], (disp_w, disp_h)), cv2.COLOR_GRAY2BGR)
    p4 = cv2.resize(results["overlay_qlf"], (disp_w, disp_h))

    def put_lbl(im, text):
        out = im.copy()
        cv2.rectangle(out, (0, 0), (disp_w, 32), (20, 20, 20), -1)
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return out

    top_row = np.hstack([put_lbl(p1, "1. Input QLF Clinical Image"), put_lbl(p2, "2. Blackout & Color-Adapted RGB")])
    bot_row = np.hstack([put_lbl(p3, "3. Porphyrin Red (Calculus Signal)"), put_lbl(p4, "4. Pinpoint AI Screening Overlay")])
    dashboard = np.vstack([top_row, bot_row])

    dash_path = os.path.join(args.output_dir, f"{base_name}_screening_dashboard.png")
    cv2.imwrite(dash_path, dashboard)

    pred_mask = results["pred_mask"]
    total_px = pred_mask.size
    print("\n=== AI Screening Diagnosis Summary (Blackout Optimized) ===")
    for c_id, name in CLASS_NAMES.items():
        count = np.sum(pred_mask == c_id)
        ratio = (count / total_px) * 100.0
        print(f" - {name}: {count:,} px ({ratio:.2f}%)")

    print(f"\nScreening results saved successfully to: {args.output_dir}")
    print(f"Dashboard image: {dash_path}")


if __name__ == "__main__":
    main()