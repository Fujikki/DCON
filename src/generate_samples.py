import os
import glob
import random
import argparse
import cv2
import numpy as np

def render_calibrated_qlf(img_bgr, lbl):
    """
    Physical Optical Simulation of QLF (405nm excitation + 520nm long-pass filter):
    - Continuous enamel weight preserving 100% natural texture and gradients (no jagged mask cutouts).
    - Gums / Oral Mucosa: Dark maroon-crimson with anatomical depth preserved.
    - Retractors / Mouth Cavity: Muted deep dark tones.
    - Calculus / Plaque (Label 2): Optical red fluorescence emission with distance-based density decay and soft bloom.
    - Caries (Labels 3, 4): Natural demineralization fluorescence loss (Delta-F).
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    redness_ratio = (r - g) / (r + g + 0.05)
    
    # 1. Continuous Enamel Probability Map
    enamel_prob = np.clip((lum - 0.15) / 0.30, 0.0, 1.0) * np.clip((0.24 - redness_ratio) / 0.16, 0.0, 1.0)
    
    # Suppress blue retractor/bib reflection
    blue_excess = np.clip((b - r * 0.95) / 0.18, 0.0, 1.0)
    enamel_prob *= (1.0 - blue_excess)
    enamel_prob = cv2.GaussianBlur(enamel_prob, (7, 7), 0)

    # 2. Oral Mucosa / Gums Base:
    tissue_r = np.power(r, 1.15) * 0.42 + 0.02
    tissue_g = np.power(g, 1.25) * 0.14 + 0.005
    tissue_b = np.power(b, 1.25) * 0.10 + 0.005

    # Retractor / Bib darkening
    tissue_r *= (1.0 - 0.85 * blue_excess)
    tissue_g *= (1.0 - 0.85 * blue_excess)
    tissue_b *= (1.0 - 0.85 * blue_excess)

    # 3. Sound Enamel Autofluorescence (calibrated to Inspektor QLF-D: R:164, G:183, B:134)
    tooth_lum = np.power(lum, 0.82)
    enamel_r = tooth_lum * (164.0 / 220.0)
    enamel_g = np.clip(tooth_lum * (183.0 / 220.0), 0.0, 1.0)
    enamel_b = tooth_lum * (134.0 / 220.0)

    # Smooth continuous blend
    base_r = tissue_r * (1.0 - enamel_prob) + enamel_r * enamel_prob
    base_g = tissue_g * (1.0 - enamel_prob) + enamel_g * enamel_prob
    base_b = tissue_b * (1.0 - enamel_prob) + enamel_b * enamel_prob

    # 4. Calculus / Plaque (Label 2): Additive Porphyrin Red Fluorescence Emission
    calc_mask = (lbl == 2).astype(np.uint8) * 255
    if np.any(calc_mask > 0):
        dist = cv2.distanceTransform(calc_mask, cv2.DIST_L2, 5)
        max_d = np.max(dist)
        if max_d > 0:
            core_density = np.clip(dist / (max_d * 0.6), 0.0, 1.0)
        else:
            core_density = (calc_mask / 255.0).astype(np.float32)
            
        bloom = cv2.GaussianBlur((calc_mask / 255.0).astype(np.float32), (13, 13), 0)
        plaque_glow = core_density * 0.65 + bloom * 0.45
        
        emission_r = plaque_glow * np.clip(lum * 0.7 + 0.45, 0.0, 1.2) * 0.85
        base_r = np.clip(base_r * (1.0 - plaque_glow * 0.15) + emission_r * 0.95, 0.0, 1.0)
        base_g = np.clip(base_g * (1.0 - plaque_glow * 0.55) + emission_r * 0.18, 0.0, 1.0)
        base_b = np.clip(base_b * (1.0 - plaque_glow * 0.60) + emission_r * 0.10, 0.0, 1.0)

    # 5. Caries (Labels 3, 4): Delta-F Fluorescence Loss
    caries_mask = ((lbl == 3) | (lbl == 4)).astype(np.uint8) * 255
    if np.any(caries_mask > 0):
        caries_blur = cv2.GaussianBlur((caries_mask / 255.0).astype(np.float32), (7, 7), 0)
        loss = 1.0 - 0.45 * caries_blur
        base_r *= loss
        base_g *= loss
        base_b *= (1.0 - 0.55 * caries_blur)

    out_bgr = np.dstack([
        np.clip(base_b * 255.0, 0, 255).astype(np.uint8),
        np.clip(base_g * 255.0, 0, 255).astype(np.uint8),
        np.clip(base_r * 255.0, 0, 255).astype(np.uint8)
    ])
    return out_bgr

def create_mask_overlay(img_bgr, lbl):
    overlay = img_bgr.copy()
    color_map = {
        1: (0, 0, 255),    # Gingivitis (Red)
        2: (0, 255, 255),  # Calculus (Yellow)
        3: (255, 100, 0),  # Caries (Blue)
        4: (255, 100, 0)   # Caries (Blue)
    }
    for val, col in color_map.items():
        mask = (lbl == val)
        if np.any(mask):
            overlay[mask] = col
    return cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)

def add_header(img, text):
    h, w = img.shape[:2]
    header = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, img])

def main():
    parser = argparse.ArgumentParser(description="Generate sample Pseudo-QLF images from dataset.")
    parser.add_argument("--num", "-n", type=int, default=6, help="Number of random samples to generate")
    parser.add_argument("--ids", nargs="+", help="Specific image IDs/filenames to process")
    parser.add_argument("--by-category", action="store_true", help="Pick representative samples across disease categories")
    parser.add_argument("--outdir", default="sample_outputs", help="Output directory for generated samples")
    parser.add_argument("--panels", choices=["2", "3"], default="3", help="Panel style: '2' or '3'")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    lbl_files = sorted(glob.glob("dataset/lables/*.png"))
    if not lbl_files:
        lbl_files = sorted(glob.glob("agent_dir/dataset/lables/*.png"))
    
    if not lbl_files:
        print("Error: Could not find labels in dataset/lables")
        return

    base_dir = os.path.dirname(os.path.dirname(lbl_files[0]))
    images_dir = os.path.join(base_dir, "images")
    lables_dir = os.path.join(base_dir, "lables")

    selected_bases = []

    if args.ids:
        for item in args.ids:
            name = os.path.splitext(os.path.basename(item))[0]
            selected_bases.append(name)
    elif args.by_category:
        print("Selecting representative samples across categories (Calculus, Caries, Gingivitis)...")
        cats = {'calculus': [], 'caries': [], 'gingivitis': [], 'healthy': []}
        for f in lbl_files:
            bname = os.path.splitext(os.path.basename(f))[0]
            lbl = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if lbl is None: continue
            u = set(np.unique(lbl))
            if (3 in u or 4 in u) and len(cats['caries']) < 3:
                cats['caries'].append(bname)
            elif 2 in u and len(cats['calculus']) < 3:
                cats['calculus'].append(bname)
            elif 1 in u and len(cats['gingivitis']) < 3:
                cats['gingivitis'].append(bname)
            elif u == {0} and len(cats['healthy']) < 2:
                cats['healthy'].append(bname)
            if all(len(v) >= 2 for v in cats.values()):
                break
        for v in cats.values():
            selected_bases.extend(v)
    else:
        all_bases = [os.path.splitext(os.path.basename(f))[0] for f in lbl_files]
        selected_bases = random.sample(all_bases, min(args.num, len(all_bases)))

    print(f"\nProcessing {len(selected_bases)} samples: {selected_bases}\n")
    generated_paths = []

    for base in selected_bases:
        img_path = os.path.join(images_dir, f"{base}.jpg")
        lbl_path = os.path.join(lables_dir, f"{base}.png")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"Skipping {base}: file not found.")
            continue

        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None: continue

        qlf = render_calibrated_qlf(img, lbl)
        
        p1 = add_header(img, "Original RGB")
        p3 = add_header(qlf, "Calibrated Optical Pseudo-QLF")

        if args.panels == "3":
            mask_vis = create_mask_overlay(img, lbl)
            p2 = add_header(mask_vis, "Labels (Red:Ging, Yel:Calc, Blu:Caries)")
            combined = np.hstack([p1, p2, p3])
        else:
            combined = np.hstack([p1, p3])

        out_path = os.path.join(args.outdir, f"sample_{base}.jpg")
        cv2.imwrite(out_path, combined)
        generated_paths.append(out_path)
        print(f" Saved: {out_path}")

    # Generate lightweight HTML preview gallery
    html_path = os.path.join(args.outdir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Calibrated Optical Pseudo-QLF Sample Viewer</title>")
        f.write("<style>body{background:#121212;color:#eee;font-family:system-ui,-apple-system,sans-serif;padding:30px;text-align:center}")
        f.write(".sample{margin-bottom:35px;background:#1e1e1e;padding:20px;border-radius:10px;box-shadow:0 4px 14px rgba(0,0,0,0.6);display:inline-block;max-width:98%}")
        f.write("img{max-width:100%;height:auto;border-radius:6px}h3{margin:0 0 12px 0;color:#58a6ff;font-size:1.2rem}</style></head><body>")
        f.write("<h1>Calibrated Optical Pseudo-QLF Sample Viewer</h1>")
        f.write("<p style='color:#aaa;'>Optical Physics Model: Additive Porphyrin Red Fluorescence + Continuous Enamel Autofluorescence</p>")
        for p in generated_paths:
            bname = os.path.basename(p)
            f.write(f"<div class='sample'><h3>{bname}</h3><img src='{bname}' /></div><br/>")
        f.write("</body></html>")

    print(f"\nHTML Preview generated: {html_path}")
    print("Done!")

if __name__ == '__main__':
    main()
