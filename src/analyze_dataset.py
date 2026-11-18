import os
import glob
import cv2
import numpy as np

def analyze():
    lbl_files = sorted(glob.glob('dataset/lables/*.png'))
    print(f"Total label files: {len(lbl_files)}")

    val_counts = {}
    samples_by_val = {v: [] for v in range(5)}
    color_stats = {v: [] for v in range(5)}

    for f in lbl_files:
        lbl = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            continue
        u, counts = np.unique(lbl, return_counts=True)
        for val, c in zip(u, counts):
            val_counts[val] = val_counts.get(val, 0) + c
        
        for v in range(5):
            if v in u and len(samples_by_val[v]) < 3:
                samples_by_val[v].append(os.path.basename(f))

    print("\n--- Pixel Count by Label Value (All Samples) ---")
    for k in sorted(val_counts.keys()):
        print(f"Label {k}: {val_counts[k]:,} pixels")

    print("\n--- Sample Filenames per Label Value ---")
    for v, fs in samples_by_val.items():
        print(f"Label {v}: {fs}")

    # Inspect average RGB in first 150 images
    for f in lbl_files[:150]:
        base = os.path.basename(f)
        img_name = os.path.splitext(base)[0] + '.jpg'
        img_path = os.path.join('dataset', 'images', img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        lbl = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None or img.shape[:2] != lbl.shape[:2]:
            continue

        for v in range(5):
            mask = (lbl == v)
            if np.any(mask):
                color_stats[v].append(img[mask].mean(axis=0))

    print("\n--- Average Color (RGB) by Label Region (First 150 images) ---")
    for v in range(5):
        if color_stats[v]:
            avg_bgr = np.mean(color_stats[v], axis=0)
            print(f"Label {v} -> R:{avg_bgr[2]:.1f}, G:{avg_bgr[1]:.1f}, B:{avg_bgr[0]:.1f} (Sample count: {len(color_stats[v])})")

if __name__ == '__main__':
    analyze()
