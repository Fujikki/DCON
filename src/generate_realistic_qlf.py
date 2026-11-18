import cv2
import numpy as np
import os
from generate_samples import render_calibrated_qlf

def run():
    os.makedirs('pseudo_qlf_refined', exist_ok=True)
    sample_files = ['(1).jpg', '(1002).jpg', '(1168).jpg', '(1032).jpg', '(1254).jpg']
    for s in sample_files:
        base = os.path.splitext(s)[0]
        img = cv2.imread(f'dataset/images/{s}')
        lbl = cv2.imread(f'dataset/lables/{base}.png', cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None:
            continue
        
        qlf = render_calibrated_qlf(img, lbl)
        comparison = np.hstack([img, qlf])
        cv2.imwrite(f'pseudo_qlf_refined/{base}_compare.jpg', comparison)
        print(f"Generated refined comparison: pseudo_qlf_refined/{base}_compare.jpg")

if __name__ == '__main__':
    run()
