import cv2
import numpy as np
import os
import glob

def generate_pseudo_qlf(img_bgr, lbl):
    """
    Simulate QLF (405nm excitation + orange long-pass barrier filter):
    - Sound tooth enamel: Strong green autofluorescence (preserving tooth surface texture/shading)
    - Dental calculus/plaque (Label 2): Bright red/orange fluorescence (porphyrin emission at ~635nm)
    - Dental caries (Label 3, 4): Dark regions due to fluorescence loss (Delta-F)
    - Gingiva/soft tissue (Label 1): Attenuated dark red-brown
    - Background/Retractors: Attenuated dark
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Base luminance / texture from the original image
    # Normalize luminance
    norm_lum = gray.astype(np.float32) / 255.0
    
    # 1. Tooth mask estimation:
    # Within Label 0, tooth surfaces typically have higher brightness and lower-to-medium saturation compared to pink/red gums and dark mouth cavity
    # Hue in HSV: 0-255. Red/pink gums are Hue near 0-15 or 160-180. Yellow/white teeth are Hue 15-45 or low saturation.
    h, s, v = cv2.split(hsv)
    
    # Tooth heuristic in label 0:
    is_label_0 = (lbl == 0)
    # Gums/lips in label 0 often have higher red saturation or lower brightness
    is_tooth_candidate = is_label_0 & (v > 70) & ((s < 90) | ((h >= 15) & (h <= 45)))
    
    # Morphological cleanup on tooth candidate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth_mask = cv2.morphologyEx(is_tooth_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    tooth_mask = cv2.GaussianBlur(tooth_mask.astype(np.float32), (7, 7), 0)

    # Initialize QLF image in float [0, 255] (BGR format)
    qlf_b = np.zeros_like(norm_lum)
    qlf_g = np.zeros_like(norm_lum)
    qlf_r = np.zeros_like(norm_lum)
    
    # 2. Sound Enamel Green Autofluorescence:
    # Under orange filter, enamel emits green (~520-540nm) with some yellow component:
    # High Green, moderate Red (giving yellowish-green glow), very low Blue (blocked by filter).
    enamel_brightness = np.clip(norm_lum * 1.3, 0.0, 1.0)
    enamel_g = enamel_brightness * 230.0
    enamel_r = enamel_brightness * 120.0
    enamel_b = enamel_brightness * 15.0
    
    qlf_g += enamel_g * tooth_mask
    qlf_r += enamel_r * tooth_mask
    qlf_b += enamel_b * tooth_mask

    # 3. Soft tissue / Gingiva (Label 1) and Non-tooth Label 0:
    # Gingival tissue under 405nm + orange filter appears dark reddish-brown / maroon
    gum_mask = (lbl == 1).astype(np.float32)
    other_soft_tissue = (is_label_0 & (tooth_mask < 0.3)).astype(np.float32)
    tissue_mask = np.clip(gum_mask + other_soft_tissue * 0.7, 0.0, 1.0)
    
    tissue_lum = np.clip(norm_lum * 0.5, 0.0, 1.0)
    tissue_r = tissue_lum * 110.0
    tissue_g = tissue_lum * 25.0
    tissue_b = tissue_lum * 10.0
    
    # Blend tissue where tooth is not present
    qlf_r += tissue_r * tissue_mask * (1.0 - tooth_mask)
    qlf_g += tissue_g * tissue_mask * (1.0 - tooth_mask)
    qlf_b += tissue_b * tissue_mask * (1.0 - tooth_mask)

    # 4. Calculus (Label 2):
    # Intense red/orange bacterial porphyrin fluorescence (~635nm)
    # Bright red with slight orange/yellow highlights
    calculus_mask = (lbl == 2).astype(np.float32)
    calculus_blur = cv2.GaussianBlur(calculus_mask, (5, 5), 0)
    
    calc_lum = np.clip(norm_lum * 1.2 + 0.2, 0.0, 1.0)
    calc_r = calc_lum * 255.0
    calc_g = calc_lum * 35.0  # slight orange/warmth
    calc_b = calc_lum * 10.0
    
    # Blend calculus onto tooth / tissue
    qlf_r = qlf_r * (1.0 - calculus_blur) + calc_r * calculus_blur
    qlf_g = qlf_g * (1.0 - calculus_blur) + calc_g * calculus_blur
    qlf_b = qlf_b * (1.0 - calculus_blur) + calc_b * calculus_blur

    # 5. Dental Caries (Label 3, 4):
    # Demineralization / Caries: Significant loss of green autofluorescence (Delta-F)
    # Appears dark/blackish-brown against the green fluorescent tooth.
    caries_mask = ((lbl == 3) | (lbl == 4)).astype(np.float32)
    caries_blur = cv2.GaussianBlur(caries_mask, (5, 5), 0)
    
    # Delta-F: Reduce intensity by 75-85%, dark brownish/black void
    darkening_factor = 1.0 - 0.80 * caries_blur
    qlf_r *= darkening_factor
    qlf_g *= darkening_factor
    qlf_b *= darkening_factor

    # Combine into uint8 BGR
    qlf = np.dstack([
        np.clip(qlf_b, 0, 255).astype(np.uint8),
        np.clip(qlf_g, 0, 255).astype(np.uint8),
        np.clip(qlf_r, 0, 255).astype(np.uint8)
    ])
    
    return qlf

def test_samples():
    os.makedirs('pseudo_qlf_samples', exist_ok=True)
    sample_files = ['(1).jpg', '(1002).jpg', '(1168).jpg', '(1032).jpg', '(1254).jpg']
    for s in sample_files:
        base = os.path.splitext(s)[0]
        img = cv2.imread(f'dataset/images/{s}')
        lbl = cv2.imread(f'dataset/lables/{base}.png', cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None: continue
        
        qlf = generate_pseudo_qlf(img, lbl)
        comparison = np.hstack([img, qlf])
        cv2.imwrite(f'pseudo_qlf_samples/{base}_compare.jpg', comparison)
        print(f"Generated pseudo_qlf_samples/{base}_compare.jpg")

if __name__ == '__main__':
    test_samples()
