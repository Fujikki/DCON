import cv2
import numpy as np
import os

def optical_qlf_simulation(img_bgr, lbl):
    """
    Physical Optical Simulation of QLF (405nm excitation + 520nm long-pass filter):
    - Teeth: Soft mint/cream autofluorescence preserving 100% natural texture and gradients.
    - Gums: Dark maroon-crimson with anatomical contours preserved.
    - Retractors/Mouth Cavity: Muted deep dark tones.
    - Calculus / Plaque: Optical red fluorescence EMISSION (additive physics, soft organic diffusion, not a flat sticker).
    - Caries: Natural demineralization fluorescence loss (Delta-F).
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    redness_ratio = (r - g) / (r + g + 0.05)
    
    # 1. Continuous Enamel Weight (soft transition, no jagged masks)
    # Gums have high redness_ratio (> 0.20), teeth have low redness_ratio (< 0.15)
    enamel_prob = np.clip((lum - 0.15) / 0.30, 0.0, 1.0) * np.clip((0.24 - redness_ratio) / 0.16, 0.0, 1.0)
    
    # Suppress blue retractor/bib reflection
    blue_excess = np.clip((b - r * 0.95) / 0.18, 0.0, 1.0)
    enamel_prob *= (1.0 - blue_excess)
    enamel_prob = cv2.GaussianBlur(enamel_prob, (7, 7), 0)

    # 2. Oral Mucosa / Gums Base:
    # Under 405nm + orange filter, gums look dark maroon/crimson with subtle anatomy
    tissue_r = np.power(r, 1.15) * 0.42 + 0.02
    tissue_g = np.power(g, 1.25) * 0.14 + 0.005
    tissue_b = np.power(b, 1.25) * 0.10 + 0.005

    # Retractor / Bib darkening
    tissue_r *= (1.0 - 0.85 * blue_excess)
    tissue_g *= (1.0 - 0.85 * blue_excess)
    tissue_b *= (1.0 - 0.85 * blue_excess)

    # 3. Sound Enamel Autofluorescence:
    # Calibrated to clinical QLF (R: 164, G: 183, B: 134)
    tooth_lum = np.power(lum, 0.82)
    enamel_r = tooth_lum * (164.0 / 220.0)
    enamel_g = np.clip(tooth_lum * (183.0 / 220.0), 0.0, 1.0)
    enamel_b = tooth_lum * (134.0 / 220.0)

    # Smooth continuous blend
    base_r = tissue_r * (1.0 - enamel_prob) + enamel_r * enamel_prob
    base_g = tissue_g * (1.0 - enamel_prob) + enamel_g * enamel_prob
    base_b = tissue_b * (1.0 - enamel_prob) + enamel_b * enamel_prob

    # 4. Calculus / Plaque (Label 2): Additive Porphyrin Red Fluorescence
    # True fluorescence is an emission of light, glowing organically over the tooth texture
    calc_mask = (lbl == 2).astype(np.uint8) * 255
    if np.any(calc_mask > 0):
        # Distance transform to create soft organic density (thicker in center, softer at edges)
        dist = cv2.distanceTransform(calc_mask, cv2.DIST_L2, 5)
        max_d = np.max(dist)
        if max_d > 0:
            core_density = np.clip(dist / (max_d * 0.6), 0.0, 1.0)
        else:
            core_density = (calc_mask / 255.0).astype(np.float32)
            
        # Optical blur to simulate light diffusion / glow
        bloom = cv2.GaussianBlur((calc_mask / 255.0).astype(np.float32), (13, 13), 0)
        
        # Combined organic fluorescence intensity map
        plaque_glow = core_density * 0.65 + bloom * 0.45
        
        # Emission strength modulated by natural illumination
        emission_r = plaque_glow * np.clip(lum * 0.7 + 0.45, 0.0, 1.2) * 0.85
        
        # Add red emission and naturally absorb a bit of the underlying green enamel glow
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

def test():
    os.makedirs("optical_test", exist_ok=True)
    for name in ['(1)', '(1001)', '(1002)', '(1168)', '(1254)']:
        img = cv2.imread(f'dataset/images/{name}.jpg')
        lbl = cv2.imread(f'dataset/lables/{name}.png', cv2.IMREAD_UNCHANGED)
        if img is None: continue
        res = optical_qlf_simulation(img, lbl)
        comp = np.hstack([img, res])
        cv2.imwrite(f'optical_test/{name}_opt2.jpg', comp)
        print(f"Saved optical_test/{name}_opt2.jpg")

if __name__ == '__main__':
    test()
