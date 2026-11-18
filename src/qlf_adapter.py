import cv2
import numpy as np

class QLFColorAdapter:
    """
    Domain adaptation for QLF (405 nm + orange barrier) images.

    Optical priors, measured on the Inspektor QLF-D reference:
      enamel autofluorescence  ~ RGB (164, 183, 134), bright, G >= R
      porphyrin plaque         ~ RGB (185, 106, 72),  R - G > 60
      gingiva                  ~ RGB (64, 19, 16),    dark red, not plaque
      background               ~ RGB (13, 4, 5)
    """
    def __init__(self, margin_dilation_kernel=23, clahe_clip=2.0):
        k = margin_dilation_kernel if margin_dilation_kernel % 2 == 1 else margin_dilation_kernel + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def extract_masks(self, img_bgr):
        """
        Extract anatomical, optical, and blackout masks from a QLF image.
        """
        bgr = img_bgr.astype(np.float32)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        brightness = (r + g + b) / 3.0
        delta_r = r - g

        # 1. Background & white paper border
        paper = (r > 235) & (g > 230) & (b > 220) & (np.abs(r - g) < 18)
        bg_mask = ((r < 22) & (g < 22) & (b < 22)) | paper

        # 2. Luminous enamel (健全歯面)
        green_enamel = (g > 105) & (g > r * 0.88) & (brightness > 85)
        bright_tooth = (brightness > 150) & (g > 95) & (delta_r < 50)
        enamel_raw = (green_enamel | bright_tooth) & (~bg_mask)
        enamel_u8 = cv2.morphologyEx(enamel_raw.astype(np.uint8), cv2.MORPH_CLOSE, self.close_kernel)
        enamel_u8 = cv2.morphologyEx(enamel_u8, cv2.MORPH_OPEN, self.open_kernel)
        enamel_mask = enamel_u8.astype(bool)

        # 3. Mature plaque / calculus (強く光るポルフィリン赤色蛍光の芯に絞る)
        plaque_raw = (
            (delta_r > 58)
            & (r > 155)
            & (r > g * 1.32)
            & (g < 160)
            & (brightness > 65)
            & (brightness < 190)
            & (~enamel_mask)
            & (~bg_mask)
        )
        plaque_u8 = cv2.morphologyEx(plaque_raw.astype(np.uint8), cv2.MORPH_CLOSE, self.open_kernel)
        plaque_u8 = cv2.morphologyEx(plaque_u8, cv2.MORPH_OPEN, self.open_kernel)
        plaque_mask = plaque_u8.astype(bool) & (~enamel_mask)

        # 4. 歯とプラークを合わせた前景シルエットを穴埋めし、境界の黒いスジを防止
        foreground_u8 = (enamel_mask | plaque_mask).astype(np.uint8)
        foreground_filled = cv2.morphologyEx(foreground_u8, cv2.MORPH_CLOSE, self.close_kernel)

        # 5. Gingival margin ROI (歯茎マージンを広めに確保し、境界の明るさ上限も145まで許容)
        dilated = cv2.dilate(foreground_filled, self.kernel).astype(bool)
        margin_roi = (
            dilated
            & (~enamel_mask)
            & (~plaque_mask)
            & (~bg_mask)
            & (brightness > 20)
            & (brightness < 145)
        )

        # 6. Blackout mask (上記以外の暗い口腔内背景・粘膜は黒塗り)
        blackout_mask = (~enamel_mask) & (~plaque_mask) & (~margin_roi)

        return {
            "enamel": enamel_mask,
            "plaque": plaque_mask,
            "margin_roi": margin_roi,
            "blackout": blackout_mask,
            "bg": bg_mask,
        }

    def adapt_image(self, img_bgr, mode="balanced"):
        """
        Transforms QLF image applying blackout filtering and domain color calibration.
        """
        bgr = img_bgr.astype(np.float32)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        brightness = (r + g + b) / 3.0

        masks = self.extract_masks(img_bgr)
        enamel_mask = masks["enamel"]
        plaque_mask = masks["plaque"]
        margin_roi = masks["margin_roi"]
        blackout_mask = masks["blackout"]

        adapted = np.zeros_like(bgr, dtype=np.float32)

        # 1. Tooth Enamel: Natural ivory-white
        enamel_luma = g[enamel_mask] / 255.0
        adapted[:, :, 2][enamel_mask] = np.clip(enamel_luma * 215.0 + 20, 0, 255) # R
        adapted[:, :, 1][enamel_mask] = np.clip(enamel_luma * 205.0 + 15, 0, 255) # G
        adapted[:, :, 0][enamel_mask] = np.clip(enamel_luma * 185.0 + 10, 0, 255) # B

        # 2. Porphyrin Red (Calculus): Yellowish-brown
        plaque_luma = r[plaque_mask] / 255.0
        adapted[:, :, 2][plaque_mask] = np.clip(plaque_luma * 210.0, 0, 255) # R
        adapted[:, :, 1][plaque_mask] = np.clip(plaque_luma * 170.0, 0, 255) # G
        adapted[:, :, 0][plaque_mask] = np.clip(plaque_luma * 70.0, 0, 255)  # B

        # 3. Gingival Margin ROI: Natural pink-red mucosa
        margin_luma = np.clip(brightness[margin_roi] / 80.0, 0.4, 1.2)
        adapted[:, :, 2][margin_roi] = np.clip(margin_luma * 185.0, 0, 255) # R
        adapted[:, :, 1][margin_roi] = np.clip(margin_luma * 95.0, 0, 255)  # G
        adapted[:, :, 0][margin_roi] = np.clip(margin_luma * 105.0, 0, 255) # B

        # 4. Blackout Filter
        adapted[blackout_mask] = 0

        adapted_uint8 = np.clip(adapted, 0, 255).astype(np.uint8)
        adapted_uint8 = cv2.bilateralFilter(adapted_uint8, d=5, sigmaColor=25, sigmaSpace=25)

        return adapted_uint8

    def extract_features(self, img_bgr):
        """
        Extract optical fluorometry maps:
          - porphyrin_red: delta R intensity (R - G)
        """
        bgr = img_bgr.astype(np.float32)
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

        porphyrin_map = np.clip(r - g, 0, 255).astype(np.uint8)
        porphyrin_norm = cv2.normalize(porphyrin_map, None, 0, 255, cv2.NORM_MINMAX)

        return {
            "porphyrin_red": porphyrin_norm,
            "porphyrin_raw": porphyrin_map
        }


def test_adapter_on_reference():
    import os
    ref_path = "extracted_qlf_ref/fig1_row_b.png"
    if not os.path.exists(ref_path):
        print(f"Reference file {ref_path} not found.")
        return

    adapter = QLFColorAdapter()
    qlf_img = cv2.imread(ref_path)
    adapted = adapter.adapt_image(qlf_img)
    feats = adapter.extract_features(qlf_img)

    out_dir = r"C:\Users\waaam\.gemini\antigravity-cli\brain\64abefda-bc93-44e9-b26a-2b29f68585a1\scratch\qlf_adaptation"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "qlf_original_ref.png"), qlf_img)
    cv2.imwrite(os.path.join(out_dir, "qlf_adapted_rgb.png"), adapted)
    cv2.imwrite(os.path.join(out_dir, "qlf_porphyrin_red.png"), feats["porphyrin_red"])

    h = min(qlf_img.shape[0], adapted.shape[0])
    w = int(qlf_img.shape[1] * (h / qlf_img.shape[0]))
    q_res = cv2.resize(qlf_img, (w, h))
    a_res = cv2.resize(adapted, (w, h))
    p_res = cv2.cvtColor(cv2.resize(feats["porphyrin_red"], (w, h)), cv2.COLOR_GRAY2BGR)

    combined = np.hstack([q_res, a_res, p_res])
    cv2.imwrite(os.path.join(out_dir, "qlf_adaptation_comparison.png"), combined)
    print(f"QLF Blackout & Adaptation successfully executed! Output at {out_dir}")


if __name__ == "__main__":
    test_adapter_on_reference()