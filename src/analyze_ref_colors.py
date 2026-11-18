import cv2
import numpy as np

img = cv2.imread("extracted_qlf_ref/img_4_78.png")
h, w = img.shape[:2]
print(f"Fig 1 Image size: {w}x{h}")

# In Fig 1, there are 3 rows: A (top), B (middle, blue light QLF), C (bottom, disclosing solution)
# Let's crop row B
row_b = img[int(h * 0.33):int(h * 0.67), :]
cv2.imwrite("extracted_qlf_ref/fig1_row_b.png", row_b)

# Analyze colors in row B
# Sample points on teeth, gums, and plaque in row B
rb_rgb = cv2.cvtColor(row_b, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(row_b, cv2.COLOR_BGR2HSV)

# Teeth in QLF: high brightness
bright_teeth = (row_b[:, :, 1] > 100) & (row_b[:, :, 2] < 200) # Green channel strong, not dominated by red
if np.any(bright_teeth):
    mean_tooth_rgb = rb_rgb[bright_teeth].mean(axis=0)
    std_tooth_rgb = rb_rgb[bright_teeth].std(axis=0)
    print(f"Actual QLF Enamel Mean RGB: R={mean_tooth_rgb[0]:.1f}, G={mean_tooth_rgb[1]:.1f}, B={mean_tooth_rgb[2]:.1f}")
    print(f"Actual QLF Enamel Std RGB:  R={std_tooth_rgb[0]:.1f}, G={std_tooth_rgb[1]:.1f}, B={std_tooth_rgb[2]:.1f}")

# Plaque/Calculus in QLF: strong red fluorescence (R >> G and R >> B)
red_plaque = (row_b[:, :, 2] > 140) & (row_b[:, :, 2] > row_b[:, :, 1] * 1.3)
if np.any(red_plaque):
    mean_plaque_rgb = rb_rgb[red_plaque].mean(axis=0)
    print(f"Actual QLF Plaque Mean RGB: R={mean_plaque_rgb[0]:.1f}, G={mean_plaque_rgb[1]:.1f}, B={mean_plaque_rgb[2]:.1f}")
else:
    print("Plaque mask condition need adjustment")

# Gums / oral mucosa in QLF: low luminance, warm dark red-brown
gums = (row_b[:, :, 2] > 30) & (row_b[:, :, 2] < 120) & (row_b[:, :, 1] < 60) & (row_b[:, :, 0] < 40)
if np.any(gums):
    mean_gum_rgb = rb_rgb[gums].mean(axis=0)
    print(f"Actual QLF Gums Mean RGB:   R={mean_gum_rgb[0]:.1f}, G={mean_gum_rgb[1]:.1f}, B={mean_gum_rgb[2]:.1f}")

# Background: dark
bg = (row_b[:, :, 0] < 20) & (row_b[:, :, 1] < 20) & (row_b[:, :, 2] < 20)
if np.any(bg):
    mean_bg_rgb = rb_rgb[bg].mean(axis=0)
    print(f"Actual QLF Dark BG RGB:    R={mean_bg_rgb[0]:.1f}, G={mean_bg_rgb[1]:.1f}, B={mean_bg_rgb[2]:.1f}")
