import fitz  # PyMuPDF
import cv2
import numpy as np
import os

doc = fitz.open("text/qlf_art.pdf")
os.makedirs("extracted_qlf_ref", exist_ok=True)

# Render Page 4 (Fig 1) and Page 5 (Fig 2) at high DPI
for pno in [3, 4]:  # 0-indexed: page 4 is 3, page 5 is 4
    page = doc[pno]
    pix = page.get_pixmap(dpi=300)
    out_path = f"extracted_qlf_ref/page_{pno+1}.png"
    pix.save(out_path)
    print(f"Saved {out_path}")

print("Extracting embedded images from doc...")
img_count = 0
for i, page in enumerate(doc):
    for img in page.get_images(full=True):
        xref = img[0]
        base_img = doc.extract_image(xref)
        img_bytes = base_img["image"]
        img_ext = base_img["ext"]
        out_name = f"extracted_qlf_ref/img_{i+1}_{xref}.{img_ext}"
        with open(out_name, "wb") as f:
            f.write(img_bytes)
        img_count += 1
        print(f"Saved embedded image: {out_name} ({base_img['width']}x{base_img['height']})")

doc.close()
