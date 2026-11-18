import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OralDataset(Dataset):
    """
    Oral cavity disease segmentation dataset.
    Classes:
      0: Background / Healthy Tooth Enamel
      1: Gingivitis (歯肉炎)
      2: Dental Calculus (歯石)
      3: Dental Caries (う蝕 - merged labels 3 & 4)
    """
    def __init__(self, data_pairs, img_size=(256, 384), is_train=True):
        """
        data_pairs: list of tuples (img_path, lbl_path)
        img_size: (height, width)
        """
        self.data_pairs = data_pairs
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.data_pairs[idx]

        # Load image (BGR -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask (single channel)
        mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read label: {lbl_path}")

        # Resize
        h, w = self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Merge label 4 into label 3 (both represent dental caries)
        mask[mask == 4] = 3

        # Data augmentation for training
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # Random vertical flip (slight probability)
            if random.random() > 0.8:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            # Slight color jitter (brightness / contrast)
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15) # contrast
                beta = random.uniform(-15, 15)      # brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Normalize image to [0, 1] and standard ImageNet mean/std
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, os.path.basename(img_path)


def get_data_splits(root_dir="dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, val, and test subsets reproducibly.
    """
    img_dir = os.path.join(root_dir, "images")
    lbl_dir = os.path.join(root_dir, "lables")

    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    lbl_files = glob.glob(os.path.join(lbl_dir, "*.*"))

    lbl_map = {os.path.splitext(os.path.basename(f))[0]: f for f in lbl_files}
    pairs = []
    for img_p in img_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        if base in lbl_map:
            pairs.append((img_p, lbl_map[base]))

    pairs = sorted(pairs, key=lambda x: x[0])
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"Dataset Split Summary: Total={n_total}, Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


if __name__ == "__main__":
    train_p, val_p, test_p = get_data_splits()
    ds = OralDataset(train_p[:10], is_train=True)
    img, mask, name = ds[0]
    print(f"Sample item -> Image: {img.shape}, Mask: {mask.shape}, Unique labels: {torch.unique(mask).tolist()}, Name: {name}")
