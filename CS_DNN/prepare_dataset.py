"""
prepare_dataset.py
------------------
One-time script: reads the raw `archive/` folder and builds a clean
`dataset/` structure with 80/20 train/test split.

archive/
  fresh_apple/    →  normal
  fresh_banana/   →  normal
  ...
  stale_apple/    →  damaged
  stale_banana/   →  damaged
  ...

Output:
dataset/
  train/
    normal/
    damaged/
  test/
    normal/
    damaged/

All images are resized to 128×128 px.
"""

import os
import shutil
import random
import cv2

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARCHIVE    = os.path.join(BASE_DIR, "archive")
DATASET    = os.path.join(BASE_DIR, "dataset")

IMG_SIZE   = (128, 128)
SPLIT      = 0.8          # 80 % train, 20 % test
SEED       = 42

# ── Helpers ──────────────────────────────────────────────────────────────────
def gather_images(root, prefix):
    """Return all (image_path, fruit_name) where parent folder starts with `prefix`."""
    paths = []
    for folder in os.listdir(root):
        if folder.lower().startswith(prefix):
            # E.g. "fresh_apple" -> "apple"
            fruit_name = folder.lower()[len(prefix):].strip("_")
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    paths.append((os.path.join(folder_path, fname), fruit_name))
    return paths


def copy_resized(src_info, dst_dir, label):
    """Copy & resize images to dst_dir, returning count of successfully copied images."""
    os.makedirs(dst_dir, exist_ok=True)
    ok = 0
    for i, (src, fruit_name) in enumerate(src_info):
        img = cv2.imread(src)
        if img is None:
            continue
        img_resized = cv2.resize(img, IMG_SIZE)
        ext      = os.path.splitext(src)[1]
        dst_name = f"{fruit_name}_{label}_{i:05d}{ext}"
        cv2.imwrite(os.path.join(dst_dir, dst_name), img_resized)
        ok += 1
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    print("Scanning archive …")
    normal_imgs  = gather_images(ARCHIVE, "fresh")
    damaged_imgs = gather_images(ARCHIVE, "stale")

    random.shuffle(normal_imgs)
    random.shuffle(damaged_imgs)

    def split(lst):
        n = int(len(lst) * SPLIT)
        return lst[:n], lst[n:]

    normal_train,  normal_test  = split(normal_imgs)
    damaged_train, damaged_test = split(damaged_imgs)

    print(f"  Normal  -> train: {len(normal_train):4d}  test: {len(normal_test):4d}")
    print(f"  Damaged -> train: {len(damaged_train):4d}  test: {len(damaged_test):4d}")

    # Remove old dataset folder so we start clean
    if os.path.exists(DATASET):
        shutil.rmtree(DATASET)

    pairs = [
        (normal_train,  os.path.join(DATASET, "train", "normal"),  "normal"),
        (normal_test,   os.path.join(DATASET, "test",  "normal"),  "normal"),
        (damaged_train, os.path.join(DATASET, "train", "damaged"), "damaged"),
        (damaged_test,  os.path.join(DATASET, "test",  "damaged"), "damaged"),
    ]

    total = 0
    for paths, dst, label in pairs:
        n = copy_resized(paths, dst, label)
        print(f"  Copied {n:4d} images -> {os.path.relpath(dst, BASE_DIR)}")
        total += n

    print(f"\nDone. {total} images written to dataset/")


if __name__ == "__main__":
    main()
