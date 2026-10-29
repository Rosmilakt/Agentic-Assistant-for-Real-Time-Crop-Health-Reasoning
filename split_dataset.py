"""
split_dataset.py
PlantVillage is often distributed as one folder per class with no train/val
split. This script creates an 80/20 split expected by train.py.

Usage:
    python src/split_dataset.py --src raw_plantvillage --dst data --val_ratio 0.2
"""

import argparse
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="raw_plantvillage", help="Folder containing one subfolder per class")
    parser.add_argument("--dst", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    classes = [d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))]
    print(f"Found {len(classes)} classes")

    for cls in classes:
        cls_dir = os.path.join(args.src, cls)
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_val = int(len(images) * args.val_ratio)
        val_imgs, train_imgs = images[:n_val], images[n_val:]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = os.path.join(args.dst, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(os.path.join(cls_dir, img), os.path.join(out_dir, img))

        print(f"  {cls}: {len(train_imgs)} train / {len(val_imgs)} val")

    print(f"Done. Split written to {args.dst}/train and {args.dst}/val")


if __name__ == "__main__":
    main()
