"""
TrashNet - Train / Val / Test Split
=====================================
data/raw/dataset-resized/ altindaki 6 sinifi okur,
%70 train / %15 val / %15 test olarak data/split/ klasorune kopyalar.
"""

import os
import shutil
import random
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RAW_DIR   = os.path.join("data", "raw", "dataset-resized")
SPLIT_DIR = os.path.join("data", "split")

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def prepare_split():
    random.seed(SEED)

    stats = defaultdict(dict)

    print("=" * 55)
    print("  TrashNet - Train/Val/Test Split Hazirlaniyor")
    print("=" * 55)

    for cls in CLASSES:
        src_cls = os.path.join(RAW_DIR, cls)
        if not os.path.exists(src_cls):
            print(f"  UYARI: {src_cls} bulunamadi, atlaniyor.")
            continue

        # Goruntu dosyalarini listele
        files = [f for f in os.listdir(src_cls)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            dst_dir = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in split_files:
                shutil.copy2(
                    os.path.join(src_cls, fname),
                    os.path.join(dst_dir, fname)
                )
            stats[cls][split_name] = len(split_files)

    # Rapor
    print(f"\n  {'Sinif':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Toplam':>8}")
    print("  " + "-" * 47)
    total_train = total_val = total_test = 0
    for cls in CLASSES:
        if cls not in stats:
            continue
        tr = stats[cls]["train"]
        va = stats[cls]["val"]
        te = stats[cls]["test"]
        total_train += tr
        total_val   += va
        total_test  += te
        print(f"  {cls:<12} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}")

    print("  " + "-" * 47)
    total = total_train + total_val + total_test
    print(f"  {'TOPLAM':<12} {total_train:>8} {total_val:>8} {total_test:>8} {total:>8}")
    print(f"\n  Oranlar - Train: %{total_train/total*100:.1f}  "
          f"Val: %{total_val/total*100:.1f}  "
          f"Test: %{total_test/total*100:.1f}")
    print("\n  Split tamamlandi!")
    print(f"  Konum: {os.path.abspath(SPLIT_DIR)}")
    print("=" * 55)
    print("\nSonraki adim: python src/train.py")


if __name__ == "__main__":
    prepare_split()
