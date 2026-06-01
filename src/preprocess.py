import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

IMG_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


for split in ["train", "val", "test"]:
    (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)


classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]

print(f"Found classes: {classes}")

for class_name in classes:
    class_path = RAW_DIR / class_name

    images = [
        img for img in class_path.iterdir()
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"{class_name}: {len(images)} images")

    # -------- SPLIT DATA --------
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=42
    )

    split_map = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }


    for split, img_list in split_map.items():
        out_dir = PROCESSED_DIR / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                save_path = out_dir / img_path.name
                img.save(save_path)

            except Exception as e:
                print(f"Skipped {img_path.name}: {e}")

print("\nPreprocessing complete.")
