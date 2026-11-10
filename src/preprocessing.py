"""
preprocessing.py
----------------
Everything that happens BEFORE feature extraction:

  1. Find the dataset folder (works even if it is nested one or two levels deep)
  2. Build a "manifest" table: one row per image (path, fruit, label, hash ...)
  3. Verify the dataset: counts, unreadable images, duplicate images
  4. Load an image safely and resize it to a fixed size
  5. Split into train / validation / test WITHOUT data leakage

The original images are only ever READ. Nothing is moved, edited or deleted.

Run a full dataset check + create the split file with:
    python -m src.preprocessing
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Settings (change here, nowhere else)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (128, 128)          # every image is resized to 128 x 128
FRUITS = ["Apple", "Banana", "Strawberry"]
CLASS_NAMES = ["Fresh", "Rotten"]   # index 0 = Fresh, index 1 = Rotten
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15                  # 70% train / 15% validation / 15% test


# ----------------------------------------------------------------------------
# 1. Find the dataset root
# ----------------------------------------------------------------------------
def find_dataset_root(base=DATASET_DIR):
    """
    Search `base` for the folder that directly contains Apple/Banana/Strawberry.
    Your layout is  dataset/Fruit Freshness Dataset/Apple/...  so the answer is
    the 'Fruit Freshness Dataset' folder. Folder names are matched ignoring
    upper/lower case.
    """
    base = Path(base)
    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    candidates = [base] + [p for p in base.rglob("*") if p.is_dir()]
    for folder in candidates:
        names = {p.name.lower() for p in folder.iterdir() if p.is_dir()}
        if all(f.lower() in names for f in FRUITS):
            return folder
    raise FileNotFoundError(
        f"Could not find Apple/Banana/Strawberry folders anywhere inside {base}"
    )


# ----------------------------------------------------------------------------
# 2. Build the manifest (one row per image)
# ----------------------------------------------------------------------------
def _md5(path):
    """Fingerprint of the file's bytes. Identical files -> identical hash."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build_manifest(root=None):
    """Return a DataFrame describing every image file in the dataset."""
    root = Path(root) if root else find_dataset_root()
    rows = []
    for fruit_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in fruit_dir.iterdir() if p.is_dir()):
            label = label_dir.name.capitalize()          # 'fresh' -> 'Fresh'
            if label not in CLASS_NAMES:
                continue
            for f in sorted(label_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "path": str(f),
                        "fruit": fruit_dir.name.capitalize(),
                        "label": label,
                        "label_id": CLASS_NAMES.index(label),
                        "md5": _md5(f),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found. Check the dataset path.")
    return df


# ----------------------------------------------------------------------------
# 3. Image loading
# ----------------------------------------------------------------------------
def load_image(source, size=IMAGE_SIZE):
    """
    Load an image from a file path OR an uploaded file object (Streamlit) and
    return a uint8 RGB array of shape (size[1], size[0], 3).

    Why Pillow? It reads JPG, PNG, WEBP and JFIF reliably on Windows, and it
    handles file objects (needed for the Streamlit upload).
    Why convert to RGB? Some images are RGBA (transparent) or grayscale, but
    every feature extractor expects exactly 3 colour channels.
    """
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)        # fix phone-photo rotation

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)   # transparent -> white

    img = img.convert("RGB")
    arr = np.asarray(img)
    # INTER_AREA is the best OpenCV interpolation for shrinking images
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# 4. Dataset verification
# ----------------------------------------------------------------------------
def verify_dataset(df):
    """
    Print counts, check every image can be opened, detect duplicate files.
    Returns the manifest with extra columns 'readable' and 'is_duplicate'.
    """
    df = df.copy()

    # --- unreadable / corrupted images -------------------------------------
    readable = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()                       # checks file integrity
            load_image(p)                         # checks it really decodes
            readable.append(True)
        except Exception:
            readable.append(False)
    df["readable"] = readable

    # --- exact duplicates ---------------------------------------------------
    # Keep the first file of each hash as the "original"; the rest are copies.
    df["is_duplicate"] = df.duplicated(subset="md5", keep="first")

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print("\nImage counts found on disk (Fruit x Class):")
    print(pd.crosstab(df["fruit"], df["label"], margins=True, margins_name="Total"))

    n_bad = int((~df["readable"]).sum())
    print(f"\nUnreadable / corrupted images : {n_bad}")
    for p in df.loc[~df["readable"], "path"]:
        print("   BAD:", p)

    n_dup = int(df["is_duplicate"].sum())
    print(f"Exact duplicate files          : {n_dup}  (identical bytes to another image)")

    # A hash that appears under BOTH Fresh and Rotten would be a labelling conflict
    conflict = df.groupby("md5")["label"].nunique()
    print(f"Duplicates with conflicting labels: {int((conflict > 1).sum())}")

    usable = df[df["readable"] & ~df["is_duplicate"]]
    print(f"\nUnique, readable images used for the model: {len(usable)}")
    print(pd.crosstab(usable["fruit"], usable["label"], margins=True, margins_name="Total"))
    print("\n(The duplicate files are NOT deleted from disk; the model simply "
          "ignores the extra copies.)")
    return df


# ----------------------------------------------------------------------------
# 5. Leak-free stratified split
# ----------------------------------------------------------------------------
def make_splits(df, seed=RANDOM_SEED):
    """
    Split into train / val / test.

    * Only unique, readable images are used, so a copy of a training image can
      never appear in the test set (that would be data leakage).
    * Stratified on fruit+label together (6 groups), so every split has the
      same Fresh/Rotten ratio inside every fruit type.
    Returns the manifest with a new column 'split'.
    """
    data = df[df["readable"] & ~df["is_duplicate"]].copy()
    data["stratum"] = data["fruit"] + "_" + data["label"]

    # First carve out the test set, then split the rest into train / val
    trainval, test = train_test_split(
        data, test_size=TEST_SIZE, stratify=data["stratum"], random_state=seed)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        trainval, test_size=val_fraction, stratify=trainval["stratum"],
        random_state=seed)

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    out = pd.concat([train, val, test]).drop(columns="stratum")

    # Safety checks: no image (by content hash) may appear in two splits
    sets = {s: set(out.loc[out["split"] == s, "md5"]) for s in ["train", "val", "test"]}
    assert not (sets["train"] & sets["val"]), "LEAK: train/val overlap"
    assert not (sets["train"] & sets["test"]), "LEAK: train/test overlap"
    assert not (sets["val"] & sets["test"]), "LEAK: val/test overlap"
    return out.reset_index(drop=True)


def save_splits(splits, path=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = Path(path) if path else OUTPUT_DIR / "splits.csv"
    splits.to_csv(path, index=False)
    return path


def load_splits(path=None):
    path = Path(path) if path else OUTPUT_DIR / "splits.csv"
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# Run as a script:  python -m src.preprocessing
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    root = find_dataset_root()
    print("Dataset root detected:", root)

    manifest = verify_dataset(build_manifest(root))
    splits = make_splits(manifest)

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT  (no leakage - verified by assertions)")
    print("=" * 60)
    print(pd.crosstab([splits["fruit"], splits["label"]], splits["split"],
                      margins=True, margins_name="Total")[["train", "val", "test", "Total"]])
    print("\nSaved split file to:", save_splits(splits))