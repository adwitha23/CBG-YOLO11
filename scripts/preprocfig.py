import os
import glob
import shutil
import yaml
import zipfile
import numpy as np
from sklearn.model_selection import KFold, train_test_split


# ==============================================================================
# PATHS - Update these before running
# ==============================================================================
HOME_DIR      = "/path/to/home/dir"
ZIP_PATH      = os.path.join(HOME_DIR, "figshare.zip")
EXTRACT_PATH  = HOME_DIR
DATASET_ROOT  = os.path.join(EXTRACT_PATH, "Figshare")
FOLDS_DIR     = os.path.join(DATASET_ROOT, "5_folds")
TEST_SET_DIR  = os.path.join(HOME_DIR, "figshare_final_test_set")


os.makedirs(HOME_DIR, exist_ok=True)
os.chdir(HOME_DIR)


# ==============================================================================
# 1. DATA PREPARATION & HOLD-OUT (15%)
# ==============================================================================
if os.path.exists(ZIP_PATH) and not os.path.exists(DATASET_ROOT):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

def find_label_path(image_path):
    lbl_path = image_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        if lbl_path.lower().endswith(ext):
            return lbl_path[:-len(ext)] + '.txt'
    return lbl_path

all_images = []
for ext in ["**/*.jpg", "**/*.jpeg", "**/*.png"]:
    all_images.extend(glob.glob(os.path.join(DATASET_ROOT, ext), recursive=True))

valid_images = np.array([img for img in all_images if os.path.exists(find_label_path(img))])

X_cv, X_test = train_test_split(valid_images, test_size=0.15, random_state=42)

if os.path.exists(TEST_SET_DIR): shutil.rmtree(TEST_SET_DIR)
os.makedirs(os.path.join(TEST_SET_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(TEST_SET_DIR, "labels"), exist_ok=True)

for img in X_test:
    shutil.copy(img, os.path.join(TEST_SET_DIR, "images"))
    shutil.copy(find_label_path(img), os.path.join(TEST_SET_DIR, "labels"))


# ==============================================================================
# 2. GENERATE K-FOLD YAMLs (Using .txt list method)
# ==============================================================================
os.makedirs(FOLDS_DIR, exist_ok=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
    fold_num = fold + 1
    train_paths = X_cv[train_idx]
    val_paths = X_cv[val_idx]

    train_txt = os.path.join(FOLDS_DIR, f"train_fold_{fold_num}.txt")
    val_txt = os.path.join(FOLDS_DIR, f"val_fold_{fold_num}.txt")

    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_paths))
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_paths))

    yaml_dict = {
        'path': DATASET_ROOT,
        'train': train_txt,
        'val': val_txt,
        'test': os.path.join(TEST_SET_DIR, "images"),
        'nc': 1,
        'names': ['Tumor']
    }

    yaml_path = os.path.join(FOLDS_DIR, f"data_fold_{fold_num}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
