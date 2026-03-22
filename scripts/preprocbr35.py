import os
import json
import glob
import yaml
import numpy as np
import cv2
from sklearn.model_selection import KFold


# ==========================================
# PATHS - Update these before running
# ==========================================
RAW_DATASET_PATH = "/path/to/raw/dataset"       # e.g. Br35H-Mask-RCNN root
OUT_DATASET_PATH = "/path/to/output/dataset"    # e.g. br35h_clahe_detect


# ==========================================
# 1. CREATE DIRECTORY STRUCTURE
# ==========================================
for split in ['TRAIN', 'VAL', 'TEST']:
    os.makedirs(os.path.join(OUT_DATASET_PATH, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DATASET_PATH, 'labels', split), exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ==========================================
# 2. CONVERT JSON ANNOTATIONS TO YOLO FORMAT + APPLY CLAHE
# ==========================================
json_path = os.path.join(RAW_DATASET_PATH, "annotations_all.json")
with open(json_path, 'r') as f:
    via_data = json.load(f)

for img_key, img_info in via_data.items():
    img_name = img_info['filename']
    regions = img_info.get('regions', [])

    img_path_local = None
    target_split = None
    for split in ['TRAIN', 'VAL', 'TEST']:
        temp_path = os.path.join(RAW_DATASET_PATH, split, img_name)
        if os.path.exists(temp_path):
            img_path_local = temp_path
            target_split = split
            break

    if img_path_local is None:
        continue

    img = cv2.imread(img_path_local)
    if img is None:
        continue

    img_h, img_w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    dst_img_path = os.path.join(OUT_DATASET_PATH, 'images', target_split, img_name)
    cv2.imwrite(dst_img_path, final_img)

    txt_path = os.path.join(OUT_DATASET_PATH, 'labels', target_split, img_name.rsplit('.', 1)[0] + '.txt')

    with open(txt_path, 'w') as txt_file:
        for region in regions:
            shape_attr = region['shape_attributes'] if isinstance(region, dict) else region[0]['shape_attributes']

            if shape_attr['name'] == 'polygon':
                x_points = shape_attr['all_points_x']
                y_points = shape_attr['all_points_y']

                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)

                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                norm_w = (x_max - x_min) / img_w
                norm_h = (y_max - y_min) / img_h

                txt_file.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")


# ==========================================
# 3. GENERATE data.yaml
# ==========================================
data_yaml = {
    'path': OUT_DATASET_PATH,
    'train': 'images/TRAIN',
    'val': 'images/VAL',
    'test': 'images/TEST',
    'task': 'detect',
    'nc': 1,
    'names': ['Tumor']
}

yaml_path = os.path.join(OUT_DATASET_PATH, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, sort_keys=False)


# ==========================================
# 4. GENERATE 5-FOLD CROSS VALIDATION SPLITS
# ==========================================
FOLDS_DIR = os.path.join(OUT_DATASET_PATH, "5_folds")
os.makedirs(FOLDS_DIR, exist_ok=True)

train_imgs = glob.glob(os.path.join(OUT_DATASET_PATH, "images", "TRAIN", "*.jpg"))
val_imgs = glob.glob(os.path.join(OUT_DATASET_PATH, "images", "VAL", "*.jpg"))

cv_pool = np.array(train_imgs + val_imgs)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(cv_pool)):
    fold_num = fold + 1

    train_paths = cv_pool[train_idx]
    val_paths = cv_pool[val_idx]

    train_txt = os.path.join(FOLDS_DIR, f"train_fold_{fold_num}.txt")
    val_txt = os.path.join(FOLDS_DIR, f"val_fold_{fold_num}.txt")

    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_paths))
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_paths))

    yaml_dict = {
        'path': OUT_DATASET_PATH,
        'train': train_txt,
        'val': val_txt,
        'test': 'images/TEST',
        'task': 'detect',
        'nc': 1,
        'names': ['Tumor']
    }

    fold_yaml_path = os.path.join(FOLDS_DIR, f"data_fold_{fold_num}.yaml")
    with open(fold_yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
