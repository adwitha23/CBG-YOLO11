# Brain Tumor Detection using Custom YOLO11 Architectures

This is the source code for brain tumor detection using modified YOLO11 architectures with GhostNet convolutions, CBAM attention, and BiFPN feature aggregation, validated on two datasets using 5-fold cross-validation. The Figshare training additionally applies L1 regularization via a custom callback to promote weight sparsity given the larger dataset size.

## Instructions

After cloning the repository, the custom module files in `edits/` must be copied into your Ultralytics installation before running any training. The scripts are organized as follows: preprocessing and fold generation are handled separately from training, so run the preprocessing script first and then the training script.

**Clone the repository**

```
git clone https://github.com/adwitha23/CBG-YOLO11.git
cd CBG-YOLO11
```

**Create and activate a virtual environment**

```
python3 -m venv venv
source venv/bin/activate
```

**Install dependencies**

```
pip install ultralytics scikit-learn opencv-python pyyaml numpy torch
```

**Copy architecture YAMLs into Ultralytics**

```
cp models/11/ghost_cbam_bifpn_yolo11s.yaml venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/
cp models/11/ghost_cbam_bifpn3x_yolo11s.yaml venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/
cp models/11/ghost_yolo11s.yaml venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/
```

**Copy custom module files into Ultralytics**

This step is required. The custom layers (CbamBiFPN_Neck, GhostConv, CBAM, BiFPN) are defined in the `edits/` folder and must replace the corresponding stock Ultralytics files.

```
cp edits/block.py venv/lib/python3.12/site-packages/ultralytics/nn/modules/block.py
cp edits/tasks.py venv/lib/python3.12/site-packages/ultralytics/nn/tasks.py
cp edits/__init__.py venv/lib/python3.12/site-packages/ultralytics/nn/modules/__init__.py
```

**Download pretrained weights**

```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt -P /path/to/your/working/directory
```

Every time you open a new terminal session, reactivate the virtual environment before running any scripts:

```
source venv/bin/activate
```

## For the First Dataset (Br35H)

Download the Br35H dataset from IEEEDataPort.

**Dataset Preparation**

Unzip the downloaded file. The dataset comes with VIA polygon annotations in a single JSON file. Ensure the raw folder structure looks like this:

```
Br35H-Mask-RCNN/
├── TRAIN/
├── VAL/
├── TEST/
└── annotations_all.json
```

**Preprocessing**

Open `scripts/preprocbr35.py` and set the following at the top of the file:

```python
RAW_DATASET_PATH = "/path/to/Br35H-Mask-RCNN"
OUT_DATASET_PATH = "/path/to/br35h_clahe_detect"
```

Run the script:

```
python scripts/preprocbr35.py
```

This will create the output dataset directory, apply CLAHE enhancement to all images, convert polygon annotations to YOLO bounding box format, generate a `data.yaml`, and produce the 5-fold split files under `br35h_clahe_detect/5_folds/`. The final output structure will be:

```
br35h_clahe_detect/
├── images/
│   ├── TRAIN/
│   ├── VAL/
│   └── TEST/
├── labels/
│   ├── TRAIN/
│   ├── VAL/
│   └── TEST/
├── 5_folds/
│   ├── train_fold_1.txt ... train_fold_5.txt
│   ├── val_fold_1.txt   ... val_fold_5.txt
│   └── data_fold_1.yaml ... data_fold_5.yaml
└── data.yaml
```

**Training**

Open `scripts/br35_5fold.py` and set the following at the top of the file:

```python
DATASET_ROOT       = "/path/to/br35h_clahe_detect"
ARCHITECTURE_YAML  = "/path/to/temp/models/11/ghost_cbam_bifpn_yolo11s.yaml"
PRETRAINED_WEIGHTS = "/path/to/yolo11s.pt"
RUN_NAME_PREFIX    = "your_run_name"
```

Run the script:

```
python scripts/br35_5fold.py
```

This will train for 5 folds sequentially. Each fold initializes a fresh model and injects matched pretrained weights before training. After each fold, the best checkpoint is evaluated on the hold-out TEST set using test-time augmentation and the results are printed to the console.

## For the Second Dataset (Figshare)

Download the Figshare brain tumor MRI dataset from IEEEDataPort.

**Dataset Preparation**

Unzip the downloaded file. The dataset should already have images and YOLO-format label files. Ensure the folder is structured with images and their matching `.txt` label files accessible recursively. Rename the folder to `Figshare` if it is not already named that.

**Preprocessing**

Open `scripts/preprocfig.py` and set the following at the top of the file:

```python
HOME_DIR = "/path/to/working/directory"
```

The script expects `figshare.zip` inside `HOME_DIR`, or an already-extracted `Figshare` folder at `HOME_DIR/Figshare`. If the zip is present and the folder does not exist yet, it will be extracted automatically.

Run the script:

```
python scripts/preprocfig.py
```

This will validate all image-label pairs, extract a 15% stratified hold-out test set into a separate `figshare_final_test_set/` directory, and generate the 5-fold split files under `Figshare/5_folds/`. The hold-out test set remains fixed across all folds.

**Training**

Open `scripts/figshare_5fold.py` and set the following at the top of the file:

```python
DATASET_ROOT       = "/path/to/Figshare"
ARCHITECTURE_YAML  = "/path/to/temp/models/11/ghost_cbam_bifpn_yolo11s.yaml"
PRETRAINED_WEIGHTS = "/path/to/yolo11s.pt"
RUN_NAME_PREFIX    = "your_run_name"
```

Run the script:

```
python scripts/figshare_5fold.py
```

Training runs for 5 folds sequentially.

## Dependencies

```
pip install ultralytics scikit-learn opencv-python pyyaml numpy torch
```

CUDA is recommended. Adjust batch size according to available VRAM.

## Notes

- Always activate the virtual environment before running any script (`source venv/bin/activate`).
- The `edits/` copy step is mandatory. Without it, training will fail with a `KeyError` on the custom layer names.
- Do not run the training script before the preprocessing script. The training script reads the fold YAMLs generated by the preprocessing script.
- Each fold re-initializes the model from scratch to prevent weight leakage between folds.
- Final performance should be reported as the mean across all 5 folds on the hold-out test set.

It is a publicly available code and free to use.
