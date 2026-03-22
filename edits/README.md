# Architectural Modifications (Ultralytics Source Edits)

This folder contains the core architectural contributions of our paper: the **Ghost Convolution**, **CBAM (Convolutional Block Attention Module)**, and the **BiFPN** feature fusion neck. 

Because we built upon the official YOLOv11 framework, these custom modules must be injected directly into the installed `ultralytics` Python package to allow the network parser to build our proposed model from the YAML configuration.

## File Mapping
You will need to replace four default Ultralytics files with the modified versions provided in this folder. 

| Modified File | Target Directory in `ultralytics` Package | Purpose of Edit |
| :--- | :--- | :--- |
| `conv.py` | `ultralytics/nn/modules/conv.py` | Injects the `GhostConv` class definitions. |
| `block.py` | `ultralytics/nn/modules/block.py` | Injects the `CBAM` and `BiFPN_Concat` class definitions. |
| `__init__.py` | `ultralytics/nn/modules/__init__.py` | Exposes the new classes so the model parser can import them. |
| `tasks.py` | `ultralytics/nn/tasks.py` | Modifies `parse_model()` to recognize our custom module names in the `.yaml` file. |

---


