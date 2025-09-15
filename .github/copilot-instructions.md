# Copilot Instructions for GPS_research

This codebase integrates GPS data with computer vision models (YOLOv8/YOLOv10) for enhanced detection and analysis. It also includes Hugging Face Transformers components for NLP and vision tasks. Follow these guidelines to be productive as an AI coding agent:

## Architecture Overview
- **GPS_v10/**: Core directory for GPS-enhanced YOLO training, validation, and prediction. Key files:
  - `true_gps_training_v4.2.py`, `true_gps_training_v4.py`, `true_gps_training_v3.py`, `true_gps_training_v2.py`: Main scripts for training/validation/prediction with GPS fusion. Each version refines the GPS encoding and fusion logic.
  - `val.py`: Standalone validation script for YOLO models.
  - `data_gps.yaml`, `data.yaml`: Dataset configuration files (class names, paths).
- **dataset/**: Contains images, labels, and GPS CSVs. Used for training and validation.
- **交接/Seg/transformers/**: Hugging Face Transformers library and examples. See `AGENTS.md` for agent-specific conventions.

## Key Patterns & Conventions
- **GPS Feature Fusion**: GPS coordinates are encoded (see `ImprovedGPSEncoder`) and fused into YOLO feature maps via hooks (`AdaptiveGPSFeatureFusion`). Hooks are registered on specific layers (see `get_yolov10_fusion_layers`).
- **Data Management**: Use `GPSDataManager` to load GPS CSVs and compute statistics for normalization. Always pass correct CSV paths via CLI args.
- **Training/Validation/Prediction**:
  - Use CLI arguments (`--data`, `--gps-train-csv`, `--gps-val-csv`, `--model`, etc.) to control behavior.
  - For prediction, batch mode is supported but single-image is recommended for best GPS fusion.
- **Results Output**: Validation results are saved as CSVs with per-class metrics. See `val.py` and main scripts for output logic.
- **YOLO Integration**: Models are loaded via `YOLO(model_path)`. Hooks are added for GPS fusion. Use `ultralytics` package.

## Developer Workflows
- **Install dependencies**: `pip install -r requirements.txt` (if present), plus `pip install ultralytics torch pandas`.
- **Run training**: Example:
  ```bash
  python GPS_v10/true_gps_training_v4.2.py --data GPS_v10/data_gps.yaml --gps-train-csv dataset/Alldata/gps_train.csv --gps-val-csv dataset/Alldata/gps_val.csv --model yolov10n.pt --epochs 100 --batch-size 16
  ```
- **Run validation**:
  ```bash
  python GPS_v10/val.py
  ```
- **Transformers workflows**: See `交接/Seg/transformers/AGENTS.md` and example scripts for NLP/vision tasks. Use `make fixup` to sync modular/modeling files if editing core library code.

## Project-Specific Conventions
- **Copied from** comments in Transformers code indicate auto-synced code blocks. Edit base functions and run `make fixup` to propagate changes.
- **Modular files**: Only edit modular files, not generated modeling files. Use `make fixup` after changes.
- **GPS fusion hooks**: Always ensure hooks are registered before training/validation/prediction. See `_register_hooks()` in trainer classes.
- **Class names and dataset paths**: Always check YAML config for correct class mappings and image paths.

## External Dependencies & Integration
- **ultralytics**: For YOLO models and training.
- **torch**: For all deep learning components.
- **pandas**: For CSV data management.
- **Hugging Face Transformers**: For NLP/vision tasks in `交接/Seg/transformers/`.

## References
- See `GPS_v10/true_gps_training_v4.2.py` for the latest GPS fusion logic.
- See `交接/Seg/transformers/AGENTS.md` for agent conventions in Transformers code.
- See `GPS_v10/val.py` for validation and result output patterns.

---

If any section is unclear or missing, please provide feedback for further refinement.