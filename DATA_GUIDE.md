# Data Preparation Guide

<!--
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Data Directory Structure

The Vision Transformer project expects data to be organized in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/ (optional)
    ├── class1/
    └── ...
```

## Data Preparation Tools

### 1. Create Directory Structure

```bash
python data_preparation.py --action create_structure --target_dir data
```

### 2. Split Data into Train/Val/Test

If you have all your data in one directory organized by classes:

```bash
python data_preparation.py --action split \
    --source_dir path/to/your/data \
    --target_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 3. Organize Data by Classes

If your images are in a flat directory:

```bash
python data_preparation.py --action organize \
    --source_dir path/to/flat/directory \
    --target_dir data/organized
```

### 4. Get Dataset Statistics

```bash
python data_preparation.py --action stats --target_dir data/train
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Data Requirements

- **Minimum images per class**: 10-20 (more is better)
- **Recommended images per class**: 100+
- **Image size**: Any size (will be resized to 224x224 by default)
- **Color format**: RGB (color images)

## Data Augmentation

The project includes various augmentation strategies:

- **Standard**: Basic augmentations (crop, flip, color jitter)
- **Aggressive**: Heavy augmentations for small datasets
- **Light**: Minimal augmentations
- **AutoAugment**: Learned augmentation policies

Configure in `config.yaml` or use `augmentation.py` for custom pipelines.

## Example Workflow

1. **Collect your images** in a directory structure
2. **Organize by classes** if needed:
   ```bash
   python data_preparation.py --action organize --source_dir raw_data --target_dir data/organized
   ```
3. **Split into train/val/test**:
   ```bash
   python data_preparation.py --action split --source_dir data/organized --target_dir data
   ```
4. **Check statistics**:
   ```bash
   python data_preparation.py --action stats --target_dir data/train
   ```
5. **Update config.yaml** with your number of classes
6. **Start training**:
   ```bash
   python train.py --config config.yaml
   ```

## Tips

- Ensure balanced classes (similar number of images per class)
- Use validation set to monitor overfitting
- Test set should be separate and only used for final evaluation
- For small datasets, use aggressive augmentation
- For large datasets, standard augmentation is usually sufficient

## Author

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

