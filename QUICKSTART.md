# Quick Start Guide

<!--
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Installation

1. **Clone or download the project**

2. **Run setup script**:
   ```bash
   python setup.py
   ```
   This will:
   - Check Python version
   - Install all dependencies
   - Create necessary directories
   - Verify installation

3. **Or install manually**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Test

Test that everything is working:

```bash
# Test model components
python test_model.py

# Run example usage
python example_usage.py
```

## Prepare Your Data

1. **Organize your images** by class in folders
2. **Split into train/val/test**:
   ```bash
   python data_preparation.py --action split \
       --source_dir your_data \
       --target_dir data \
       --train_ratio 0.7 \
       --val_ratio 0.15 \
       --test_ratio 0.15
   ```
3. **Check your data**:
   ```bash
   python data_preparation.py --action stats --target_dir data/train
   ```

## Configure

Edit `config.yaml`:
- Update `num_classes` to match your dataset
- Adjust `image_size` and `patch_size` if needed
- Set training parameters (batch_size, epochs, learning_rate)

## Train

```bash
python train.py --config config.yaml
```

Monitor training:
- Check `logs/` directory for TensorBoard logs
- View with: `tensorboard --logdir logs`

## Evaluate

```bash
python evaluate.py \
    --model_path models/best_model.pth \
    --config config.yaml \
    --plot
```

## Inference

```bash
python inference.py \
    --model_path models/best_model.pth \
    --image_path path/to/image.jpg \
    --config config.yaml
```

## Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook vit_image_classification.ipynb
```

## Common Tasks

### Compare Different Models

```bash
python model_comparison.py
```

### Visualize Augmentations

```python
from augmentation import visualize_augmentations
visualize_augmentations('path/to/image.jpg', 'aggressive')
```

### Visualize Patches

```python
from visualization import visualize_patches
visualize_patches('path/to/image.jpg', patch_size=16, image_size=224)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Use smaller model (reduce `dim`, `depth`, `heads`)

### Slow Training
- Enable `pin_memory` in config
- Increase `num_workers` (if you have multiple CPU cores)
- Use GPU if available

### Poor Accuracy
- Check data quality and balance
- Try different augmentation strategies
- Adjust learning rate
- Train for more epochs

## Next Steps

- Read `DATA_GUIDE.md` for detailed data preparation
- Check `PROJECT_INFO.md` for project overview
- Explore `vit_image_classification.ipynb` for examples

## Author

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

