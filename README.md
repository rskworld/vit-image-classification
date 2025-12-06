# Vision Transformer (ViT) for Image Classification

<!--
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: Vision Transformer architecture for image classification using patch-based embeddings and self-attention mechanisms
-->

## Description

This project implements Vision Transformer (ViT) for image classification. Unlike CNNs, ViT splits images into patches and processes them as sequences using transformer architecture. It includes patch embedding, positional encoding, and multi-head self-attention layers, achieving state-of-the-art results on image classification tasks.

## Features

### Core Features
- Vision Transformer (ViT) architecture
- Patch-based image embedding
- Multi-head self-attention mechanism
- Positional encoding for spatial information
- State-of-the-art classification accuracy

### Additional Features
- **Data Preparation**: Automated data splitting and organization
- **Advanced Augmentation**: MixUp, CutMix, Random Erasing, AutoAugment
- **Visualization Tools**: Attention maps, patch visualization, training curves
- **Evaluation Metrics**: Comprehensive evaluation with confusion matrix, per-class accuracy
- **Model Comparison**: Compare different ViT configurations
- **Testing Suite**: Unit tests for all model components
- **Easy Setup**: Automated environment setup script

## Technologies

- Python
- PyTorch
- TensorFlow
- Vision Transformer (ViT)
- Patch Embedding
- Self-Attention
- Transformers
- Jupyter Notebook

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config config.yaml
```

### Inference

```bash
python inference.py --model_path models/vit_model.pth --image_path path/to/image.jpg
```

### Jupyter Notebook

Open `vit_image_classification.ipynb` for interactive exploration.

### Data Preparation

```bash
# Create directory structure
python data_preparation.py --action create_structure

# Split data into train/val/test
python data_preparation.py --action split --source_dir your_data --target_dir data

# Get dataset statistics
python data_preparation.py --action stats --target_dir data/train
```

### Evaluation

```bash
python evaluate.py --model_path models/best_model.pth --config config.yaml --plot
```

### Model Comparison

```bash
python model_comparison.py
```

### Testing

```bash
python test_model.py
```

### Setup

```bash
python setup.py
```

## Project Structure

```
vit-image-classification/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore file
├── DATA_GUIDE.md                      # Data preparation guide
├── PROJECT_INFO.md                    # Project information
│
├── Core Model Files
├── vit_model.py                       # ViT model implementation
├── train.py                           # Training script
├── inference.py                       # Inference script
├── utils.py                           # Utility functions
│
├── Data & Augmentation
├── data_preparation.py                # Data preparation utilities
├── augmentation.py                    # Advanced augmentation
│
├── Evaluation & Analysis
├── evaluate.py                         # Evaluation metrics
├── model_comparison.py                # Model comparison tools
├── visualization.py                   # Visualization tools
│
├── Testing & Setup
├── test_model.py                      # Unit tests
├── setup.py                           # Setup script
├── example_usage.py                    # Example usage
│
├── Notebooks
├── vit_image_classification.ipynb     # Jupyter notebook
│
└── Directories (created during setup)
    ├── data/                          # Training data
    ├── models/                        # Saved models
    ├── logs/                          # Training logs
    └── evaluation_results/            # Evaluation outputs
```

## Author

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

## License

This project is provided as-is for educational and research purposes.

