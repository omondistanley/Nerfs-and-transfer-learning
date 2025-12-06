# Project 1: Equivariant Neural Rendering with Transfer Learning

## Project Overview

This project implements and extends **Equivariant Neural Rendering**, a deep learning approach for novel view synthesis from single images. The system learns to infer 3D scene representations from 2D images and can render novel viewpoints by rotating these representations. The project focuses on transfer learning techniques, specifically transferring knowledge from a model trained on mugs to a bowls dataset.

### Key Accomplishments

- **Transfer Learning**: Successfully transferred a pre-trained model from mugs dataset to bowls dataset
- **Progressive Freezing**: Implemented and experimented with progressive freezing strategies for fine-tuning
- **Enhanced Training**: Added perceptual loss (VGG-based) and Exponential Moving Average (EMA) for improved training stability
- **Multiple Experiments**: Conducted extensive ablation studies on freezing strategies, loss functions, and learning rates
- **Novel View Synthesis**: Generated high-quality novel view animations for chairs, mugs, and bowls

## Codebase Structure

```
proj1/
├── README.md                          
├── Project 1 Write up.pdf             # Detailed project analysis and results
├── animations.gif                      
├── ml-equivariant-neural-rendering-main/
│   ├── README.md                     
│   ├── experiments.py                
│   ├── training.py                    
│   ├── testing.py                     
│   ├── testingmugs.py               
│   ├── evaluate_psnr.py              
│   ├── exploration.ipynb              # Jupyter notebook for visualization
│   ├── config.json                    # Main configuration file
│   ├── config1.json                   # Transfer learning config
│   ├── config2.json                   # Progressive freezing config
│   ├── config3.json                   # Fine-tuning config
│   ├── mugs.gif                      
│   ├── chair.gif                     
│   ├── bowl_transfer*.gif            # Transfer learning results 
│   ├── models/                      
│   ├── training/                  
│   ├── misc/                         
│   └── transforms3d/              
├── pre-trained_models/                
│   ├── mugs.pt
│   ├── chairs.pt
│   ├── cars.pt
│   └── ...
└── output/                         
    └── transfer1.gif
```

## Key Modifications

### Enhanced Training Class (`training.py`)

The base training code was significantly enhanced with the following features:

#### 1. **Progressive Freezing Support**
- Implements a multi-phase training strategy where different parts of the model are frozen/unfrozen at different epochs
- Allows gradual fine-tuning of transfer learning models
- Configurable via `progressive_freezing` in config files

```python
"progressive_freezing": {
    "phase_1": {
        "epochs": [1, 5],
        "freeze_parts": ["inv_transform_3d", "inv_projection", "inv_transform_2d"],
        "lr": 5e-5
    },
    "phase_2": {
        "epochs": [6, 10],
        "freeze_parts": ["inv_transform_3d", "inv_projection"],
        "lr": 5e-5
    },
    # ... more phases
}
```

#### 2. **Perceptual Loss (VGG-based)**
- Added VGG19-based perceptual loss to improve visual quality
- Extracts features from multiple layers (default: layers 2, 7, 12)
- Configurable weight via `perceptual_loss_weight` in config

#### 3. **Exponential Moving Average (EMA)**
- Maintains shadow weights of model parameters using EMA
- Helps stabilize training and improve final model quality
- Enabled via `use_ema: true` in config

#### 4. **Configurable Model Freezing**
- Supports freezing specific model parts during transfer learning
- Uses substring matching for flexible part selection
- Configurable via `freeze_parts` list in config

#### 5. **Fine-tuning Learning Rate**
- Separate learning rate for fine-tuning (`finetune_lr`)
- Allows different learning rates for initial training vs. fine-tuning

### Enhanced Experiments Script (`experiments.py`)

- Added support for loading pre-trained models via `pretrained_model_path`
- Automatic model freezing based on config
- Support for progressive freezing schedules
- Enhanced logging of trainable parameters

## Experiments

### 1. Mugs Training
- **Dataset**: MugsHQ dataset
- **Purpose**: Initial training to create a base model for transfer learning
- **Config**: Standard training configuration
- **Output**: Pre-trained model saved in `2025-10-08_03-19_mugs-experiment/`

### 2. Bowls Transfer Learning

Multiple transfer learning experiments were conducted, transferring from the mugs pre-trained model:

#### a) **Baseline Transfer** (`bowls-original-experiment`)
- Direct transfer from mugs model
- All layers trainable
- Baseline for comparison

#### b) **Progressive Freezing** (`transfer-bowls-experiment-progressive-freeze`)
- Multi-phase training strategy:
  - **Phase 1 (Epochs 1-5)**: Freeze inverse 3D transform, inverse projection, and inverse 2D transform
  - **Phase 2 (Epochs 6-10)**: Unfreeze inverse 2D transform
  - **Phase 3 (Epochs 11-15)**: Unfreeze inverse projection
  - **Phase 4 (Epochs 16-20)**: Unfreeze all layers
- Uses perceptual loss and EMA
- Multiple runs with different hyperparameters

#### c) **Ablation Studies**
Various experiments exploring:
- Different freezing strategies (2D only, 3D only, projection only, combinations)
- Different perceptual loss weights
- Different learning rates
- EMA vs. no EMA

### 3. Chairs Training
- Training on ShapeNet chairs dataset
- Standard training procedure
- Results visualized in `chair.gif`

## Results & Visualizations

### Output Animations

The following GIF files demonstrate the results of the experiments:

#### `animations.gif` (in `proj1/`)
General animation showcasing novel view synthesis capabilities across different datasets.

#### `mugs.gif` (in `ml-equivariant-neural-rendering-main/`)
Novel view synthesis results for the mugs dataset. Shows the model's ability to generate smooth camera rotations around mug objects.

#### `chair.gif` (in `ml-equivariant-neural-rendering-main/`)
Results for the chairs dataset, demonstrating the model's performance on ShapeNet chair models.

#### `bowl_transfer.gif` and Variants (in `ml-equivariant-neural-rendering-main/`)
Multiple variants of transfer learning results for bowls:

- **`bowl_transfer.gif`**: Main transfer learning result
- **`bowl_transfer_allfreeze.gif`**: Results with all layers frozen
- **`bowl_transfer_Nofreeze.gif`**: Results with no freezing
- **`bowl_transfer_3dFrozen.gif`**: Results with 3D layers frozen
- **`bowl_transfer_2dFrozen.gif`**: Results with 2D layers frozen
- **`bowl_transfer_progression.gif`**: Shows progression through training phases
- **`bowl_transfer_PL*.gif`**: Various perceptual loss experiments
- And many more ablation study results

These animations show the model's ability to:
1. Infer 3D scene representations from single images
2. Rotate these representations in 3D space
3. Render novel viewpoints with high visual quality
4. Successfully transfer knowledge from mugs to bowls domain

### Quantitative Results

For detailed quantitative analysis, loss curves, and performance metrics, please refer to:
- **`Project 1 Write up.pdf`**: Comprehensive analysis of all experiments, results, and findings
- Experiment directories: Each experiment folder contains `loss_history.json`, `epoch_loss_history.json`, and `val_loss_history.json`

## Running the Code

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.13.1 (or compatible version)
- CUDA-capable GPU (recommended)
- Required packages (see `ml-equivariant-neural-rendering-main/requirements.txt`)

### Setup

1. **Navigate to the project directory:**
   ```bash
   cd proj1/ml-equivariant-neural-rendering-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include:
   - torch
   - torchvision
   - pytorch-msssim
   - imageio
   - numpy
   - pillow

3. **Download datasets** (if not already present):
   - Follow instructions in `ml-equivariant-neural-rendering-main/README.md`
   - Datasets should be placed in the `ml-equivariant-neural-rendering-main/` directory

### Training a Model

#### Standard Training
```bash
python experiments.py config.json
```

#### Transfer Learning with Progressive Freezing
```bash
python experiments.py config2.json
```

#### Fine-tuning
```bash
python experiments.py config3.json
```

### Configuration Files

Key configuration parameters:

- **`id`**: Experiment identifier
- **`path_to_data`**: Path to training data
- **`path_to_test_data`**: Path to validation/test data
- **`pretrained_model_path`**: Path to pre-trained model (for transfer learning)
- **`freeze_parts`**: List of model parts to freeze (substring matching)
- **`progressive_freezing`**: Multi-phase freezing schedule
- **`perceptual_loss_weight`**: Weight for VGG perceptual loss
- **`use_ema`**: Enable Exponential Moving Average
- **`finetune_lr`**: Learning rate for fine-tuning
- **`batch_size`**: Training batch size
- **`epochs`**: Number of training epochs
- **`lr`**: Learning rate
- **`ssim_loss_weight`**: Weight for SSIM loss

### Evaluation

#### Quantitative Evaluation (PSNR)
```bash
python evaluate_psnr.py <path_to_model> <path_to_data>
```

#### Visualization and Exploration
Use the Jupyter notebook:
```bash
jupyter notebook exploration.ipynb
```

This notebook allows you to:
- Load trained models
- Generate novel views from single images
- Create visualization GIFs
- Explore the learned scene representations

### Generating GIFs

GIFs can be generated using the visualization utilities in `misc/viz.py`:

```python
from misc.viz import save_img_sequence_as_gif, generate_novel_views
# ... load model and data ...
# Generate novel views and save as GIF
save_img_sequence_as_gif(img_sequence, 'output.gif', nrow=4)
```

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **pytorch-msssim**: Multi-scale SSIM loss
- **imageio**: Image I/O and GIF creation
- **numpy**: Numerical computing
- **Pillow**: Image processing

### Optional Dependencies
- **Jupyter**: For exploration notebook
- **matplotlib**: For plotting (used in notebook)

See `ml-equivariant-neural-rendering-main/requirements.txt` for complete list.

## Pre-trained Models

Pre-trained models are available in the `pre-trained_models/` directory:

- **`mugs.pt`**: Model trained on MugsHQ dataset
- **`chairs.pt`**: Model trained on ShapeNet chairs
- **`cars.pt`**: Model trained on ShapeNet cars
- **`mountains.pt`**: Model trained on 3D mountains dataset

Additionally, experiment-specific models are saved in their respective experiment directories (e.g., `2025-10-08_03-19_mugs-experiment/best_model.pt`).

## Project Structure Details

### Model Architecture

The neural renderer consists of:

1. **Inverse Rendering Path** (Image → 3D Scene):
   - ResNet2D: Processes input image
   - Inverse Projection: Projects 2D features to 3D
   - ResNet3D: Processes 3D features to scene representation

2. **Forward Rendering Path** (3D Scene → Image):
   - ResNet3D: Processes scene representation
   - Projection: Projects 3D features to 2D
   - ResNet2D: Renders final image

3. **Rotation Layer**: Rotates 3D scene representations for novel view synthesis

### Training Process

1. **Forward Pass**: 
   - Input image → Infer scene representation → Rotate scene → Render novel view
   
2. **Loss Computation**:
   - Regression loss (L1 or L2)
   - SSIM loss (optional)
   - Perceptual loss (optional, VGG-based)
   
3. **Backward Pass**: Update model parameters (respecting freezing schedule)

4. **EMA Update**: Update shadow weights if EMA is enabled

## References

### Original Paper
- **Equivariant Neural Rendering** (ICML 2020)
  - Authors: E. Dupont, M. A. Bautista, A. Colburn, A. Sankar, C. Guestrin, J. Susskind, Q. Shan
  - arXiv: https://arxiv.org/abs/2006.07630
  - Citation:
    ```bibtex
    @article{dupont2020equivariant,
      title={Equivariant Neural Rendering},
      author={Dupont, Emilien and Miguel Angel, Bautista and Colburn, Alex and Sankar, Aditya and Guestrin, Carlos and Susskind, Josh and Shan, Qi},
      journal={arXiv preprint arXiv:2006.07630},
      year={2020}
    }
    ```

### Base Repository
- Original codebase: `ml-equivariant-neural-rendering-main/`
- Base README: `ml-equivariant-neural-rendering-main/README.md`
- License: Apple Sample Code License

### Datasets
- **ShapeNet**: http://www.shapenet.org/ (chairs, cars)
- **MugsHQ**: High-quality mug renderings
- **3D Mountains**: Satellite imagery-based dataset

See `ml-equivariant-neural-rendering-main/README-DATA.md` for dataset licensing information.

## Notes

- All experiments were run with validation sets for monitoring overfitting
- Best models are saved based on validation loss
- Training progress (losses, generated images) is saved in experiment directories

