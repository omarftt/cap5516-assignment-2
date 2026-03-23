# Brain Tumor Segmentation—CAP 5516 Assignment 2

2D U-Net for brain tumor segmentation on the BraTS dataset with 5-fold cross-validation.

## Setup

```bash
pip install torch monai scikit-learn medpy matplotlib nibabel
```

## Dataset

Download Task01_BrainTumour.tar from [here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) and extract so the structure is:
```
dataset/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

## Usage

**Visualize data samples:**
```bash
python main.py --mode visualize_data
```

**Run 5-fold cross-validation:**
```bash
python main.py --mode crossval
```

## Project Structure

```
├── config.py              # Hyperparameters and paths
├── main.py                # Entry point
├── data/dataset.py        # Data loading, augmentation, 2D slice extraction
├── models/unet2d.py       # 2D U-Net architecture
├── training/losses.py     # Dice + CrossEntropy combined loss
├── training/trainer.py    # Training loop, evaluation
├── utils/utils.py         # Metrics (Dice, HD95), visualization
└── README.md
```

## Outputs

Results are saved to `./outputs/`:
- `checkpoints/` — best model weights per fold
- `figures/` — segmentation examples (prediction vs ground truth)
- `visualizations/` — raw data visualization samples