# Brain Tumor Segmentation—CAP 5516 Assignment 2

2D U-Net for brain tumor segmentation on the BraTS dataset with 5-fold cross-validation.

## Setup

```bash
pip install torch monai scikit-learn medpy matplotlib nibabel tqdm
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
├── config.py
├── main.py 
├── dataset/ 
├── data/dataset.py
├── models/unet2d.py
├── training/losses.py
├── training/trainer.py
├── utils/utils.py
└── README.md
```

## Outputs

Results are saved to `./outputs/`:
- `checkpoints/` for best model weights per fold
- `figures/` for segmentation examples (prediction vs ground truth)
- `visualizations/` for raw data visualization samples