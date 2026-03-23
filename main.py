from data.dataset import get_file_list, get_fold_dataloaders
from training.trainer import train_fold
from utils.utils import region_names, plot_sample
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

import numpy as np
import os
import argparse

import config


def run_crossval():
    # Train and eval on 5-fold cross-val
    data_dicts = get_file_list(config.DATA_DIR)
    all_results = []

    # Execution per each of the 5 folds
    for fold in range(5):
        train_loader, val_loader= get_fold_dataloaders(fold, data_dicts)
        fold_results = train_fold(train_loader, val_loader, val_loader.dataset, fold)
        all_results.append(fold_results)

    # Showing all results
    for fold, res in enumerate(all_results):
        for name in region_names:
            print(f"Fold {fold+1} in {name}: Dice={res[f'Dice_{name}']:.4f} HD95={res[f'HD95_{name}']:.2f}")

    # Sow avg for regions
    for name in region_names:
        avg_dice = np.mean([r[f"Dice_{name}"] for r in all_results])
        avg_hd = np.mean([r[f"HD95_{name}"] for r in all_results])
        print(f"Average for {name}: Dice={avg_dice:.4f} HD95={avg_hd:.2f}")


def run_visualize_data():
    # Generating visualization figures for MRI samples and segmentation masks

    data_dicts = get_file_list(config.DATA_DIR)
    vis_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Minimal transforms just for loading
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    for i in range(3):
        vol = transform(data_dicts[i])
        image = vol["image"]
        label = vol["label"][0].numpy().astype(int)
        mid = image.shape[-1] // 2

        save_path = os.path.join(vis_dir,f"sample_{i}_slice{mid}.png")
        plot_sample(
            image[:, :, :, mid].numpy(),
            label[:, :, mid],
            pred=None,
            slice_idx=mid,
            save_path=save_path,
        )

    print(f"Saved on {os.path.abspath(vis_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="crossval", choices=["crossval", "visualize_data"])
    args = parser.parse_args()

    if args.mode == "crossval":
        run_crossval()
    elif args.mode == "visualize_data":
        run_visualize_data()


if __name__ == "__main__":
    main()