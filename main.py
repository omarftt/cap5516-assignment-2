from data.dataset import get_file_list, get_fold_dataloaders
from training.trainer import train_fold
from utils.utils import region_names

import numpy as np
import os
import argparse
import shutil

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
    # Extracting some raw samples for ITK-SNAP tool
    data_dicts = get_file_list(config.DATA_DIR)
    samples_dir = os.path.join(config.OUTPUT_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for i in range(3):
        src_img = data_dicts[i]["image"]
        src_lbl = data_dicts[i]["label"]
        name = os.path.basename(src_img).replace(".nii.gz", "")

        shutil.copy2(src_img, os.path.join(samples_dir, f"{name}_image.nii.gz"))
        shutil.copy2(src_lbl, os.path.join(samples_dir, f"{name}_label.nii.gz"))

    print(f"Saved 3 samples to {os.path.abspath(samples_dir)}")


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