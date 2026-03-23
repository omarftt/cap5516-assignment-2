from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
import json
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
)

import config

PREPROCESSED_DIR = os.path.join(config.DATA_DIR, "preprocessing")


def get_file_list(data_dir):
    # Read dataset.json and return list of image and label path
    json_path = os.path.join(data_dir, "dataset.json")
    with open(json_path, "r") as f:
        dataset_info = json.load(f)
    data_dicts = []
    for item in dataset_info["training"]:
        img_path = os.path.join(data_dir,item["image"].replace("./",""))
        label_path = os.path.join(data_dir,item["label"].replace("./",""))
        data_dicts.append({"image": img_path, "label": label_path})

    return data_dicts


def get_transforms():
    # MONAI transforms for volume preprocessing
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
    ])


def preprocess_slices(data_dicts):
    # Extract  valid 2D slices from 3D volumes and save as .pt files

    # If already preprocessed not run again
    if os.path.exists(PREPROCESSED_DIR) and len(os.listdir(PREPROCESSED_DIR)) > 0:
        print(f"Preprocessed slices already exist")
        return

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    transform = get_transforms()

    print("Preprocessing volumes to 2D slices (one-time operation)...")
    for vol_idx, data_dict in enumerate(tqdm(data_dicts, desc="Volumes")):
        vol = transform(data_dict)
        image = vol["image"]
        label = vol["label"]
        num_slices = image.shape[-1]

        for s in range(num_slices):
            slice_data = image[:, :, :, s]
            brain_frac = (slice_data.abs().sum(dim=0) > 0).float().mean()

            # To avoid many useless slices
            if brain_frac > config.MIN_BRAIN_FRACTION:
                img_slice = slice_data.clone().detach().float()
                lbl_slice = label[0, :, :, s].clone().detach().byte()
                save_path = os.path.join(PREPROCESSED_DIR, f"vol{vol_idx:04d}_s{s:04d}.pt")
                torch.save({"image": img_slice, "label": lbl_slice}, save_path)

    print(f"Preprocessing done. Saved to {PREPROCESSED_DIR}")


class PreprocessedSliceDataset(Dataset):
    """
    Loads pre-extracted 2D slices from .pt files on disk.
    """

    def __init__(self, vol_indices):
        # Collect all slice files belonging to the given volume indices
        self.slice_files = []
        for vol_idx in vol_indices:
            prefix = f"vol{vol_idx:04d}_"
            for fname in os.listdir(PREPROCESSED_DIR):
                if fname.startswith(prefix):
                    self.slice_files.append(os.path.join(PREPROCESSED_DIR, fname))
        self.slice_files.sort()
        print(f"Found {len(self.slice_files)} slices for {len(vol_indices)} volumes")

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        data = torch.load(self.slice_files[idx], weights_only=True)
        return data["image"], data["label"].long()


def get_fold_dataloaders(fold, data_dicts):
    # Preprocess all slices to disk if not done yet
    preprocess_slices(data_dicts)

    # Get a train and val dataloaders for a fold
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    splits = list(kf.split(data_dicts))
    train_idx,val_idx = splits[fold]

    train_dataset = PreprocessedSliceDataset(train_idx)
    val_dataset = PreprocessedSliceDataset(val_idx)

    # Store val volume indices and dicts for evaluation later
    val_dataset.val_dicts = [data_dicts[i] for i in val_idx]

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader