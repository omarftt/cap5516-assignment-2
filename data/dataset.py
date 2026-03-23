from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
import json

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.data import CacheDataset

import config


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


def get_train_transforms():
    # MONAI transforms for training (preprocessing and data augmentation)
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


def get_val_transforms():
    # MONAI transforms for validation (preprocessing only)
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
    ])


class BraTSSliceDataset(Dataset):
    """
    Use MONAI for 3D volumes and serves individual 2D axial slices
    """

    def __init__(self, data_dicts, transform):
        print(f"Loading {len(data_dicts)} volumes...")
        self.volume_ds = CacheDataset(
            data_dicts, transform,
            cache_rate=1.0,
            num_workers=config.NUM_WORKERS,
        )

        # Build flat slice index
        self.indices = []
        for vol_idx in range(len(self.volume_ds)):
            vol = self.volume_ds[vol_idx]
            image = vol["image"]
            num_slices = image.shape[-1]
            for s in range(num_slices):
                slice_data = image[:, :, :, s]
                brain_frac = (slice_data.abs().sum(dim=0) > 0).float().mean()

                # To avoid many useless slices
                if brain_frac > config.MIN_BRAIN_FRACTION:
                    self.indices.append((vol_idx, s))

        print(f"Total valid slices:{len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.indices[idx]
        vol = self.volume_ds[vol_idx]
        image = vol["image"][:, :, :, slice_idx]
        label = vol["label"][0, :, :, slice_idx].long()
        return image, label


def get_fold_dataloaders(fold, data_dicts):
    # Get a train and val dataloaders for a fold
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    splits = list(kf.split(data_dicts))
    train_idx,val_idx = splits[fold]
    train_dicts = [data_dicts[i] for i in train_idx]
    validation_dicts = [data_dicts[i] for i in val_idx]
    
    train_dataset = BraTSSliceDataset(train_dicts, get_train_transforms())
    val_dataset = BraTSSliceDataset(validation_dicts, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader