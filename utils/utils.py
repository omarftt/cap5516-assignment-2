import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95

region_names = ["WT", "TC", "ET"]

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
 
 
def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
 
 
def labels_to_regions(label_map):
    """
    Converting integer label map to binary masks for regions WT, TC, ET
    Dataset labels: 0=bg, 1=edema, 2=non-enhancing tumor, 3=enhancing tumor
    WT (all tumor)= labels 1, 2, 3
    TC (tumor core)= labels 2, 3 (non-enhancing and enhancing)
    ET (enhancing tumor)= label 3
    """
    wt = (label_map > 0).astype(np.uint8)
    tc = ((label_map == 2) | (label_map == 3)).astype(np.uint8)
    et = (label_map == 3).astype(np.uint8)
    return {"WT": wt, "TC": tc, "ET": et}
 
 
def compute_dice(pred, gt):
    # Computing dice score
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    return (2.0 * intersection) / union
 
 
def compute_hd95(pred, gt):
    # Computing Hausdorff distance at 95%
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return np.nan

    return hd95(pred, gt)
 
 
def evaluate_volume(pred_labels, gt_labels):
    # Computing Dice and HD for WT, TC, ET on a 3D volume
    pred_regions = labels_to_regions(pred_labels)
    gt_regions = labels_to_regions(gt_labels)
    results = {}
    for name in region_names:
        results[f"Dice_{name}"]= compute_dice(pred_regions[name],gt_regions[name])
        results[f"HD95_{name}"]= compute_hd95(pred_regions[name],gt_regions[name])
    return results

def label_map_to_rgb(label_map):
    # Transform integer label map  to RGB image
    label_colors = {
        0: [0.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0],
        2: [1.0, 1.0, 0.0],
        3: [1.0, 0.0, 0.0],
    }

    h, w = label_map.shape
    rgb = np.zeros((h, w, 3))
    for val, color in label_colors.items():
        rgb[label_map == val] = color
    return rgb


def plot_sample(image, gt, pred=None, slice_idx=None, save_path=None):
    # Plot MRI modalities, groudn trurth, and prediction for one slice
    modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]
    ncols = 6 if pred is not None else 5
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3))

    # Show each MRI modal
    for i in range(4):
        axes[i].imshow(image[i], cmap="gray")
        axes[i].set_title(modality_names[i])
        axes[i].axis("off")

    # Create a gt overlay
    gt_rgb = label_map_to_rgb(gt)
    axes[4].imshow(image[0], cmap="gray", alpha=0.3)
    axes[4].imshow(gt_rgb, alpha=0.7)
    axes[4].set_title("Ground Truth")
    axes[4].axis("off")

    if pred is not None:
        pred_rgb = label_map_to_rgb(pred)
        axes[5].imshow(image[0], cmap="gray", alpha=0.3)
        axes[5].imshow(pred_rgb, alpha=0.7)
        axes[5].set_title("Prediction")
        axes[5].axis("off")

    title = f"Slice {slice_idx}" if slice_idx is not None else "BraTS Sample"
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regions(image, gt, pred, slice_idx=None, save_path=None):
    # Plot MRI modalities and GT vs Pred for WT, TC, ET regions
    from matplotlib.gridspec import GridSpec

    gt_regions = labels_to_regions(gt)
    pred_regions = labels_to_regions(pred)

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 4, figure=fig)

    # 4 MRI modalities
    modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(image[i], cmap="gray")
        ax.set_title(modality_names[i])
        ax.axis("off")

    # GT and Pred for each region
    for row, name in enumerate(region_names):
        ax_gt = fig.add_subplot(gs[row + 1, 1])
        ax_gt.imshow(image[0], cmap="gray", alpha=0.3)
        ax_gt.imshow(gt_regions[name], cmap="Reds", alpha=0.7)
        ax_gt.set_title(f"GT {name}")
        ax_gt.axis("off")

        ax_pred = fig.add_subplot(gs[row + 1, 2])
        ax_pred.imshow(image[0], cmap="gray", alpha=0.3)
        ax_pred.imshow(pred_regions[name], cmap="Reds", alpha=0.7)
        ax_pred.set_title(f"Pred {name}")
        ax_pred.axis("off")

    title = f"Region Comparison"
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()