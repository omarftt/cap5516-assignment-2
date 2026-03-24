import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
from models.unet2d import UNet2D
from training.losses import DiceCELoss
from utils.utils import save_checkpoint, load_checkpoint, evaluate_volume, plot_sample, region_names
from data.dataset import get_transforms


def train_single_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)

        labels = labels.to(device)
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_single_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader.dataset), correct / total


def evaluate_on_volumes(model, val_dataset, device, fold=0):
    # Make a prediction on all slices per volume and save figures
    model.eval()
    transform = get_transforms()
    val_dicts = val_dataset.val_dicts
    all_results = []

    with torch.no_grad():
        # Loop on validation volumes
        for vol_idx in range(len(val_dicts)):
            vol = transform(val_dicts[vol_idx])
            image = vol["image"]
            ground_truth = vol["label"][0].numpy().astype(int)
            num_slices = image.shape[-1]
            pred_volume = np.zeros_like(ground_truth)

            # Predictone slice per execution
            for n_slice in range(num_slices):
                img_slice = image[:, :, :, n_slice].unsqueeze(0).to(device)
                output = model(img_slice)
                pred_slice = output.argmax(dim=1).cpu().numpy()[0]
                pred_volume[:, :, n_slice] = pred_slice

            # Compute metrics
            results = evaluate_volume(pred_volume, ground_truth)
            all_results.append(results)

            # Save example figures for first 3 volumes
            if vol_idx < 3:
                mid = num_slices // 2
                save_path = os.path.join(config.FIGURES_DIR, f"fold{fold}_vol{vol_idx}_slice{mid}.png")
                plot_sample(
                    image[:, :, :, mid].numpy(),
                    ground_truth[:, :, mid],
                    pred_volume[:, :, mid],
                    slice_idx=mid,
                    save_path=save_path,
                )

    # Average across volumes
    mean_results = {}
    for key in all_results[0]:
        mean_results[key] = np.mean([r[key] for r in all_results])

    return mean_results


def train_fold(train_loader, val_loader, val_dataset, fold, device):
    # Train one fold and evaluate 

    model = UNet2D(in_channels=4, out_channels=4).to(device)
    criterion = DiceCELoss(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_fold{fold}.pth")

    best_val_loss = float("inf")
    patience_counter = 0
    log_dir = os.path.join(config.OUTPUT_DIR, "runs", f"fold{fold}")
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"Training fold {fold + 1}")
    pbar = tqdm(range(config.NUM_EPOCHS), desc=f"Fold {fold+1}")
    for epoch in pbar:
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_single_epoch(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

        # If there is new best model, save it
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best and evaluate on volumes
    print(f"Evaluating fold {fold + 1} on volumes...")
    model = load_checkpoint(UNet2D(in_channels=4, out_channels=4).to(device), ckpt_path, device)
    mean_results = evaluate_on_volumes(model, val_dataset, device, fold)

    for name in region_names:
        print(f"{name}: Dice={mean_results[f'Dice_{name}']:.4f}, HD95={mean_results[f'HD95_{name}']:.2f}")
        writer.add_scalar(f"Dice/{name}", mean_results[f"Dice_{name}"], fold)
        writer.add_scalar(f"HD95/{name}", mean_results[f"HD95_{name}"], fold)

    writer.close()
    return mean_results