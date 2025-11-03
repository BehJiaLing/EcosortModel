import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainingConfig:
    data_root: str
    mapping_json: str
    out_dir: str = "runs/waste-cnn"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 4
    label_smoothing: float = 0.1
    recycle_loss_weight: float = 1.0
    amp: bool = True
    seed: int = 42
    resume: str = ""
    # targets for virtual expansion
    train_target: int = 500
    val_target: int = 200
    test_target: int = 100
    eval_test_each_epoch: bool = False  # set True if you want test metrics every epoch


# ----------------------------
# Dataset wrappers
# ----------------------------
class WasteMultiTaskDataset(Dataset):
    """
    Wraps an ImageFolder to return (image, class_idx, recyclable_label)
    recyclable_label is derived from class_name via a provided mapping.
    """
    def __init__(self, imagefolder: datasets.ImageFolder, class_to_recycle: Dict[str, int]):
        self.base = imagefolder
        self.samples = imagefolder.samples
        self.class_to_idx = imagefolder.class_to_idx
        inv_idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.recycle_targets = []
        for _, class_idx in self.samples:
            cname = inv_idx_to_class[class_idx]
            if cname not in class_to_recycle:
                raise KeyError(
                    f"Class '{cname}' not found in mapping JSON. "
                    f"Make sure every folder/class has a 0/1 mapping."
                )
            self.recycle_targets.append(int(class_to_recycle[cname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        path, class_idx = self.samples[i]
        img = self.base.loader(path)
        if self.base.transform is not None:
            img = self.base.transform(img)
        recycle_label = self.recycle_targets[i]
        return img, class_idx, torch.tensor(recycle_label, dtype=torch.float32)


class AugmentToLengthDataset(Dataset):
    """
    Virtually increases the dataset length to target_len by adding extra, *augmented*
    samples. It never duplicates a file on disk. Each base image appears at most twice:
    once with the base transform, and at most once with 'extra_transform'.
    """
    def __init__(self, core_ds: WasteMultiTaskDataset, target_len: int, extra_transform: transforms.Compose):
        self.core = core_ds
        self.base_len = len(core_ds)
        self.target_len = max(target_len, self.base_len)
        self.extra_transform = extra_transform

        self.extra_count = self.target_len - self.base_len
        # choose unique base indices (without replacement) for the extras
        if self.extra_count > 0:
            rng = random.Random(1337)  # fixed for reproducibility
            all_idx = list(range(self.base_len))
            rng.shuffle(all_idx)
            if self.extra_count > self.base_len:
                # if target_len is more than 2x, cycle without repeating within 'extras' selection
                cycles = self.extra_count // self.base_len
                remainder = self.extra_count % self.base_len
                self.extra_indices = []
                for c in range(cycles):
                    rng.shuffle(all_idx)
                    self.extra_indices.extend(all_idx)  # each base appears again with aug
                rng.shuffle(all_idx)
                self.extra_indices.extend(all_idx[:remainder])
            else:
                self.extra_indices = all_idx[:self.extra_count]
        else:
            self.extra_indices = []

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx: int):
        if idx < self.base_len:
            # original sample with the dataset's transform
            return self.core[idx]
        # extra augmented sample
        base_idx = self.extra_indices[idx - self.base_len]
        path, class_idx = self.core.samples[base_idx]
        img = self.core.base.loader(path)
        img = self.extra_transform(img)
        recycle_label = self.core.recycle_targets[base_idx]
        return img, class_idx, torch.tensor(recycle_label, dtype=torch.float32)


# ----------------------------
# Model
# ----------------------------
class ResNetMultiHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, num_classes),
        )
        self.recycle_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        class_logits = self.classifier(feats)
        recycle_logit = self.recycle_head(feats).squeeze(-1)
        return class_logits, recycle_logit


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == target).float().mean().item()


# ----------------------------
# Train / Val loops
# ----------------------------
def train_one_epoch(model, loader, device, optimizer, scaler, ce_loss, bce_loss, cfg: TrainingConfig):
    model.train()
    total, loss_sum, acc_sum, racc_sum = 0, 0.0, 0.0, 0.0
    for imgs, y_class, y_recycle in loader:
        imgs = imgs.to(device, non_blocking=True)
        y_class = y_class.to(device, non_blocking=True)
        y_recycle = y_recycle.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            class_logits, recycle_logit = model(imgs)
            loss_class = ce_loss(class_logits, y_class)
            loss_recycle = bce_loss(recycle_logit, y_recycle)
            loss = loss_class + cfg.recycle_loss_weight * loss_recycle

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bsz = imgs.size(0)
        total += bsz
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(class_logits, y_class) * bsz
        racc_sum += binary_accuracy(recycle_logit, y_recycle) * bsz

    return {"loss": loss_sum / total, "class_acc": acc_sum / total, "recycle_acc": racc_sum / total}


@torch.no_grad()
def evaluate(model, loader, device, ce_loss, bce_loss, cfg: TrainingConfig):
    model.eval()
    total, loss_sum, acc_sum, racc_sum = 0, 0.0, 0.0, 0.0
    for imgs, y_class, y_recycle in loader:
        imgs = imgs.to(device, non_blocking=True)
        y_class = y_class.to(device, non_blocking=True)
        y_recycle = y_recycle.to(device, non_blocking=True)

        class_logits, recycle_logit = model(imgs)
        loss_class = ce_loss(class_logits, y_class)
        loss_recycle = bce_loss(recycle_logit, y_recycle)
        loss = loss_class + cfg.recycle_loss_weight * loss_recycle

        bsz = imgs.size(0)
        total += bsz
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(class_logits, y_class) * bsz
        racc_sum += binary_accuracy(recycle_logit, y_recycle) * bsz

    return {"loss": loss_sum / total, "class_acc": acc_sum / total, "recycle_acc": racc_sum / total}


# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer, scheduler, scaler, best_val_metric: float, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "best_val_metric": best_val_metric,
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val_metric", 0.0)
    meta = ckpt.get("meta", {})
    return start_epoch, best_val, meta


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CNN for waste type + recyclability (3-way split)")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing train/ val/ test/")
    parser.add_argument("--mapping_json", type=str, required=True, help="JSON mapping: {class_name: 0|1}")
    parser.add_argument("--out_dir", type=str, default="runs/waste-cnn")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--recycle_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--train_target", type=int, default=500)
    parser.add_argument("--val_target", type=int, default=200)
    parser.add_argument("--test_target", type=int, default=100)
    parser.add_argument("--eval_test_each_epoch", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig(
        data_root=args.data_root,
        mapping_json=args.mapping_json,
        out_dir=args.out_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        label_smoothing=args.label_smoothing,
        recycle_loss_weight=args.recycle_loss_weight,
        amp=not args.no_amp,
        seed=args.seed,
        resume=args.resume,
        train_target=args.train_target,
        val_target=args.val_target,
        test_target=args.test_target,
        eval_test_each_epoch=args.eval_test_each_epoch,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # --------- Transforms ----------
    # Base (train) transforms + a stronger 'extra' transform for augmented extras
    train_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.CenterCrop(int(cfg.img_size * 1.15)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extra_train_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.20)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Val/Test: evaluation-friendly base + a light extra transform for virtual expansion
    eval_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extra_eval_tfms = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --------- Datasets ----------
    train_dir = os.path.join(cfg.data_root, "train")
    val_dir = os.path.join(cfg.data_root, "val")
    test_dir = os.path.join(cfg.data_root, "test")

    train_base = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_base = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_base = datasets.ImageFolder(test_dir, transform=eval_tfms)

    with open(cfg.mapping_json, "r") as f:
        class_to_recycle = json.load(f)

    # Ensure mapping covers all classes in TRAIN split
    missing = set(train_base.class_to_idx.keys()) - set(class_to_recycle.keys())
    if missing:
        raise KeyError(f"Classes missing from mapping JSON: {sorted(missing)}")

    # Wrap with multitask labels
    train_core = WasteMultiTaskDataset(train_base, class_to_recycle)
    val_core = WasteMultiTaskDataset(val_base, class_to_recycle)
    test_core = WasteMultiTaskDataset(test_base, class_to_recycle)

    # Virtual expansion to exact targets (or leave as-is if already larger)
    train_ds = AugmentToLengthDataset(train_core, cfg.train_target, extra_transform=extra_train_tfms)
    val_ds = AugmentToLengthDataset(val_core, cfg.val_target, extra_transform=extra_eval_tfms)
    test_ds = AugmentToLengthDataset(test_core, cfg.test_target, extra_transform=extra_eval_tfms)

    num_classes = len(train_base.classes)
    print(f"Classes ({num_classes}): {train_base.classes}")
    print(f"Train base: {len(train_core)} -> using {len(train_ds)}")
    print(f"Val   base: {len(val_core)} -> using {len(val_ds)}")
    print(f"Test  base: {len(test_core)} -> using {len(test_ds)}")

    # --------- Loaders ----------
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True, drop_last=False)

    # --------- Model / Optim ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetMultiHead(num_classes=num_classes).to(device)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    start_epoch, best_val_metric = 0, 0.0
    meta = {
        "classes": train_base.classes,
        "class_to_idx": train_base.class_to_idx,
        "class_to_recycle": class_to_recycle,
        "config": asdict(cfg),
    }
    if cfg.resume and os.path.isfile(cfg.resume):
        start_epoch, best_val_metric, _meta = load_checkpoint(cfg.resume, model, optimizer, scheduler, scaler)
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}, best metric {best_val_metric:.4f}")

    print(f"Using device: {device}, AMP: {cfg.amp}")

    for epoch in range(start_epoch, cfg.epochs):
        train_stats = train_one_epoch(model, train_loader, device, optimizer, scaler, ce_loss, bce_loss, cfg)
        val_stats = evaluate(model, val_loader, device, ce_loss, bce_loss, cfg)
        if cfg.eval_test_each_epoch:
            test_stats = evaluate(model, test_loader, device, ce_loss, bce_loss, cfg)
        else:
            test_stats = None
        scheduler.step()

        msg = (f"Epoch [{epoch+1}/{cfg.epochs}] "
               f"Train: loss={train_stats['loss']:.4f} class_acc={train_stats['class_acc']:.4f} "
               f"recycle_acc={train_stats['recycle_acc']:.4f} | "
               f"Val: loss={val_stats['loss']:.4f} class_acc={val_stats['class_acc']:.4f} "
               f"recycle_acc={val_stats['recycle_acc']:.4f}")
        if test_stats:
            msg += (f" | Test: loss={test_stats['loss']:.4f} class_acc={test_stats['class_acc']:.4f} "
                    f"recycle_acc={test_stats['recycle_acc']:.4f}")
        print(msg)

        # save 'latest'
        save_checkpoint(os.path.join(cfg.out_dir, "latest.pt"), epoch, model, optimizer, scheduler, scaler, best_val_metric, meta)

        # choose best by mean of class_acc + recycle_acc on VAL
        main_metric = (val_stats["class_acc"] + val_stats["recycle_acc"]) / 2.0
        if main_metric > best_val_metric:
            best_val_metric = main_metric
            save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), epoch, model, optimizer, scheduler, scaler, best_val_metric, meta)
            print(f"➡  New best checkpoint saved (metric={best_val_metric:.4f})")

    print("Training complete.")
    print(f"Best validation metric: {best_val_metric:.4f}")
    print(f"Checkpoints in: {cfg.out_dir}")
    print("Use the 'best.pt' for inference.")


if __name__ == "__main__":
    main()
