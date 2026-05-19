import argparse
import shutil
import tarfile
import urllib.request
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


CFG = {
    "data_dir": "./flowers102",
    "checkpoint_dir": "./checkpoints",
    "num_classes": 102,
    "image_size": 299,
    "batch_size": 32,
    "num_epochs": 20,
    "num_workers": 0 if platform.system() == "Windows" else 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "optimizer": "adam",
    "scheduler": "cosine",
    "freeze_layers": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    "splits": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate InceptionV3 on Oxford 102 Flowers."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval"),
        default="train_eval",
        help="Whether to train, evaluate an existing checkpoint, or do both.",
    )
    parser.add_argument("--data-dir", default=CFG["data_dir"])
    parser.add_argument("--checkpoint-dir", default=CFG["checkpoint_dir"])
    parser.add_argument("--epochs", type=int, default=CFG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=CFG["num_workers"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=CFG["weight_decay"])
    parser.add_argument("--dropout", type=float, default=CFG["dropout"])
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd"),
        default=CFG["optimizer"],
    )
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "step", "none"),
        default=CFG["scheduler"],
    )
    parser.add_argument("--freeze-layers", action="store_true")
    parser.add_argument(
        "--device",
        default=CFG["device"],
        help="Device to use, e.g. cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_model.pth",
        help="Checkpoint filename inside checkpoint-dir.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download step if you already prepared the data locally.",
    )
    return parser.parse_args()


def build_runtime_cfg(args):
    cfg = dict(CFG)
    cfg.update(
        {
            "data_dir": args.data_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "freeze_layers": args.freeze_layers,
            "device": resolve_device(args.device),
        }
    )
    return cfg


def resolve_device(requested_device: str) -> str:
    requested_device = requested_device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested, but this PyTorch build cannot access CUDA. Falling back to CPU.")
        return "cpu"
    return requested_device


def print_device_info(device: str):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Active GPU: {torch.cuda.get_device_name(index)}")
    elif device.startswith("cuda"):
        print("Requested CUDA, but this environment only has a CPU PyTorch build.")


def download_dataset(data_dir: str):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dest = data_dir / Path(url).name
        if not dest.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, dest)

    tgz = data_dir / "102flowers.tgz"
    jpg_dir = data_dir / "jpg"
    if not jpg_dir.exists():
        print("Extracting images ...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(data_dir)

    print("Dataset ready.")
    return data_dir


def build_split_folders(data_dir: str):
    data_dir = Path(data_dir)
    jpg_dir = data_dir / "jpg"

    labels_mat = scipy.io.loadmat(str(data_dir / "imagelabels.mat"))
    splits_mat = scipy.io.loadmat(str(data_dir / "setid.mat"))

    labels = labels_mat["labels"].flatten()
    trnids = splits_mat["trnid"].flatten()
    valid = splits_mat["valid"].flatten()
    tstid = splits_mat["tstid"].flatten()

    split_map = {}
    for i in trnids:
        split_map[i] = "train"
    for i in valid:
        split_map[i] = "val"
    for i in tstid:
        split_map[i] = "test"

    for split in ("train", "val", "test"):
        for cls in range(1, CFG["num_classes"] + 1):
            (data_dir / split / str(cls)).mkdir(parents=True, exist_ok=True)

    for img_id in range(1, len(labels) + 1):
        split = split_map[img_id]
        cls = labels[img_id - 1]
        src = jpg_dir / f"image_{img_id:05d}.jpg"
        dst = data_dir / split / str(cls) / f"image_{img_id:05d}.jpg"
        if not dst.exists():
            shutil.copy(src, dst)

    print("Split folders created.")


class FlowerDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples = []
        split_dir = Path(root) / split
        for cls_dir in sorted(split_dir.iterdir(), key=lambda p: int(p.name)):
            cls_id = int(cls_dir.name) - 1
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_loaders(data_dir, batch_size, image_size, num_workers, device):
    loaders = {}
    pin_memory = device.startswith("cuda")
    for split in ("train", "val", "test"):
        ds = FlowerDataset(
            data_dir, split, transform=get_transforms(image_size, split == "train")
        )
        loader_kwargs = {
            "dataset": ds,
            "batch_size": batch_size,
            "shuffle": (split == "train"),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 1
        loaders[split] = DataLoader(**loader_kwargs)
        print(f"  {split:5s}: {len(ds):5d} images")
    return loaders


def build_model(
    num_classes: int, dropout: float, freeze_backbone: bool, pretrained: bool = True
):
    weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.inception_v3(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feat, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )

    aux_in = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_in, num_classes)

    return model


def unfreeze_model(model, unfreeze_from_layer: int = 7):
    children = list(model.children())
    for child in children[unfreeze_from_layer:]:
        for param in child.parameters():
            param.requires_grad = True
    print(f"Unfrozen layers from index {unfreeze_from_layer}")


def build_optimizer(model, cfg: dict):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = cfg["optimizer"]
    lr = cfg["lr"]
    wd = cfg["weight_decay"]

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name = cfg["scheduler"]
    epochs = cfg["num_epochs"]

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return None


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, use_aux=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        if use_aux and isinstance(outputs, tuple):
            logits, aux = outputs
            loss = criterion(logits, labels) + 0.4 * criterion(aux, labels)
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.StepLR):
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_predictions=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, labels)
        preds = logits.argmax(1)

        running_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if collect_predictions:
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if collect_predictions:
        return (
            running_loss / total,
            correct / total,
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
        )

    return running_loss / total, correct / total


def train(cfg: dict, loaders: dict, checkpoint_path: Path):
    device = cfg["device"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(
        cfg["num_classes"], cfg["dropout"], cfg["freeze_layers"], pretrained=True
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg["num_epochs"] + 1):
        if cfg["freeze_layers"] and epoch == cfg["num_epochs"] // 2 + 1:
            unfreeze_model(model)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, criterion, device
        )
        vl_loss, vl_acc = evaluate(model, loaders["val"], criterion, device)

        if isinstance(scheduler, optim.lr_scheduler.StepLR):
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:3d}/{cfg['num_epochs']} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {vl_loss:.4f} acc {vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return model, history, best_val_acc


def plot_history(history: dict, save_path: Path):
    if not history["train_loss"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved training curves -> {save_path}")


def build_confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray, save_path: Path, class_labels=None, normalize=True
):
    plot_matrix = matrix.astype(np.float32)
    if normalize:
        row_sums = plot_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_matrix = plot_matrix / row_sums

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(plot_matrix, interpolation="nearest", cmap="Blues")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_labels is None:
        class_labels = [str(i) for i in range(matrix.shape[0])]
    tick_step = max(1, len(class_labels) // 12)
    tick_positions = list(range(0, len(class_labels), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([class_labels[i] for i in tick_positions], rotation=90)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([class_labels[i] for i in tick_positions])

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def evaluate_and_save_reports(model, loaders, cfg, checkpoint_dir: Path):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, labels, preds = evaluate(
        model,
        loaders["test"],
        criterion,
        cfg["device"],
        collect_predictions=True,
    )
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    confusion_matrix = build_confusion_matrix(labels, preds, cfg["num_classes"])
    np.save(checkpoint_dir / "test_confusion_matrix.npy", confusion_matrix)
    plot_confusion_matrix(
        confusion_matrix,
        checkpoint_dir / "test_confusion_matrix.png",
        class_labels=[str(i) for i in range(1, cfg["num_classes"] + 1)],
        normalize=True,
    )


def main():
    args = parse_args()
    cfg = build_runtime_cfg(args)
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    print_device_info(cfg["device"])

    if not args.skip_download:
        download_dataset(cfg["data_dir"])
    build_split_folders(cfg["data_dir"])

    print("\nBuilding data loaders ...")
    loaders = get_loaders(
        cfg["data_dir"],
        cfg["batch_size"],
        cfg["image_size"],
        cfg["num_workers"],
        cfg["device"],
    )

    history = None

    if args.mode in ("train", "train_eval"):
        model, history, _ = train(cfg, loaders, checkpoint_path)
        plot_history(history, checkpoint_dir / "training_curves.png")
    else:
        model = build_model(
            cfg["num_classes"],
            cfg["dropout"],
            cfg["freeze_layers"],
            pretrained=False,
        ).to(cfg["device"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train first or pass a valid checkpoint."
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg["device"]))

    if args.mode in ("eval", "train_eval"):
        evaluate_and_save_reports(model, loaders, cfg, checkpoint_dir)


if __name__ == "__main__":
    main()
