import argparse
import json
import logging
import math
import platform
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import scipy.io
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "flowers102"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints_adaptive_loss"
NUM_CLASSES = 102
IMAGE_SIZE = 299

QUICK_PROFILE = {
    "epochs": 5,
    "optuna_trials": 3,
    "patience": 3,
}

FULL_PROFILE = {
    "epochs": 25,
    "optuna_trials": 12,
    "patience": 7,
}


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    profile: str
    num_classes: int = NUM_CLASSES
    image_size: int = IMAGE_SIZE
    batch_size: int = 16
    num_epochs: int = QUICK_PROFILE["epochs"]
    optuna_trials: int = QUICK_PROFILE["optuna_trials"]
    patience: int = QUICK_PROFILE["patience"]
    num_workers: int = 0 if platform.system() == "Windows" else 4
    weight_decay: float = 1e-4
    dropout: float = 0.4
    aux_loss_weight: float = 0.4
    grad_clip_norm: float = 1.0
    min_delta: float = 1e-4
    max_loss_threshold: float = 1e4
    eps: float = 1e-8
    seed: int = 42
    disable_amp: bool = False
    pretrained: bool = True
    study_name: str = "adaptive_hybrid_loss_study"
    save_every_epoch: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train InceptionV3 on Oxford Flowers-102 with a custom adaptive hybrid "
            "loss: CE + lambda1 * Focal + lambda2 * AdaptiveConfusion."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the Oxford Flowers-102 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder used for all adaptive-loss checkpoints and reports.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="Quick uses 5 epochs by default; full uses a longer schedule.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the profile epoch count.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Override the profile Optuna trial count.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size. A conservative default helps RTX 4060 stability.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0 if platform.system() == "Windows" else 4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision even when CUDA is available.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="adaptive_hybrid_loss_study",
        help="Friendly name for the Optuna study outputs.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save an epoch checkpoint after each epoch in addition to the best model.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    profile_cfg = QUICK_PROFILE if args.profile == "quick" else FULL_PROFILE
    return ExperimentConfig(
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        profile=args.profile,
        batch_size=args.batch_size,
        num_epochs=args.epochs if args.epochs is not None else profile_cfg["epochs"],
        optuna_trials=(
            args.optuna_trials
            if args.optuna_trials is not None
            else profile_cfg["optuna_trials"]
        ),
        patience=args.patience if args.patience is not None else profile_cfg["patience"],
        num_workers=args.num_workers,
        seed=args.seed,
        disable_amp=args.disable_amp,
        study_name=args.study_name,
        save_every_epoch=args.save_every_epoch,
    )


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("adaptive_loss_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def get_device(logger: logging.Logger) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        logger.info("Active GPU: %s", torch.cuda.get_device_name(gpu_index))
    return device


# ============================================================
# Dataset Loading
# ============================================================

def ensure_split_folders(data_dir: Path, num_classes: int, logger: logging.Logger) -> None:
    required_splits = [data_dir / split for split in ("train", "val", "test")]
    if all(split.exists() for split in required_splits):
        logger.info("Detected existing train/val/test folders. Reusing local dataset layout.")
        return

    labels_path = data_dir / "imagelabels.mat"
    splits_path = data_dir / "setid.mat"
    jpg_dir = data_dir / "jpg"
    if not labels_path.exists() or not splits_path.exists() or not jpg_dir.exists():
        raise FileNotFoundError(
            "Dataset folders are missing and the raw Oxford Flowers files needed to "
            "rebuild them were not found."
        )

    logger.info("Split folders not found. Rebuilding train/val/test folders from the Oxford metadata.")
    labels = scipy.io.loadmat(str(labels_path))["labels"].flatten()
    split_ids = scipy.io.loadmat(str(splits_path))
    split_map: Dict[int, str] = {}

    for image_id in split_ids["trnid"].flatten():
        split_map[int(image_id)] = "train"
    for image_id in split_ids["valid"].flatten():
        split_map[int(image_id)] = "val"
    for image_id in split_ids["tstid"].flatten():
        split_map[int(image_id)] = "test"

    for split in ("train", "val", "test"):
        for class_id in range(1, num_classes + 1):
            (data_dir / split / str(class_id)).mkdir(parents=True, exist_ok=True)

    for image_id in range(1, len(labels) + 1):
        class_id = int(labels[image_id - 1])
        split_name = split_map[image_id]
        source = jpg_dir / f"image_{image_id:05d}.jpg"
        destination = data_dir / split_name / str(class_id) / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

    logger.info("Finished rebuilding split folders.")


class FlowerDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_names: List[str] = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_dirs = sorted(
            [path for path in split_dir.iterdir() if path.is_dir()],
            key=lambda path: int(path.name),
        )
        self.class_names = [path.name for path in class_dirs]

        for class_dir in class_dirs:
            class_index = int(class_dir.name) - 1
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, class_index))

        if not self.samples:
            raise RuntimeError(f"No images were found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ============================================================
# Transforms
# ============================================================

def build_transforms(image_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
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


def create_data_loaders(
    cfg: ExperimentConfig, device: torch.device, logger: logging.Logger
) -> Dict[str, DataLoader]:
    pin_memory = device.type == "cuda"
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        dataset = FlowerDataset(
            cfg.data_dir,
            split,
            transform=build_transforms(cfg.image_size, is_train=(split == "train")),
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": cfg.batch_size,
            "shuffle": split == "train",
            "num_workers": cfg.num_workers,
            "pin_memory": pin_memory,
        }
        if cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loaders[split] = DataLoader(**loader_kwargs)
        logger.info("%s split: %d images", split, len(dataset))

    return loaders


# ============================================================
# Model
# ============================================================

def build_model(cfg: ExperimentConfig) -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.inception_v3(weights=weights, aux_logits=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=cfg.dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=cfg.dropout / 2.0),
        nn.Linear(512, cfg.num_classes),
    )

    aux_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_features, cfg.num_classes)
    return model


def extract_logits(outputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(outputs, "logits"):
        aux_logits = outputs.aux_logits if hasattr(outputs, "aux_logits") else None
        return outputs.logits, aux_logits
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        return outputs[0], None
    return outputs, None


# ============================================================
# Custom Loss
# ============================================================

class AdaptiveHybridLoss(nn.Module):
    """
    Research-style hybrid loss for fine-grained classification:

        L = CE + lambda1 * Focal + lambda2 * AdaptiveConfusion

    Why the components matter:
    - Cross-Entropy anchors the model to the correct class likelihood.
    - Focal Loss magnifies hard examples by increasing pressure when pt is low.
    - Adaptive Confusion multiplies prediction entropy by (1 - pt), so it only
      penalizes uncertainty strongly when the target class confidence is weak.

    This makes the confusion penalty adaptive instead of uniformly suppressing
    entropy for every sample.
    """

    def __init__(self, gamma: float, lambda1: float, lambda2: float, eps: float = 1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.eps = float(eps)
        self.log_eps = math.log(self.eps)

    def compute_components(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 2:
            raise AssertionError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise AssertionError(f"Expected targets with shape [B], got {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise AssertionError("Logits batch size and target batch size must match.")

        # log_softmax is numerically stable and prevents direct log(softmax(.)).
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().clamp(min=self.eps, max=1.0)

        gather_index = targets.unsqueeze(1)
        target_log_probs = log_probs.gather(dim=1, index=gather_index).squeeze(1)
        target_log_probs = torch.clamp(target_log_probs, min=self.log_eps)
        pt = target_log_probs.exp().clamp(min=self.eps, max=1.0)

        ce_loss = -target_log_probs

        # Focal term down-weights already easy samples and emphasizes mistakes.
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        focal_loss = -focal_weight * target_log_probs

        # Entropy quantifies how spread-out the prediction is across classes.
        entropy = -(probs * probs.log()).sum(dim=1)

        # Adaptive confusion penalizes high-entropy predictions more when the
        # model is not confident in the correct class (pt is small).
        confusion_loss = (1.0 - pt) * entropy

        total_loss = ce_loss + self.lambda1 * focal_loss + self.lambda2 * confusion_loss

        return {
            "ce": ce_loss.mean(),
            "focal": focal_loss.mean(),
            "confusion": confusion_loss.mean(),
            "entropy": entropy.mean(),
            "pt": pt.mean(),
            "total_loss": total_loss.mean(),
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_components(logits, targets)["total_loss"]


# ============================================================
# Train Loop
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, num_epochs: int):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))


def save_json(payload: Dict, save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def batch_to_device(
    images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def assert_batch_shapes(
    images: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> None:
    if images.ndim != 4:
        raise AssertionError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")
    if labels.ndim != 1:
        raise AssertionError(f"Expected labels with shape [B], got {tuple(labels.shape)}")
    if images.size(0) != labels.size(0):
        raise AssertionError("Image batch size and label batch size must match.")
    if images.size(1) != 3:
        raise AssertionError("InceptionV3 expects three input channels.")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise AssertionError("Labels contain values outside the expected class range.")


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Detected non-finite values in {name}.")


def validate_loss_value(loss: torch.Tensor, max_loss_threshold: float) -> None:
    if not torch.isfinite(loss):
        raise ValueError("Loss became NaN or Inf.")
    if loss.detach().item() > max_loss_threshold:
        raise ValueError(
            f"Loss exceeded the safety threshold ({max_loss_threshold:.1f}). "
            "Training was stopped to avoid runaway updates."
        )


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_acc: float,
    hyperparams: Dict[str, float],
    cfg: ExperimentConfig,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "hyperparameters": hyperparams,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, save_path)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdaptiveHybridLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    cfg: ExperimentConfig,
    epoch: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_samples = 0
    running = {
        "loss": 0.0,
        "ce": 0.0,
        "focal": 0.0,
        "confusion": 0.0,
        "correct": 0,
    }

    progress = tqdm(
        loader,
        desc=f"Train {epoch:02d}",
        leave=False,
        unit="batch",
    )

    for images, labels in progress:
        assert_batch_shapes(images, labels, cfg.num_classes)
        images, labels = batch_to_device(images, labels, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            logits, aux_logits = extract_logits(outputs)

            assert_finite_tensor("train_logits", logits)
            components = criterion.compute_components(logits, labels)
            loss = components["total_loss"]

            if aux_logits is not None:
                aux_components = criterion.compute_components(aux_logits, labels)
                loss = loss + cfg.aux_loss_weight * aux_components["total_loss"]

        validate_loss_value(loss, cfg.max_loss_threshold)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        preds = logits.argmax(dim=1)

        total_samples += batch_size
        running["loss"] += loss.detach().item() * batch_size
        running["ce"] += components["ce"].detach().item() * batch_size
        running["focal"] += components["focal"].detach().item() * batch_size
        running["confusion"] += components["confusion"].detach().item() * batch_size
        running["correct"] += (preds == labels).sum().item()

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            acc=f"{running['correct'] / max(1, total_samples):.4f}",
        )

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": running["loss"] / total_samples,
        "ce": running["ce"] / total_samples,
        "focal": running["focal"] / total_samples,
        "confusion": running["confusion"] / total_samples,
        "accuracy": running["correct"] / total_samples,
    }


# ============================================================
# Validation Loop
# ============================================================

@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdaptiveHybridLoss,
    device: torch.device,
    cfg: ExperimentConfig,
    epoch: int,
    use_amp: bool,
    collect_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    total_samples = 0
    running = {
        "loss": 0.0,
        "ce": 0.0,
        "focal": 0.0,
        "confusion": 0.0,
        "correct": 0,
    }
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"Val   {epoch:02d}",
        leave=False,
        unit="batch",
    )

    for images, labels in progress:
        assert_batch_shapes(images, labels, cfg.num_classes)
        images, labels = batch_to_device(images, labels, device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            logits, _ = extract_logits(outputs)
            assert_finite_tensor("val_logits", logits)
            components = criterion.compute_components(logits, labels)
            loss = components["total_loss"]

        validate_loss_value(loss, cfg.max_loss_threshold)

        batch_size = images.size(0)
        preds = logits.argmax(dim=1)

        total_samples += batch_size
        running["loss"] += loss.detach().item() * batch_size
        running["ce"] += components["ce"].detach().item() * batch_size
        running["focal"] += components["focal"].detach().item() * batch_size
        running["confusion"] += components["confusion"].detach().item() * batch_size
        running["correct"] += (preds == labels).sum().item()

        if collect_predictions:
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            acc=f"{running['correct'] / max(1, total_samples):.4f}",
        )

    results: Dict[str, object] = {
        "loss": running["loss"] / total_samples,
        "ce": running["ce"] / total_samples,
        "focal": running["focal"] / total_samples,
        "confusion": running["confusion"] / total_samples,
        "accuracy": running["correct"] / total_samples,
    }
    if collect_predictions:
        results["labels"] = np.concatenate(all_labels)
        results["preds"] = np.concatenate(all_preds)
    return results


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: AdaptiveHybridLoss,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    cfg: ExperimentConfig,
    hyperparams: Dict[str, float],
    logger: logging.Logger,
    checkpoint_path: Optional[Path] = None,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    use_amp = device.type == "cuda" and not cfg.disable_amp
    scaler = GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "train_ce": [],
        "train_focal": [],
        "train_confusion": [],
        "val_ce": [],
        "val_focal": [],
        "val_confusion": [],
        "learning_rate": [],
    }

    best_val_acc = -float("inf")
    best_state_dict = None

    epoch_bar = tqdm(
        range(1, cfg.num_epochs + 1),
        desc="Epochs",
        leave=True,
        unit="epoch",
    )

    for epoch in epoch_bar:
        train_stats = run_train_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            use_amp=use_amp,
        )
        val_stats = run_validation_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            cfg=cfg,
            epoch=epoch,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_stats["loss"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["train_ce"].append(train_stats["ce"])
        history["train_focal"].append(train_stats["focal"])
        history["train_confusion"].append(train_stats["confusion"])
        history["val_ce"].append(val_stats["ce"])
        history["val_focal"].append(val_stats["focal"])
        history["val_confusion"].append(val_stats["confusion"])
        history["learning_rate"].append(current_lr)

        epoch_bar.set_postfix(
            train_loss=f"{train_stats['loss']:.4f}",
            val_acc=f"{val_stats['accuracy']:.4f}",
            lr=f"{current_lr:.2e}",
        )

        logger.info(
            (
                "Epoch %02d/%02d | "
                "Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | LR: %.6e | "
                "Train[CE=%.4f, Focal=%.4f, Conf=%.4f] | "
                "Val[CE=%.4f, Focal=%.4f, Conf=%.4f]"
            ),
            epoch,
            cfg.num_epochs,
            train_stats["loss"],
            train_stats["accuracy"],
            val_stats["loss"],
            val_stats["accuracy"],
            current_lr,
            train_stats["ce"],
            train_stats["focal"],
            train_stats["confusion"],
            val_stats["ce"],
            val_stats["focal"],
            val_stats["confusion"],
        )

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = float(val_stats["accuracy"])
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                save_checkpoint(
                    save_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_acc=best_val_acc,
                    hyperparams=hyperparams,
                    cfg=cfg,
                )

        if cfg.save_every_epoch and checkpoint_path is not None:
            epoch_checkpoint = checkpoint_path.parent / f"epoch_{epoch:02d}.pth"
            save_checkpoint(
                save_path=epoch_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_acc=float(val_stats["accuracy"]),
                hyperparams=hyperparams,
                cfg=cfg,
            )

        if trial is not None:
            trial.report(best_val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {epoch} with best val acc {best_val_acc:.4f}."
                )

        if early_stopping.step(float(val_stats["accuracy"])):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a valid best checkpoint.")

    model.load_state_dict(best_state_dict)
    return model, history, best_val_acc


# ============================================================
# Optuna Objective
# ============================================================

def save_study_results(study: optuna.Study, output_dir: Path) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_study_results.csv", index=False)

    best_payload = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    save_json(best_payload, output_dir / "optuna_best_params.json")


def build_trial_hyperparameters(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "gamma": trial.suggest_float("gamma", 1.0, 5.0),
        "lambda1": trial.suggest_float("lambda1", 0.05, 1.0),
        "lambda2": trial.suggest_float("lambda2", 0.001, 0.1, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
    }


def objective(
    trial: optuna.Trial,
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    logger: logging.Logger,
) -> float:
    hyperparams = build_trial_hyperparameters(trial)
    logger.info("Starting Optuna trial %d with params: %s", trial.number, hyperparams)

    model = build_model(cfg).to(device)
    criterion = AdaptiveHybridLoss(
        gamma=hyperparams["gamma"],
        lambda1=hyperparams["lambda1"],
        lambda2=hyperparams["lambda2"],
        eps=cfg.eps,
    )
    optimizer = build_optimizer(
        model=model,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg.num_epochs)

    try:
        _, _, best_val_acc = train_model(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            hyperparams=hyperparams,
            logger=logger,
            checkpoint_path=None,
            trial=trial,
        )
    except ValueError as error:
        message = str(error).lower()
        if "loss became" in message or "safety threshold" in message or "non-finite" in message:
            raise optuna.TrialPruned(str(error)) from error
        raise

    logger.info("Finished Optuna trial %d with best val acc %.4f", trial.number, best_val_acc)
    return best_val_acc


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdaptiveHybridLoss,
    device: torch.device,
    cfg: ExperimentConfig,
    split_name: str,
) -> Dict[str, object]:
    use_amp = device.type == "cuda" and not cfg.disable_amp
    epoch_index = cfg.num_epochs if split_name != "test" else cfg.num_epochs + 1
    return run_validation_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        cfg=cfg,
        epoch=epoch_index,
        use_amp=use_amp,
        collect_predictions=True,
    )


def summarize_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> None:
    report_text = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    save_json(report_dict, output_dir / "classification_report.json")


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    if not history["train_loss"]:
        return

    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Adaptive Hybrid Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_heatmap(
    matrix: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: bool = True,
) -> None:
    sns.set_theme(style="white")
    plot_matrix = matrix.astype(np.float64)
    if normalize:
        row_sums = plot_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_matrix = plot_matrix / row_sums

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        plot_matrix,
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Normalized Test Confusion Matrix" if normalize else "Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bundle(
    output_dir: Path,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, float],
    best_val_accuracy: float,
) -> None:
    payload = {
        "best_val_accuracy": best_val_accuracy,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_hyperparameters": best_params,
    }
    save_json(payload, output_dir / "test_metrics.json")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    prepare_output_dir(cfg.output_dir)
    logger = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)
    device = get_device(logger)

    logger.info("Experiment profile: %s", cfg.profile)
    logger.info("Adaptive-loss outputs will be saved to: %s", cfg.output_dir)
    logger.info("Configuration: %s", json.dumps(asdict(cfg), indent=2, default=str))

    ensure_split_folders(cfg.data_dir, cfg.num_classes, logger)
    loaders = create_data_loaders(cfg, device, logger)
    class_names = loaders["test"].dataset.class_names

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=cfg.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    logger.info(
        "Starting Optuna search with %d trial(s) and %d epoch(s) per trial.",
        cfg.optuna_trials,
        cfg.num_epochs,
    )
    study.optimize(
        lambda trial: objective(trial, cfg, loaders, device, logger),
        n_trials=cfg.optuna_trials,
    )
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(
            "Optuna did not finish any complete trials. Try lowering the batch size, "
            "reducing the learning-rate search range manually, or re-running with the quick profile."
        )
    save_study_results(study, cfg.output_dir)

    best_params = {
        "gamma": float(study.best_params["gamma"]),
        "lambda1": float(study.best_params["lambda1"]),
        "lambda2": float(study.best_params["lambda2"]),
        "learning_rate": float(study.best_params["learning_rate"]),
    }
    logger.info("Best Optuna params: %s", best_params)
    logger.info("Best validation accuracy from tuning: %.4f", study.best_value)

    model = build_model(cfg).to(device)
    criterion = AdaptiveHybridLoss(
        gamma=best_params["gamma"],
        lambda1=best_params["lambda1"],
        lambda2=best_params["lambda2"],
        eps=cfg.eps,
    )
    optimizer = build_optimizer(model, best_params["learning_rate"], cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.num_epochs)
    best_model_path = cfg.output_dir / "best_model.pth"

    logger.info("Training final model with best Optuna hyperparameters.")
    model, history, best_val_accuracy = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        hyperparams=best_params,
        logger=logger,
        checkpoint_path=best_model_path,
        trial=None,
    )

    plot_training_curves(history, cfg.output_dir / "training_curves.png")

    val_results = evaluate_split(
        model=model,
        loader=loaders["val"],
        criterion=criterion,
        device=device,
        cfg=cfg,
        split_name="val",
    )
    test_results = evaluate_split(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        cfg=cfg,
        split_name="test",
    )

    val_labels = val_results["labels"]
    val_preds = val_results["preds"]
    test_labels = test_results["labels"]
    test_preds = test_results["preds"]

    val_metrics = summarize_metrics(val_labels, val_preds)
    test_metrics = summarize_metrics(test_labels, test_preds)

    matrix = confusion_matrix(
        test_labels,
        test_preds,
        labels=list(range(cfg.num_classes)),
    )
    np.save(cfg.output_dir / "test_confusion_matrix.npy", matrix)
    plot_confusion_heatmap(
        matrix=matrix,
        class_names=class_names,
        save_path=cfg.output_dir / "test_confusion_matrix.png",
        normalize=True,
    )
    save_classification_report(test_labels, test_preds, class_names, cfg.output_dir)
    save_metrics_bundle(
        output_dir=cfg.output_dir,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        best_val_accuracy=best_val_accuracy,
    )

    logger.info("Saved best model to: %s", best_model_path)
    logger.info("Saved training curves to: %s", cfg.output_dir / "training_curves.png")
    logger.info("Saved Optuna results to: %s", cfg.output_dir / "optuna_study_results.csv")
    logger.info("Saved confusion matrix image to: %s", cfg.output_dir / "test_confusion_matrix.png")
    logger.info("Saved classification report to: %s", cfg.output_dir / "classification_report.txt")

    logger.info(
        (
            "Final Metrics | "
            "Validation Accuracy: %.4f | "
            "Test Accuracy: %.4f | "
            "Precision: %.4f | Recall: %.4f | F1-score: %.4f"
        ),
        val_metrics["accuracy"],
        test_metrics["accuracy"],
        test_metrics["precision_macro"],
        test_metrics["recall_macro"],
        test_metrics["f1_macro"],
    )


if __name__ == "__main__":
    main()
