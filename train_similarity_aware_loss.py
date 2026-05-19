import argparse
import json
import logging
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints_similarity_loss"
DEFAULT_CE_CONFUSION_MATRIX = PROJECT_ROOT / "checkpoints" / "test_confusion_matrix.npy"
DEFAULT_SIMILARITY_MATRIX_PATH = PROJECT_ROOT / "similarity_matrix.npy"
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
    ce_confusion_matrix_path: Path
    similarity_matrix_path: Path
    profile: str
    num_classes: int = NUM_CLASSES
    image_size: int = IMAGE_SIZE
    batch_size: int = 16
    num_epochs: int = QUICK_PROFILE["epochs"]
    optuna_trials: int = QUICK_PROFILE["optuna_trials"]
    patience: int = QUICK_PROFILE["patience"]
    num_workers: int = 0 if platform.system() == "Windows" else 4
    dropout: float = 0.4
    aux_loss_weight: float = 0.4
    grad_clip_norm: float = 1.0
    min_delta: float = 1e-4
    max_loss_threshold: float = 1e4
    epsilon: float = 1e-7
    seed: int = 42
    disable_amp: bool = False
    pretrained: bool = True
    study_name: str = "similarity_aware_loss_study"
    save_every_epoch: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train InceptionV3 on Oxford Flowers-102 with a confusion-driven "
            "similarity-aware adaptive loss."
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
        help="Folder used for similarity-aware checkpoints, reports, and plots.",
    )
    parser.add_argument(
        "--ce-confusion-matrix",
        type=Path,
        default=DEFAULT_CE_CONFUSION_MATRIX,
        help="Path to the CE baseline confusion matrix (.npy).",
    )
    parser.add_argument(
        "--similarity-matrix-path",
        type=Path,
        default=DEFAULT_SIMILARITY_MATRIX_PATH,
        help="Path where the CE-derived similarity matrix (.npy) will be saved.",
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
        help="Mini-batch size. The default is conservative for RTX 4060 training.",
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
        default="similarity_aware_loss_study",
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
        ce_confusion_matrix_path=args.ce_confusion_matrix.resolve(),
        similarity_matrix_path=args.similarity_matrix_path.resolve(),
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
    logger = logging.getLogger("similarity_aware_training")
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

    logger.info("Split folders not found. Rebuilding train/val/test folders from Oxford metadata.")
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
# Similarity Matrix Creation / Loading
# ============================================================

def assert_square_matrix(name: str, matrix: np.ndarray, expected_size: int) -> None:
    if matrix.ndim != 2:
        raise AssertionError(f"{name} must be 2D, got shape {matrix.shape}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError(f"{name} must be square, got shape {matrix.shape}.")
    if matrix.shape[0] != expected_size:
        raise AssertionError(
            f"{name} expected shape [{expected_size}, {expected_size}], got {matrix.shape}."
        )


def load_ce_confusion_matrix(confusion_path: Path, num_classes: int) -> np.ndarray:
    if not confusion_path.exists():
        raise FileNotFoundError(
            f"CE baseline confusion matrix not found: {confusion_path}. "
            "Train or evaluate the CE baseline first so the similarity matrix can be derived from it."
        )
    matrix = np.load(confusion_path)
    assert_square_matrix("CE confusion matrix", matrix, num_classes)
    if not np.isfinite(matrix).all():
        raise ValueError("CE confusion matrix contains NaN or Inf values.")
    return matrix.astype(np.float32)


def build_similarity_matrix(confusion_matrix_array: np.ndarray) -> np.ndarray:
    """
    Convert a CE confusion matrix into a confusion-driven similarity matrix.

    Off-diagonal entries are scaled into [0, 1]. The diagonal is forced to 0 so
    only inter-class confusion contributes to the adaptive penalty.
    """

    similarity = confusion_matrix_array.astype(np.float32).copy()
    np.fill_diagonal(similarity, 0.0)

    max_value = float(similarity.max())
    if max_value > 0.0:
        similarity = similarity / max_value
    else:
        similarity = np.zeros_like(similarity, dtype=np.float32)

    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 0.0)

    if not np.isfinite(similarity).all():
        raise ValueError("Derived similarity matrix contains NaN or Inf values.")
    return similarity


def prepare_similarity_matrix(
    cfg: ExperimentConfig,
    class_names: List[str],
    logger: logging.Logger,
) -> np.ndarray:
    ce_confusion = load_ce_confusion_matrix(cfg.ce_confusion_matrix_path, cfg.num_classes)
    similarity = build_similarity_matrix(ce_confusion)

    cfg.similarity_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.similarity_matrix_path, similarity)
    np.save(cfg.output_dir / "similarity_matrix.npy", similarity)

    plot_heatmap(
        matrix=similarity,
        class_names=class_names,
        save_path=cfg.output_dir / "similarity_matrix_heatmap.png",
        title="Confusion-Derived Similarity Matrix",
        cmap="magma",
        normalize=False,
    )

    logger.info("Loaded CE confusion matrix from: %s", cfg.ce_confusion_matrix_path)
    logger.info("Saved similarity matrix to: %s", cfg.similarity_matrix_path)
    logger.info("Saved similarity heatmap to: %s", cfg.output_dir / "similarity_matrix_heatmap.png")
    return similarity


# ============================================================
# Model Creation
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
# Custom Loss Class
# ============================================================

class SimilarityAwareLoss(nn.Module):
    """
    Similarity-aware adaptive loss for fine-grained flower recognition:

        L = -log(pt) * (1 + lambda * S(y, y_hat) * (1 - pt) * I(y_hat != y))

    where:
    - pt is the probability assigned to the true class
    - y_hat is the argmax prediction
    - S(y, y_hat) is a confusion-derived similarity weight from the CE baseline
    - I(y_hat != y) activates the extra penalty only for misclassifications

    This keeps correct predictions equivalent to standard cross-entropy while
    increasing the penalty for historically confusing flower pairs.
    """

    def __init__(
        self,
        similarity_matrix: torch.Tensor,
        lambda_value: float,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise AssertionError(
                f"similarity_matrix must be square, got {tuple(similarity_matrix.shape)}"
            )
        self.register_buffer("similarity_matrix", similarity_matrix.float())
        self.lambda_value = float(lambda_value)
        self.epsilon = float(epsilon)

    def compute_components(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 2:
            raise AssertionError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise AssertionError(f"Expected targets with shape [B], got {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise AssertionError("Logits batch size and target batch size must match.")
        if logits.size(1) != self.similarity_matrix.size(0):
            raise AssertionError(
                "The logits class dimension does not match the similarity matrix size."
            )

        probs = F.softmax(logits, dim=1)
        if not torch.isfinite(probs).all():
            raise ValueError("Detected non-finite values in probabilities.")

        predictions = probs.argmax(dim=1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, self.epsilon, 1.0 - self.epsilon)

        similarity = self.similarity_matrix[targets, predictions]
        difficulty = 1.0 - pt
        incorrect = (predictions != targets).float()
        adaptive_weight = 1.0 + self.lambda_value * similarity * difficulty * incorrect
        sample_losses = -torch.log(pt) * adaptive_weight

        if not torch.isfinite(pt).all():
            raise ValueError("Detected non-finite values in pt.")
        if not torch.isfinite(sample_losses).all():
            raise ValueError("Detected non-finite values in similarity-aware loss.")

        return {
            "loss": sample_losses.mean(),
            "pt": pt.mean(),
            "similarity": similarity.mean(),
            "difficulty": difficulty.mean(),
            "adaptive_weight": adaptive_weight.mean(),
            "incorrect_rate": incorrect.mean(),
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_components(logits, targets)["loss"]


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


def get_peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))


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
    criterion: SimilarityAwareLoss,
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
        "correct": 0,
        "pt": 0.0,
        "similarity": 0.0,
        "difficulty": 0.0,
        "adaptive_weight": 0.0,
        "incorrect_rate": 0.0,
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
            loss = components["loss"]

            if aux_logits is not None:
                assert_finite_tensor("train_aux_logits", aux_logits)
                aux_components = criterion.compute_components(aux_logits, labels)
                loss = loss + cfg.aux_loss_weight * aux_components["loss"]

        validate_loss_value(loss, cfg.max_loss_threshold)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)

        total_samples += batch_size
        running["loss"] += loss.detach().item() * batch_size
        running["correct"] += (predictions == labels).sum().item()
        running["pt"] += components["pt"].detach().item() * batch_size
        running["similarity"] += components["similarity"].detach().item() * batch_size
        running["difficulty"] += components["difficulty"].detach().item() * batch_size
        running["adaptive_weight"] += components["adaptive_weight"].detach().item() * batch_size
        running["incorrect_rate"] += components["incorrect_rate"].detach().item() * batch_size

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            acc=f"{running['correct'] / max(1, total_samples):.4f}",
        )

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": running["loss"] / total_samples,
        "accuracy": running["correct"] / total_samples,
        "pt": running["pt"] / total_samples,
        "similarity": running["similarity"] / total_samples,
        "difficulty": running["difficulty"] / total_samples,
        "adaptive_weight": running["adaptive_weight"] / total_samples,
        "incorrect_rate": running["incorrect_rate"] / total_samples,
    }


# ============================================================
# Validation Loop
# ============================================================

@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SimilarityAwareLoss,
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
        "correct": 0,
        "pt": 0.0,
        "similarity": 0.0,
        "difficulty": 0.0,
        "adaptive_weight": 0.0,
        "incorrect_rate": 0.0,
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
            loss = components["loss"]

        validate_loss_value(loss, cfg.max_loss_threshold)

        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)

        total_samples += batch_size
        running["loss"] += loss.detach().item() * batch_size
        running["correct"] += (predictions == labels).sum().item()
        running["pt"] += components["pt"].detach().item() * batch_size
        running["similarity"] += components["similarity"].detach().item() * batch_size
        running["difficulty"] += components["difficulty"].detach().item() * batch_size
        running["adaptive_weight"] += components["adaptive_weight"].detach().item() * batch_size
        running["incorrect_rate"] += components["incorrect_rate"].detach().item() * batch_size

        if collect_predictions:
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            acc=f"{running['correct'] / max(1, total_samples):.4f}",
        )

    results: Dict[str, object] = {
        "loss": running["loss"] / total_samples,
        "accuracy": running["correct"] / total_samples,
        "pt": running["pt"] / total_samples,
        "similarity": running["similarity"] / total_samples,
        "difficulty": running["difficulty"] / total_samples,
        "adaptive_weight": running["adaptive_weight"] / total_samples,
        "incorrect_rate": running["incorrect_rate"] / total_samples,
    }
    if collect_predictions:
        results["labels"] = np.concatenate(all_labels)
        results["preds"] = np.concatenate(all_preds)
    return results


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: SimilarityAwareLoss,
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
        "train_pt": [],
        "val_pt": [],
        "train_adaptive_weight": [],
        "val_adaptive_weight": [],
        "train_similarity": [],
        "val_similarity": [],
        "learning_rate": [],
        "gpu_peak_memory_mb": [],
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
        peak_gpu_mb = get_peak_gpu_memory_mb(device)

        history["train_loss"].append(train_stats["loss"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["train_pt"].append(train_stats["pt"])
        history["val_pt"].append(val_stats["pt"])
        history["train_adaptive_weight"].append(train_stats["adaptive_weight"])
        history["val_adaptive_weight"].append(val_stats["adaptive_weight"])
        history["train_similarity"].append(train_stats["similarity"])
        history["val_similarity"].append(val_stats["similarity"])
        history["learning_rate"].append(current_lr)
        history["gpu_peak_memory_mb"].append(peak_gpu_mb)

        epoch_bar.set_postfix(
            train_loss=f"{train_stats['loss']:.4f}",
            val_acc=f"{val_stats['accuracy']:.4f}",
            lr=f"{current_lr:.2e}",
        )

        gpu_message = f"{peak_gpu_mb:.1f} MB" if device.type == "cuda" else "N/A"
        logger.info(
            (
                "Epoch %02d/%02d | "
                "Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | LR: %.6e | GPU Peak: %s | "
                "Train[pt=%.4f, sim=%.4f, diff=%.4f, w=%.4f] | "
                "Val[pt=%.4f, sim=%.4f, diff=%.4f, w=%.4f]"
            ),
            epoch,
            cfg.num_epochs,
            train_stats["loss"],
            train_stats["accuracy"],
            val_stats["loss"],
            val_stats["accuracy"],
            current_lr,
            gpu_message,
            train_stats["pt"],
            train_stats["similarity"],
            train_stats["difficulty"],
            train_stats["adaptive_weight"],
            val_stats["pt"],
            val_stats["similarity"],
            val_stats["difficulty"],
            val_stats["adaptive_weight"],
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
        "lambda_value": trial.suggest_float("lambda", 0.01, 0.2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def objective(
    trial: optuna.Trial,
    cfg: ExperimentConfig,
    loaders: Dict[str, DataLoader],
    similarity_matrix: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
) -> float:
    hyperparams = build_trial_hyperparameters(trial)
    logger.info("Starting Optuna trial %d with params: %s", trial.number, hyperparams)

    model = build_model(cfg).to(device)
    criterion = SimilarityAwareLoss(
        similarity_matrix=similarity_matrix,
        lambda_value=hyperparams["lambda_value"],
        epsilon=cfg.epsilon,
    ).to(device)
    optimizer = build_optimizer(
        model=model,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
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
    criterion: SimilarityAwareLoss,
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
# Visualization
# ============================================================

def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    if not history["train_loss"]:
        return

    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Similarity-Aware Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str,
    cmap: str = "Blues",
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
        cmap=cmap,
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
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
    save_json(payload, output_dir / "final_metrics.json")


def save_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    save_json(history, output_dir / "training_history.json")


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
    logger.info("Similarity-aware outputs will be saved to: %s", cfg.output_dir)
    logger.info("Configuration: %s", json.dumps(asdict(cfg), indent=2, default=str))

    ensure_split_folders(cfg.data_dir, cfg.num_classes, logger)
    loaders = create_data_loaders(cfg, device, logger)
    class_names = loaders["test"].dataset.class_names

    similarity_matrix_np = prepare_similarity_matrix(cfg, class_names, logger)
    similarity_matrix_tensor = torch.from_numpy(similarity_matrix_np)

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
        lambda trial: objective(
            trial=trial,
            cfg=cfg,
            loaders=loaders,
            similarity_matrix=similarity_matrix_tensor,
            device=device,
            logger=logger,
        ),
        n_trials=cfg.optuna_trials,
    )

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(
            "Optuna did not finish any complete trials. Try lowering the batch size, "
            "reducing the search range manually, or re-running with the quick profile."
        )
    save_study_results(study, cfg.output_dir)

    best_params = {
        "lambda_value": float(study.best_params["lambda"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "weight_decay": float(study.best_params["weight_decay"]),
    }
    logger.info("Best Optuna params: %s", best_params)
    logger.info("Best validation accuracy from tuning: %.4f", study.best_value)

    model = build_model(cfg).to(device)
    criterion = SimilarityAwareLoss(
        similarity_matrix=similarity_matrix_tensor,
        lambda_value=best_params["lambda_value"],
        epsilon=cfg.epsilon,
    ).to(device)
    optimizer = build_optimizer(
        model=model,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )
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

    save_history(history, cfg.output_dir)
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

    test_matrix = confusion_matrix(
        test_labels,
        test_preds,
        labels=list(range(cfg.num_classes)),
    )
    np.save(cfg.output_dir / "test_confusion_matrix.npy", test_matrix)

    plot_heatmap(
        matrix=test_matrix,
        class_names=class_names,
        save_path=cfg.output_dir / "test_confusion_matrix.png",
        title="Normalized Test Confusion Matrix",
        cmap="Blues",
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
    logger.info("Saved training history to: %s", cfg.output_dir / "training_history.json")
    logger.info("Saved Optuna results to: %s", cfg.output_dir / "optuna_study_results.csv")
    logger.info("Saved confusion matrix image to: %s", cfg.output_dir / "test_confusion_matrix.png")
    logger.info("Saved classification report to: %s", cfg.output_dir / "classification_report.txt")
    logger.info("Saved final metrics to: %s", cfg.output_dir / "final_metrics.json")

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
