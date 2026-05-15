
import os, urllib.request, tarfile, shutil
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found – install with: pip install optuna")



CFG = {
   
    "data_dir"      : "./flowers102",
    "checkpoint_dir": "./checkpoints",

  
    "num_classes"   : 102,
    "image_size"    : 299,        
    "batch_size"    : 32,
    "num_epochs"    : 20,
    "num_workers"   : 4,

    
    "lr"            : 1e-4,
    "weight_decay"  : 1e-4,
    "dropout"       : 0.4,
    "optimizer"     : "adam",    
    "scheduler"     : "cosine",  
    "freeze_layers" : False,     

    "n_trials"      : 20,        
    "hpo_epochs"    : 7,         
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}



URLS = {
    "images" : "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "labels" : "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    "splits" : "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
}

def download_dataset(data_dir: str):
  
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dest = data_dir / Path(url).name
        if not dest.exists():
            print(f"Downloading {name} …")
            urllib.request.urlretrieve(url, dest)

    tgz = data_dir / "102flowers.tgz"
    jpg_dir = data_dir / "jpg"
    if not jpg_dir.exists():
        print("Extracting images …")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(data_dir)

    print("Dataset ready.")
    return data_dir


def build_split_folders(data_dir: str):
   
    data_dir = Path(data_dir)
    jpg_dir  = data_dir / "jpg"

    labels_mat = scipy.io.loadmat(str(data_dir / "imagelabels.mat"))
    splits_mat = scipy.io.loadmat(str(data_dir / "setid.mat"))

    labels = labels_mat["labels"].flatten()          
    trnids = splits_mat["trnid"].flatten()           
    valid  = splits_mat["valid"].flatten()
    tstid  = splits_mat["tstid"].flatten()

    split_map = {}
    for i in trnids: split_map[i] = "train"
    for i in valid:  split_map[i] = "val"
    for i in tstid:  split_map[i] = "test"

    for split in ("train", "val", "test"):
        for cls in range(1, CFG["num_classes"] + 1):
            (data_dir / split / str(cls)).mkdir(parents=True, exist_ok=True)

    for img_id in range(1, len(labels) + 1):
        split = split_map[img_id]
        cls   = labels[img_id - 1]
        src   = jpg_dir / f"image_{img_id:05d}.jpg"
        dst   = data_dir / split / str(cls) / f"image_{img_id:05d}.jpg"
        if not dst.exists():
            shutil.copy(src, dst)

    print("Split folders created.")



class FlowerDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples   = []
        split_dir = Path(root) / split
        for cls_dir in sorted(split_dir.iterdir()):
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
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_loaders(data_dir, batch_size, image_size, num_workers):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FlowerDataset(data_dir, split,
                           transform=get_transforms(image_size, split == "train"))
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
        print(f"  {split:5s}: {len(ds):5d} images")
    return loaders



def build_model(num_classes: int, dropout: float, freeze_backbone: bool):
   
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

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
    name   = cfg["optimizer"]
    lr     = cfg["lr"]
    wd     = cfg["weight_decay"]

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name   = cfg["scheduler"]
    epochs = cfg["num_epochs"]

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch)
    elif name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return None   # "none"



def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    use_aux=True):
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
            loss   = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        if scheduler and hasattr(scheduler, "step") and \
                not isinstance(scheduler, optim.lr_scheduler.StepLR):
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds        = logits.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss    = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds         = logits.argmax(1)
        correct       += (preds == labels).sum().item()
        total         += imgs.size(0)

    return running_loss / total, correct / total



def train(cfg: dict, loaders: dict, trial=None):
    device    = cfg["device"]
    ckpt_dir  = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model     = build_model(cfg["num_classes"], cfg["dropout"],
                            cfg["freeze_layers"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    best_val_acc = 0.0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg["num_epochs"] + 1):

       
        if cfg["freeze_layers"] and epoch == cfg["num_epochs"] // 2 + 1:
            unfreeze_model(model)
            optimizer = build_optimizer(model, cfg)   # rebuild with all params
            scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer,
                                          scheduler, criterion, device)
        vl_loss, vl_acc = evaluate(model, loaders["val"], criterion, device)

        if isinstance(scheduler, optim.lr_scheduler.StepLR):
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{cfg['num_epochs']} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f}")

     
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

     
        if trial is not None:
            trial.report(vl_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()   

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return model, history, best_val_acc



def plot_history(history: dict, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved training curves → {save_path}")


def main():
    device = CFG["device"]
    print(f"Using device: {device}")

  
    download_dataset(CFG["data_dir"])
    build_split_folders(CFG["data_dir"])

  
    print("\nBuilding data loaders …")
    loaders = get_loaders(CFG["data_dir"], CFG["batch_size"],
                          CFG["image_size"], CFG["num_workers"])

   
    model.load_state_dict(
        torch.load(Path(CFG["checkpoint_dir"]) / "best_model.pth",
                   map_location=device))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

   
    plot_history(history)


if __name__ == "__main__":
    main()