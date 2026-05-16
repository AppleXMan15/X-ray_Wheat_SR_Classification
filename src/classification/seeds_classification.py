import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchvision import models

import pandas as pd
import numpy as np
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 20

# ===================== DATA PATHS =====================

# LR-images
IMG_DIR_LR = "grains"

# SR-models
SR_DIRS = {
    "LR": "grains",
    "SRCNN": "grains_SRCNN",
    "EDSR": "grains_EDSR",
    "ESRGAN": "grains_ESRGAN",
    "LDM": "grains_LDM",
    "DASR": "grains_DASR",
}

LABELS_PATH = "labels.xlsx"
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== TRANSFORMS =====================

train_transform = T.Compose([
    T.Resize((224, 224)),

    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.15, contrast=0.15),

    T.ToTensor(),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


# ===================== DATASET =====================

class GrainDataset(Dataset):
    """
    Dataset works with two name formats:
    LR:  grain_1.png ... grain_4500.png
    SR:  0001.png ... 4500.png
    """

    def __init__(self, img_dir, df, transform):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        grain_id = int(row.iloc[0])

        # ===== LR and SR use different names =====
        if "grain_" in os.listdir(self.img_dir)[0]:
            # LR-format (grain_1.png, grain_2.png ...)
            img_name = f"grain_{grain_id}.png"
        else:
            # SR-format (0001.png, 0002.png ...)
            img_name = f"{grain_id:04d}.png"

        path = os.path.join(self.img_dir, img_name)

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row.iloc[1:].values.astype(float))

        return img, label


# ===================== LOAD LABELS =====================

df = pd.read_excel(LABELS_PATH)


# ===================== MODELS =====================

def get_resnet():
    model = models.resnet18(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_efficientnet():
    model = models.efficientnet_b0(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def get_googlenet():
    model = models.googlenet(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.inception5b.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_squeezenet():
    model = models.squeezenet1_1(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
    model.num_classes = NUM_CLASSES
    return model


def get_densenet():
    model = models.densenet121(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model


def get_vit():
    model = models.vit_b_16(weights="DEFAULT")

    # freeze all
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the last blocks
    for param in model.encoder.layers[-2:].parameters():
        param.requires_grad = True
    for param in model.heads.parameters():
        param.requires_grad = True

    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model


# ===================== SAVE / LOAD =====================

def save_model(model, name):
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, name))


def load_model(name, builder):
    model = builder()
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, name), map_location=device))
    model.to(device)
    model.eval()
    return model


# ===================== TRAIN LOOP =====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)

        loss = bce * ((1 - pt) ** self.gamma)

        if self.alpha is not None:
            loss = loss * self.alpha

        return loss.mean()

def train_model(model, name, train_loader, val_loader):

    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    best_f1 = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "f1": [],
        "precision": [],
        "recall": []
    }

    for epoch in range(EPOCHS):

        # ================= TRAIN =================
        model.train()
        total_loss = 0

        # ===== DEBUG: classes distribution =====
        batch_labels_sum = torch.zeros(NUM_CLASSES)
        for imgs, labels in train_loader:
            batch_labels_sum += labels.sum(dim=0)
        print("\nTrain distribution:", batch_labels_sum)

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            if isinstance(outputs, tuple):
                main_out, aux1, aux2 = outputs

                loss = (
                        criterion(main_out, labels)
                        + 0.3 * criterion(aux1, labels)
                        + 0.3 * criterion(aux2, labels)
                )

                preds = main_out
            else:
                loss = criterion(outputs, labels)
                preds = outputs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        history["train_loss"].append(total_loss)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)

                if isinstance(outputs, tuple):
                    preds = outputs[0]
                else:
                    preds = outputs

                loss = criterion(preds, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(preds)
                thresholds = torch.tensor([0.5, 0.4, 0.6, 0.6, 0.8]).to(device)
                preds = (probs > thresholds).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # ===== DEBUG =====
        print("Predicted per class:", all_preds.sum(dim=0))
        print("(val) True per class:", all_labels.sum(dim=0))

        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        history["val_loss"].append(val_loss)
        history["f1"].append(f1)
        history["precision"].append(precision)
        history["recall"].append(recall)

        print(f"{name} | Epoch {epoch+1}: F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            if epoch <= 9:
                save_model(model, f"10_{name}.pth")
            else:
                save_model(model, f"20_{name}.pth")
            print("Weights saved")

    return history


# ===================== TABLE OUTPUT =====================

def print_table(results, dataset_name):
    print(f"\n================ {dataset_name} RESULTS ================\n")

    print(f"{'Model':15} {'F1':10} {'Precision':10} {'Recall':10}")
    print("-" * 55)

    for model_name, hist in results.items():
        print(f"{model_name:15} "
              f"{hist['f1'][-1]:.4f}      "
              f"{hist['precision'][-1]:.4f}      "
              f"{hist['recall'][-1]:.4f}")


# ===================== VISUALIZATION =====================

def plot_all(all_histories):
    """
    6 grafics:
    LR + 5 SR + metrics
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes = axes.flatten()

    i = 0

    for dataset_name, models_hist in all_histories.items():

        for model_name, hist in models_hist.items():

            axes[i].plot(hist["train_loss"], label="train")
            axes[i].plot(hist["val_loss"], label="val")
            axes[i].set_title(f"{dataset_name}-{model_name}")
            axes[i].legend()
            axes[i].grid()

            i += 1

            if i >= 6:
                break

    plt.tight_layout()
    plt.show()


# ===================== EXPERIMENT =====================

models_dict = {
    "resnet": get_resnet,
    "densenet": get_densenet,
    "efficientnet": get_efficientnet,
    "googlenet": get_googlenet,
    "squeezenet": get_squeezenet,
    "vit": get_vit
}

all_results = {}


for dataset_name, IMG_DIR in SR_DIRS.items():

    print(f"\n\n================ DATASET: {dataset_name} =================")

    # ===== fixed split for EACH dataset =====
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # ===== imbalance weights =====
    labels_array = train_df.iloc[:, 1:].values

    # a more rare class has bigger weight
    class_counts = labels_array.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)

    # each image weight = sum of weights of its classes
    sample_weights = (labels_array * class_weights).sum(axis=1)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    from torch.utils.data import WeightedRandomSampler

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        GrainDataset(IMG_DIR, train_df, train_transform),
        batch_size=BATCH_SIZE, sampler=sampler, shuffle=False
    )

    val_loader = DataLoader(
        GrainDataset(IMG_DIR, val_df, val_transform),
        batch_size=BATCH_SIZE, shuffle=False
    )

    pos_weights = (len(labels_array) - labels_array.sum(axis=0)) / labels_array.sum(axis=0)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float).to(device)

    criterion = FocalLoss()

    all_results[dataset_name] = {}

    # ===== training all classifiers on SAME dataset =====
    for model_name, builder in models_dict.items():

        print(f"\n----- {dataset_name} -> {model_name} -----")

        model = builder()

        history = train_model(
            model,
            f"{dataset_name}_{model_name}",
            train_loader,
            val_loader
        )

        all_results[dataset_name][model_name] = history


        # ===== console table per model =====
        print_table({model_name: history}, dataset_name)


# ===================== FINAL PLOTS =====================

plot_all(all_results)
