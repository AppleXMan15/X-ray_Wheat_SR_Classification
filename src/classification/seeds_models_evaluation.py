import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from torchvision import models

import pandas as pd
import numpy as np
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16

TRAIN_EPOCH_NUM = 20
LABELS_PATH = "labels.xlsx"
SAVE_DIR = f"models"

# ================= DATASETS =================

# SR-models
SR_DIRS = {
    "LR": "grains",
    "SRCNN": "grains_SRCNN",
    "EDSR": "grains_EDSR",
    "ESRGAN": "grains_ESRGAN",
    "LDM": "grains_LDM",
    "DASR": "grains_DASR",
}

# ================= TRANSFORMS =================

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ================= DATASET =================

class GrainDataset(Dataset):
    def __init__(self, img_dir, df, transform):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        grain_id = int(row.iloc[0])

        # ===== FIX: LR vs SR naming =====
        first_file = os.listdir(self.img_dir)[0]
        if "grain_" in first_file:
            img_name = f"grain_{grain_id}.png"
        else:
            img_name = f"{grain_id:04d}.png"

        path = os.path.join(self.img_dir, img_name)

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row.iloc[1:].values.astype(float))

        return img, label

# ================= MODELS =================

def get_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_densenet():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model

def get_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model

def get_googlenet():
    model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_squeezenet():
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
    model.num_classes = NUM_CLASSES
    return model

def get_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model

models_dict = {
    "resnet": get_resnet,
    "densenet": get_densenet,
    "efficientnet": get_efficientnet,
    "googlenet": get_googlenet,
    "squeezenet": get_squeezenet,
    "vit": get_vit
}

# ================= LOAD MODEL =================

def load_model(path, builder):
    model = builder()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ================= EVALUATE =================

def evaluate(model, loader):
    model.eval()

    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # ===== FIND BEST THRESHOLDS =====
    best_thresholds = []

    for c in range(NUM_CLASSES):
        best_f1 = 0
        best_t = 0.5

        for t in np.linspace(0.2, 0.8, 17):
            preds = (all_probs[:, c] > t).float()

            f1 = f1_score(
                all_labels[:, c].cpu().numpy(),
                preds.cpu().numpy()
            )

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds.append(best_t)

    preds = torch.zeros_like(all_probs)

    for c in range(NUM_CLASSES):
        preds[:, c] = (all_probs[:, c] > best_thresholds[c]).float()

    all_preds = preds

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("Pred sum per class:", all_preds.sum(dim=0))

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return f1, precision, recall


# ================= LOAD DATA =================

df = pd.read_excel(LABELS_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ================= EVALUATION =================

results = []

for dataset_name, IMG_DIR in SR_DIRS.items():

    print(f"\n===== DATASET: {dataset_name} =====")

    val_loader = DataLoader(
        GrainDataset(IMG_DIR, val_df, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for model_name, builder in models_dict.items():

        model_path = os.path.join(SAVE_DIR, f"{TRAIN_EPOCH_NUM}_{dataset_name}_{model_name}.pth")

        model = load_model(model_path, builder)

        f1, precision, recall = evaluate(model, val_loader)

        print(f"{model_name:15} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}")

        results.append([dataset_name, model_name, f1, precision, recall])

# ================= TABLE =================

results_df = pd.DataFrame(
    results,
    columns=["Dataset", "Model", "F1", "Precision", "Recall"]
)

print("\n\n===== FINAL TABLE =====")
print(results_df)

# ================= STATISTICS =================

print("\n===== STD (variation across classifiers per SR) =====")
print(results_df.groupby("Dataset")["F1"].std())
