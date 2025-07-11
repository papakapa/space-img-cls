import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DATA_DIR = "../data/EuroSAT"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 10
IMG_SIZE = 64
NUM_WORKERS = 4
MODEL_PATH = "model/resnet18_eurosat.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count())

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1459, 0.1132, 0.1137])
])


dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

def split_ds(ds: datasets.ImageFolder):
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    return train_ds, val_ds

train_ds, valid_ds = split_ds(dataset)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, pin_memory=True)


# todo: update after testing concepts
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    return model

resnet18_model = get_model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18_model.parameters(), lr=LR)

def fit(model, train_dl, loss_func, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{EPOCHS}] Train Loss: {total_loss:.4f}")

def validate(model, valid_dl):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.3f}")
    return acc

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved in {path}")

