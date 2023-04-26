import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *

# Load CIFAR-100 dataset
aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            # For CIFAR-100
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ]
)
tsfm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            # For CIFAR-100
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ]
)

trn_dataset = torchvision.datasets.CIFAR100(
    root="./data/datasets", train=True, download=True, transform=aug_tsfm
)
trn_dataloader = torch.utils.data.DataLoader(
    trn_dataset, batch_size=32, shuffle=True, num_workers=4
)

tst_dataset = torchvision.datasets.CIFAR100(
    root="./data/datasets", train=False, download=True, transform=tsfm
)
tst_dataloader = torch.utils.data.DataLoader(
    tst_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = SlimResNet18(n_classes=100, nf=20).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for __epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for __batch in trn_dataloader:
        inputs, labels = __batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {__epoch}, Loss: {running_loss / len(trn_dataloader)}")

    # Evaluate the model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for __batch in tst_dataloader:
            images, labels = __batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")
