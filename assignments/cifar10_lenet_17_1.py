"""
Assignment 17.1: Improving a CNN on CIFAR-10
Modified LeNet with 3 conv layers, ReLU, MaxPool, dropout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# =============================================================================
# 1. Data Preprocessing and Loading
# =============================================================================

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# =============================================================================
# 2. Modified LeNet (Question 1)
# =============================================================================
# - 3 conv layers: (3,16,5,2), (16,32,5,2), (32,64,3,1)
# - MaxPool 2x2 after each conv
# - ReLU instead of sigmoid
# - Dropout 0.5 after fc1 and fc2
# - Size: 32→16→8→4 → 64*4*4 = 1024

class LeNetModified(nn.Module):
    def __init__(self):
        super(LeNetModified, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected: 64*4*4 = 1024 → 120 → 84 → 10
        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv blocks: conv → ReLU → pool
        x = self.pool(self.relu(self.conv1(x)))   # 32→16
        x = self.pool(self.relu(self.conv2(x)))   # 16→8
        x = self.pool(self.relu(self.conv3(x)))   # 8→4

        x = x.view(-1, 64 * 4 * 4)

        # FC: ReLU + dropout after fc1 and fc2, raw logits from fc3
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# =============================================================================
# 3. Training and Evaluation
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNetModified().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 10
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
