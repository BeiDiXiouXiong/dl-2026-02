import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from models import SimpleCNN

# ================== 数据 ==================
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ================== 可视化 ==================
images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images[:16])
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/samples.png")
plt.close()

# ================== 设备 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 模型 ==================
model = SimpleCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================== 验证函数 ==================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return correct / total

# ================== 训练 ==================
best_acc = 0

train_losses = []
train_accs = []
test_accs = []

epochs = 5   # 🔥 建议跑5轮，结果更好看

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    total_loss = total_loss / total
    train_acc = correct / total
    test_acc = evaluate(model)

    # 保存记录（画图用）
    train_losses.append(total_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch}: Loss={total_loss:.3f}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best.pt")

print("Best Test Acc:", best_acc)

# ================== 训练曲线 ==================
os.makedirs("outputs", exist_ok=True)

# Loss 曲线
plt.figure()
plt.plot(train_losses)
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("outputs/loss_curve.png")
plt.close()

# Accuracy 曲线
plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("outputs/acc_curve.png")
plt.close()