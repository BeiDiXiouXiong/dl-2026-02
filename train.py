import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from models import SimpleCNN

# ================== 数据（新增标准化，作业规范要求） ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化，提升准确率
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CIFAR-10 类别名称（用于错例可视化和结果解释）
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ================== 可视化（保留原有，无修改） ==================
images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images[:16])
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/samples.png")
plt.close()

# ================== 设备（保留原有，无修改） ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 模型（保留原有，无修改） ==================
model = SimpleCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ================== 验证函数（保留原有，无修改） ==================
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


# ================== 新增：错例可视化函数（作业必做，至少5张错例） ==================
def visualize_wrong_samples(model, save_path="outputs/wrong_samples.png"):
    model.eval()
    wrong_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            # 筛选错误预测的样本
            wrong_idx = (preds != labels).nonzero().squeeze()
            if wrong_idx.numel() > 0:
                wrong_imgs = images[wrong_idx[:5 - len(wrong_images)]]  # 只取前5张
                wrong_true = labels[wrong_idx[:5 - len(wrong_images)]]
                wrong_pred = preds[wrong_idx[:5 - len(wrong_images)]]

                wrong_images.extend(wrong_imgs.cpu())
                true_labels.extend(wrong_true.cpu())
                pred_labels.extend(wrong_pred.cpu())

            if len(wrong_images) >= 5:
                break

    # 绘制错例图
    grid = torchvision.utils.make_grid(wrong_images[:5], nrow=5)
    plt.figure(figsize=(12, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')

    # 添加标签说明（真实标签 + 预测标签）
    label_text = []
    for i in range(5):
        true_cls = classes[true_labels[i]]
        pred_cls = classes[pred_labels[i]]
        label_text.append(f"True: {true_cls}\nPred: {pred_cls}")

    # 为每张图添加标签
    for i in range(5):
        plt.text(32 + i * 64, -10, label_text[i], ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # 返回错例信息（用于实验报告）
    return true_labels[:5], pred_labels[:5], wrong_images[:5]


# ================== 训练（保留原有，新增错例可视化调用） ==================
best_acc = 0

train_losses = []
train_accs = []
test_accs = []

epochs = 5  # 建议跑5轮，结果更好看，满足作业≥3轮要求

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

    # 优化输出格式，更清晰（可选，不影响功能）
    print(f"Epoch {epoch + 1:2d} | Loss: {total_loss:.3f} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best.pt")

print("Best Test Acc:", best_acc)

# ================== 训练曲线（保留原有，无修改） ==================
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

# ================== 调用错例可视化（作业必做） ==================
true_labels, pred_labels, wrong_images = visualize_wrong_samples(model)
