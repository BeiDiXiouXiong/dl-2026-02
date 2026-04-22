import torch
import torchvision
import torchvision.transforms as transforms
from models import SimpleCNN

# 与train.py保持一致的transform，确保评估结果准确
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集（与train.py一致）
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 类别名称，用于输出更直观
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 设备配置（与train.py一致，自动适配GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和最优权重
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("checkpoints/best.pt"))  # 加载训练好的最优模型

# 开始评估
model.eval()
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        # 计算每个类别的准确率（用于实验报告分析）
        c = (pred == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 输出最终评估结果
final_acc = correct / total
print(f"==================== 最终评估结果 ====================")
print(f"测试集总准确率: {final_acc:.2%}")
print(f"最优模型路径: checkpoints/best.pt")
print(f"\n各类别准确率:")
for i in range(10):
    if class_total[i] > 0:
        print(f"  {classes[i]:12s}: {class_correct[i] / class_total[i]:.2%}")
    else:
        print(f"  {classes[i]:12s}: 无测试样本")
