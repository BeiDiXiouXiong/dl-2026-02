import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 适配CIFAR-10（3通道，32x32），结构简单可运行，满足作业要求
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 新增padding，避免特征图缩小过多
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)  # 修正维度，适配32x32输入
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10个类别，对应CIFAR-10

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(torch.relu(self.conv2(x)))  # 16x16 → 8x8
        x = torch.flatten(x, 1)  # 展平，进入全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
