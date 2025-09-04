import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# ================= CNN 网络 =================
class CNN(nn.Module):
    """
    CNN 手写数字分类网络（用于 MNIST 数据集）

    网络结构：
    1. conv1 -> bn1 -> ReLU -> MaxPool1
        - 输入：28x28 灰度图 (1 channel)
        - 输出：32 个 14x14 特征图
        - 作用：提取低级特征并进行归一化和非线性映射
    2. conv2 -> bn2 -> ReLU -> MaxPool2
        - 输入：32 个 14x14 特征图
        - 输出：64 个 7x7 特征图
        - 作用：提取更高层次特征，缩小空间维度
    3. Flatten
        - 将 64x7x7 特征图展平为一维向量
    4. fc1 -> ReLU -> Dropout
        - 全连接层，128 个神经元
        - Dropout 用于防止过拟合
    5. fc2
        - 输出层，10 个神经元，对应 0-9 数字类别
        - CrossEntropyLoss 内部会应用 softmax
    
    可调参数：
    - dropout_rate: 控制 fc1 后的 Dropout 概率
    - num_classes: 输出类别数量，MNIST 为 10

    实现要求：
    1. 使用 nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.Dropout
    2. 输入图像大小固定为 28x28，灰度图
    3. 输出 logits，不需要 softmax
    4. 支持 GPU 训练，forward 中数据形状需保持正确
    5. Dropout 只在训练阶段生效
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN, self).__init__()
        # 第一卷积块：卷积 -> 批归一化 -> ReLU -> 最大池化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # 第二卷积块：卷积 -> 批归一化 -> ReLU -> 最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 前向传播
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x  # 输出 logits，不做 softmax



# ================= 数据加载 =================
def loadData(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ================= 计算准确率 =================
def computeAcc(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# ================= 主程序 =================
if __name__ == "__main__":
    start_time = time.time()
    print(" [Init] 开始初始化参数和模型...")

    # ================= 超参数 =================
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    dropout_rate = 0.5
    optimizer_type = 'Adam'  # 可选 'SGD', 'Adam', 'RMSProp'
    # ==========================================

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 数据
    train_loader, test_loader = loadData(batch_size=batch_size)

    # 模型
    model = CNN(dropout_rate=dropout_rate).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError("Unsupported optimizer type!")

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练
    loss_list, acc_list = [], []
    acc_max, acc_max_epoch = 0.0, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # 更新学习率

        avg_loss = running_loss / len(train_loader)
        acc_now = computeAcc(model, test_loader, device)
        loss_list.append(avg_loss)
        acc_list.append(acc_now)

        if acc_now > acc_max:
            acc_max = acc_now
            acc_max_epoch = epoch

        tqdm.write(f"[Result] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc_now:.4f}")

    print(" [Done] 模型训练完成")
    print(f"总训练用时: {time.time() - start_time:.2f} 秒")

    # 保存模型
    model_path = "mnist_cnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[Info] 模型已保存为: {model_path}")

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(len(loss_list)), loss_list, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(122)
    plt.plot(range(len(acc_list)), acc_list, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.annotate(f"Max acc: {acc_max:.4f}",
                 xy=(acc_max_epoch, acc_max),
                 xytext=(acc_max_epoch * 0.7, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)
    plt.savefig("CNN_plot.png")
    plt.show()
