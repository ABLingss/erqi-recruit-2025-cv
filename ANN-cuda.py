import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# 定义网络
class ANN(nn.Module):
    def __init__(self, input_size=784, hidden_size=15, output_size=10):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x  # 不加 softmax，CrossEntropyLoss 会处理


# 加载数据
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


# 计算准确率
def computeAcc(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # 迁移到 GPU
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


if __name__ == "__main__":
    start_time = time.time()
    print(" [Init] 开始初始化参数和模型...")

    # ================= 超参数配置（可优化空间） =================
    input_size = 784
    hidden_size = 15
    output_size = 10
    learning_rate = 0.1
    batch_size = 64
    num_epochs = 10
    # =========================================================

    # GPU 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 数据
    train_loader, test_loader = loadData(batch_size=batch_size)

    # 模型
    model = ANN(input_size, hidden_size, output_size).to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 训练过程
    loss_list, acc_list = [], []
    acc_max, acc_max_epoch = 0.0, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)  # 迁移到 GPU
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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
    model_path = "mnist_ann_cuda.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[Info] 模型已保存为: {model_path}")

    # 绘制曲线
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
    plt.savefig("ANN_plot_cuda.png")
    plt.show()
