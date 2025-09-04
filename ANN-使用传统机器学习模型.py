import torch
from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

# ================= 数据加载 =================
def load_mnist_numpy():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transform, download=True)

    X_train = train_dataset.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28*28) / 255.0
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test

# ================= 传统 ML 模型 =================
def run_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "SVM", acc

def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "KNN", acc

def run_rf_progress(X_train, y_train, X_test, y_test, n_estimators=20):
    """逐树训练 RandomForest"""
    model = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)
    print(f"[Training] RandomForest 开始训练 {n_estimators} 棵树...")
    for i in tqdm(range(1, n_estimators+1), desc="Fitting RF"):
        model.set_params(n_estimators=i)  # ✅ 用 set_params
        model.fit(X_train, y_train)
    print("[Predicting] RandomForest 开始预测...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "RandomForest", acc

def run_logistic_progress(X_train, y_train, X_test, y_test, max_iter=50):
    """逐轮训练 LogisticRegression"""
    model = LogisticRegression(max_iter=1, solver='saga', warm_start=True)
    print(f"[Training] LogisticRegression 开始训练 {max_iter} 轮...")
    for i in tqdm(range(1, max_iter+1), desc="Fitting Logistic"):
        model.set_params(max_iter=i)  # ✅ 用 set_params
        model.fit(X_train, y_train)
    print("[Predicting] LogisticRegression 开始预测...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "LogisticRegression", acc

# ================= 绘图函数 =================
def plot_results(model_name, acc):
    plt.figure(figsize=(6, 4))
    plt.bar([model_name], [acc], color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} on MNIST (Acc={acc:.4f})")
    fname = f"MNIST_{model_name}.png"
    plt.savefig(fname)
    plt.show()
    print(f"[Saved] {fname}")

# ================= 主程序 =================
if __name__ == "__main__":
    start_time = time.time()
    print("[Init] 加载 MNIST 数据...")
    X_train, y_train, X_test, y_test = load_mnist_numpy()

    # 模型池
    model_pool = {
        "svm": run_svm,
        "knn": run_knn,
        "rf": run_rf_progress,
        "logistic": run_logistic_progress
    }

    # 选择模型
    choice = "knn"  # 可改成 "svm" / "knn" / "rf" / "logistic"
    print(f"[Info] 选择模型: {choice}")

    # 可调参数
    if choice == "rf":
        n_estimators = 15  # 少量树，演示用
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test, n_estimators=n_estimators)
    elif choice == "logistic":
        max_iter = 30  # 少量迭代
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test, max_iter=max_iter)
    else:
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test)

    print(f"[Result] {model_name} Test Accuracy: {acc:.4f}")

    # 绘图
    plot_results(model_name, acc)

    print(f"[Done] 总耗时: {time.time() - start_time:.2f} 秒")
