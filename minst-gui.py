import tkinter as tk
from tkinter import messagebox, ttk
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import numpy as np
import os

# ---------- 1. 模型定义 ----------
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
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------- 2. 设备配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 3. 模型管理 ----------
class ModelManager:
    def __init__(self):
        self.models = {
            'ANN': {
                'class': ANN,
                'path': 'mnist_ann_cuda.pt',
                'name': '全连接神经网络 (ANN)'
            },
            'CNN': {
                'class': CNN,
                'path': 'mnist_cnn.pt',
                'name': '卷积神经网络 (CNN)'
            }
        }
        self.current_model = None
        self.current_model_name = None

    def load_model(self, model_key):
        if model_key not in self.models:
            raise ValueError(f"不支持的模型类型: {model_key}")
        
        model_info = self.models[model_key]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            available_models = [k for k, v in self.models.items() if os.path.exists(v['path'])]
            if available_models:
                available_str = ', '.join([self.models[k]['name'] for k in available_models])
                raise FileNotFoundError(
                    f"模型文件 {model_path} 不存在！\n"
                    f"请确保已运行相应的训练脚本。\n"
                    f"当前可用的模型: {available_str}"
                )
            else:
                raise FileNotFoundError(
                    f"未找到任何训练好的模型！\n"
                    f"请先运行 ANN-cuda.py 或 ANN-CNN.py 来训练模型。"
                )

        # 创建模型实例
        if model_key == 'ANN':
            model = ANN().to(device)
        else:  # CNN
            model = CNN().to(device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        self.current_model = model
        self.current_model_name = model_info['name']
        return model

# ---------- 4. Tkinter GUI ----------
class MnistGui:
    def __init__(self, root):
        self.root = root
        root.title("手写数字识别")
        self.canvas_width = 280
        self.canvas_height = 280
        self.inner_size = 28   # MNIST 尺寸

        # 模型管理器
        self.model_manager = ModelManager()
        self.current_model = None

        # 创建界面元素
        self.create_widgets()
        
        # 初始化默认模型
        self.load_available_models()

    def create_widgets(self):
        # 左侧画布
        self.canvas = tk.Canvas(self.root, width=self.canvas_width,
                                height=self.canvas_height, bg="black")
        self.canvas.grid(row=1, column=0, rowspan=6, padx=5, pady=5)

        # PIL 图像与画笔
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)

        # 顶部模型选择
        model_frame = tk.Frame(self.root)
        model_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                       state="readonly", width=20)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # 右边控制面板
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=1, column=1, padx=5, pady=5, sticky="n")

        tk.Button(control_frame, text="识  别", command=self.predict, 
                 height=2, width=15).pack(pady=2)
        tk.Button(control_frame, text="清  空", command=self.clear, 
                 height=2, width=15).pack(pady=2)

        # 状态显示
        self.status_label = tk.Label(self.root, text="状态: 未加载模型", 
                                   font=("Arial", 10), fg="red")
        self.status_label.grid(row=2, column=1, padx=5, pady=5)

        # 结果显示
        self.label_result = tk.Label(self.root, text="请写数字", 
                                   font=("Arial", 18))
        self.label_result.grid(row=3, column=1, padx=5, pady=5)

        self.label_prob = tk.Label(self.root, text="", font=("Arial", 10))
        self.label_prob.grid(row=4, column=1, padx=5, pady=5)

    def load_available_models(self):
        """加载可用的模型列表"""
        available_models = []
        model_names = {}
        
        for key, info in self.model_manager.models.items():
            if os.path.exists(info['path']):
                available_models.append(key)
                model_names[key] = info['name']
        
        if available_models:
            self.model_combo['values'] = [model_names[k] for k in available_models]
            self.model_combo.current(0)
            self.load_model(available_models[0])
        else:
            self.model_combo['values'] = ["无可用模型"]
            self.model_combo.current(0)
            self.status_label.config(text="状态: 无可用模型", fg="red")
            messagebox.showerror("错误", 
                "未找到任何训练好的模型！\n"
                "请先运行 ANN-cuda.py 或 ANN-CNN.py 来训练模型。")

    def load_model(self, model_key):
        """加载指定模型"""
        try:
            self.current_model = self.model_manager.load_model(model_key)
            self.status_label.config(
                text=f"状态: 已加载 {self.model_manager.current_model_name}", 
                fg="green"
            )
            self.clear()
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def on_model_change(self, event):
        """模型切换事件处理"""
        selected_name = self.model_var.get()
        for key, info in self.model_manager.models.items():
            if info['name'] == selected_name:
                self.load_model(key)
                break

    # 画线
    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # 笔刷半径
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    # 清空
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.label_result.config(text="请写数字")
        self.label_prob.config(text="")

    # 识别
    def predict(self):
        if self.current_model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return

        try:
            # 1. 缩放并居中
            img = self.image.resize((self.inner_size, self.inner_size), Image.Resampling.LANCZOS)
            img_np = np.array(img, dtype=np.float32) / 255.0
            # 2. 归一化：训练时做了 (x-0.5)/0.5
            img_np = (img_np - 0.5) / 0.5
            
            # 根据当前模型类型调整输入
            if isinstance(self.current_model, CNN):
                img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
            else:  # ANN
                img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
                img_tensor = img_tensor.view(-1, 784)

            with torch.no_grad():
                logits = self.current_model(img_tensor)
                prob = torch.softmax(logits, dim=1)
                pred = int(prob.argmax(dim=1).item())
                confidence = float(prob[0, pred].item())

            self.label_result.config(text=f"预测结果: {pred}")
            self.label_prob.config(text=f"置信度: {confidence:.2%}")
        except Exception as e:
            messagebox.showerror("识别失败", f"识别过程中出现错误:\n{str(e)}")

# ---------- 5. 启动 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = MnistGui(root)
    root.mainloop()