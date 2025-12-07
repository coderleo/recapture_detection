# 翻拍检测项目代码详解 (深度学习入门版)

本文档旨在为深度学习初学者详细解释本项目 (`Recapture_Detection`) 的核心代码 `model.py`。我们将从整体流程、关键概念到代码细节逐一拆解，帮助你理解如何构建一个图像分类模型。

---

## 1. 项目目标与整体流程

**目标**：训练一个 AI 模型，能够区分一张图片是 **“真实拍摄的 (Real)”** 还是 **“对着屏幕翻拍的 (Fake)”**。这是一个典型的 **二分类 (Binary Classification)** 问题。

**核心流程**：
1.  **准备数据**：加载图片，并进行预处理（缩放、转换格式等）。
2.  **构建模型**：使用现成的强大模型（如 EfficientNet 或 ResNet）作为基础。
3.  **定义规则**：告诉模型什么是“对”，什么是“错”（损失函数），以及如何改进（优化器）。
4.  **训练循环**：让模型反复看图片，猜结果，算误差，改参数。
5.  **保存成果**：把训练好的模型参数保存下来，供后续使用。

---

## 2. 代码逐行详解 (`model.py`)

### 2.1 导入工具包

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
```

*   **`torch` (PyTorch)**: 类似于 Numpy，但能在 GPU 上运行，是深度学习的核心库。
*   **`torch.nn`**: 神经网络模块 (Neural Network)，包含各种层（如全连接层、卷积层）和损失函数。
*   **`torchvision`**: 计算机视觉工具包，包含常用数据集 (`datasets`)、经典模型 (`models`) 和图像处理工具 (`transforms`)。
*   **`DataLoader`**: 负责把图片打包成一个个小批次 (Batch) 喂给模型。

### 2.2 配置参数

```python
DATA_DIR = './dataset'  # 数据存放路径
BATCH_SIZE = 12         # 一次训练看多少张图
NUM_EPOCHS = 25         # 所有数据反复看多少轮
LEARNING_RATE = 0.001   # 学习率：模型参数更新的步子大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有显卡用显卡，没显卡用CPU
```

*   **Batch Size (批大小)**: 模型不是看一张图改一次参数，而是看一组（比如12张），算出平均误差后再改。这样更稳定，也更快。
*   **Epoch (轮数)**: 把所有训练图片完整过一遍叫一个 Epoch。通常需要几十个 Epoch 模型才能学好。

### 2.3 数据预处理 (Data Transforms)

这是深度学习中非常关键的一步。

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 数据增强：随机裁剪
        transforms.RandomHorizontalFlip(),                   # 数据增强：随机水平翻转
        transforms.RandomRotation(15),                       # 数据增强：随机旋转
        transforms.ColorJitter(brightness=0.1, ...),         # 数据增强：颜色抖动
        transforms.ToTensor(),                               # 核心：转为 Tensor 格式
        transforms.Normalize([0.485, ...], [0.229, ...])     # 核心：归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ...], [0.229, ...])
    ]),
}
```

*   **数据增强 (Data Augmentation)**:
    *   **原理**: 训练数据总是有限的。通过随机旋转、裁剪、变色，我们可以“造”出很多新图。
    *   **作用**: 防止模型“死记硬背”原图（过拟合），强迫它学习更本质的特征（比如摩尔纹），而不是记住某张图的背景。
*   **`ToTensor()`**: 把图片（像素值 0-255）转换成 PyTorch 能处理的张量 (Tensor)，数值范围变到 [0, 1]。
*   **`Normalize()`**: 把数值标准化（减去均值，除以方差）。这能让模型训练得更快、更稳。这里的数字是 ImageNet 数据集的统计值，是行业标准。

### 2.4 加载数据

```python
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
```

*   **`ImageFolder`**: 一个非常方便的工具。只要你的文件夹结构是 `root/class_A/xxx.jpg`，它就能自动识别类别。
*   **`shuffle=True`**: 每次训练前把数据打乱。这很重要，防止模型记住数据的顺序。

### 2.5 构建模型 (Transfer Learning / 迁移学习)

```python
# 加载预训练模型 (EfficientNet-B0)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# 修改最后一层
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2) # 输出改为 2 (Real/Fake)

model = model.to(DEVICE) # 搬到 GPU/CPU
```

*   **迁移学习 (Transfer Learning)**:
    *   **原理**: 我们不需要从零开始训练。EfficientNet 已经在 ImageNet（1000万张图）上训练过了，它已经学会了如何识别线条、纹理、形状。
    *   **做法**: 我们拿来用，只把最后负责分类的那一层（原本输出1000类）换掉，换成我们需要的 2 类。
    *   **好处**: 即使我们数据很少（几十张），也能训练出很好的效果。

### 2.6 损失函数与优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

*   **损失函数 (`criterion`)**: 衡量模型猜得有多“错”。分类问题常用 `CrossEntropyLoss`。
*   **优化器 (`optimizer`)**: 负责根据误差来更新模型参数。`Adam` 是目前最流行的优化器，聪明且收敛快。
*   **学习率衰减 (`scheduler`)**: 训练后期，模型已经接近最优解了，这时候步子要迈小一点，防止走过头。这里每 7 轮把学习率缩小 10 倍。

### 2.7 训练循环 (Training Loop)

这是代码最核心的部分。

```python
for epoch in range(NUM_EPOCHS):
    for phase in ['train', 'val']:
        # 1. 模式切换
        if phase == 'train':
            model.train()  # 启用 Dropout, BatchNorm 更新
        else:
            model.eval()   # 冻结 Dropout, BatchNorm

        # 2. 遍历数据
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad() # 梯度清零 (别忘了!)

            # 3. 前向传播 (Forward)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels) # 计算误差

                # 4. 反向传播 (Backward) - 只在训练时做
                if phase == 'train':
                    loss.backward() # 算梯度 (误差来源)
                    optimizer.step() # 更新参数

        # 5. 统计与保存
        # ... 计算 Loss 和 Acc ...
        # ... 如果是 val 阶段且准确率创新高，保存模型 ...
```

*   **`model.train()` vs `model.eval()`**: 某些层（如 Dropout）在训练和预测时的行为是不一样的，必须手动切换。
*   **`optimizer.zero_grad()`**: PyTorch 默认会累加梯度，所以每次更新前要把旧的梯度清空。
*   **`loss.backward()`**: 著名的 **反向传播算法**。它利用链式法则，算出每个参数对误差“贡献”了多少。
*   **`optimizer.step()`**: 根据算出的梯度，沿着误差下降的方向推一把参数。

---

## 3. 总结

这段代码实现了一个标准的深度学习图像分类任务：
1.  利用 **数据增强** 扩充有限的数据集。
2.  利用 **迁移学习** (EfficientNet) 站在巨人的肩膀上。
3.  利用 **验证集** 监控模型表现，防止过拟合。
4.  利用 **学习率衰减** 让模型收敛得更稳。

希望这份文档能帮你建立起对深度学习代码的直观认识！
