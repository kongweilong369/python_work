import time
import torch
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from Model import AlexNet

# 定义数据预处理
transform = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../../../data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10("../../../data", train=False, download=True, transform=transform)

# 数据集长度
train_data_size = len(train_data)
print(f"训练数据集的长度为：{train_data_size}")
test_data_size = len(test_data)
print(f"测试数据集的长度为：{test_data_size}")

# 利用 DataLoader 加载数据
train_dataloader = DataLoader(train_data, 50, shuffle=True)
test_dataloader = DataLoader(test_data, 50, shuffle=True)

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = AlexNet().to(device)

# 创建损失函数和优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.1)

# 训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 7

# 用于记录损失和准确率
train_losses = []
test_losses = []
test_accuracies = []

# 训练过程
start_time = time.time()

for i in range(epoch):
    print(f"----------第{i + 1}轮训练开始----------")
    alexnet.train()  # 设置模型为训练模式

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = alexnet(imgs)  # 前向传播

        loss = loss_fn(outputs, targets)  # 计算损失

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 调优

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, loss: {loss.item()}")

    # 每个 epoch 结束后进行一次测试
    total_test_loss = 0
    total_accuracy = 0
    alexnet.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 测试阶段不需要计算梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = alexnet(imgs)  # 前向传播
            loss = loss_fn(outputs, targets)  # 计算测试损失

            total_test_loss += loss.item()  # 累加测试损失
            accuracy = (outputs.argmax(1) == targets).sum()  # 计算正确个数
            total_accuracy += accuracy.item()  # 累加正确个数

    # 记录损失和准确率
    train_losses.append(loss.item())
    test_losses.append(total_test_loss / len(test_dataloader))
    test_accuracies.append(total_accuracy / test_data_size)

    print(f"整体测试集上的loss为：{total_test_loss / len(test_dataloader)}")
    print(f"整体测试集上的正确率为：{total_accuracy / test_data_size}")

# 保存模型
print('Finished Training')
torch.save(alexnet.state_dict(), f"../../../GPU_train_模型保存处/1.深度卷积神经网络(AlexNet)/AlexNet.pth")
print("model was saved")

# 使用 Matplotlib 可视化
# 绘制训练和测试损失
plt.figure(figsize=(12, 5))

# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epoch + 1), test_losses, label='Test Loss', marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(range(1, epoch + 1), test_accuracies, label='Test Accuracy', marker='o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# 保存绘图为图片
plt.savefig("../../../GPU_train_模型保存处/1.深度卷积神经网络(AlexNet)/AlexNet_training_results.png")  # 指定保存路径和文件名
print("Training results saved as image.")

plt.show()  # 如果需要，可以显示图形
