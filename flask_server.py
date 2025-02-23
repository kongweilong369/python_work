from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

# 创建 Flask 应用
app = Flask(__name__)

# 自定义的 AlexNet 模型定义
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)  # 输出10个类别
        )

    def forward(self, x):
        return self.model(x)

# 加载自定义训练的AlexNet模型
model = AlexNet()
model.load_state_dict(torch.load('AlexNet.pth'))  # 加载训练好的模型权重
model.eval()  # 设置为评估模式

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 渲染前端 HTML 页面
@app.route('/')
def index():
    return render_template('index.html')  # 渲染模板文件

# 处理图片并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # 对图片进行预处理，并添加批次维度
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(image)  # 将图像输入模型
        _, predicted_class = torch.max(outputs, 1)  # 获取最大概率的类别索引

    # 获取类别名称
    predicted_class_name = classes[predicted_class.item()]

    return jsonify({'prediction': predicted_class_name})


# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
