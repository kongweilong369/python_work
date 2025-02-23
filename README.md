项目文件结构
/project_root
│──/AlexNet模型训练
│   ├── Model.py          # AlexNet模型定义文件
│   └── train.py          # 模型训练文件
├── /static
│   ├── animate.js        # 动画效果的 JS 文件
│   └── style.css         # 页面样式文件
│
├── /templates
│   └── index.html        # 前端页面 HTML 文件
│
├── flask_sever.py        # Flask 应用主文件
├── AlexNet.pth           # 训练好的 AlexNet 模型权重文件
└── requirements.txt      # 项目依赖的 Python 包
    
采用的模型为Alexnet，数据集为cifar-10，分类类别有10种。
用户需要先运行train.py文件来获取训练好的AlexNet模型=>生成AlexNet.pth=>运行flask_sever.py来启动flask应用=>再打开的网页中继续图片推理与预测。
这段代码实现了一个简单的 Flask Web 应用，用户可以通过浏览器上传图像，服务器使用预训练的 AlexNet 模型对图像进行分类，并返回预测的类别名称。
