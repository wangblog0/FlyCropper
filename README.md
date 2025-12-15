# 果蝇检测与翅膀分类系统

基于深度学习的两阶段方案：目标检测 + 分类

## 项目特点

- **两阶段方案**：先检测果蝇位置，再对检测到的果蝇进行长短翅分类
- **高精度**：解耦检测和分类，各自达到更好的效果
- **易扩展**：模块化设计，方便修改和扩展

## 环境要求

- Python 3.8+
- CUDA 11.x+ (可选，用于GPU加速)

## 安装

### 方法1：使用 pip 和 requirements.txt

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 方法2：使用 pyproject.toml (推荐)

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装项目及依赖
pip install -e .

# 或者安装所有依赖（包括开发工具）
pip install -e ".[all]"
```

## 使用流程

### 1. 数据准备

将COCO格式数据转换为检测和分类所需的格式：

```bash
python data_preparation.py
```

这将生成：
- `processed_data/detection/` - YOLO格式的检测数据
- `processed_data/classification/` - 裁剪后的分类数据

### 2. 训练检测模型

```bash
python train_detection.py
```

训练完成后，模型保存在 `runs/detect/fly_detector/weights/best.pt`

可选参数（在脚本中修改）：
- `model_name`: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
- `epochs`: 训练轮数，默认100
- `img_size`: 图像尺寸，默认640
- `batch_size`: 批次大小，默认16

### 3. 训练分类模型

```bash
python train_classification.py
```

训练完成后，模型保存在 `runs/classification/best_model.pth`

可选参数（在脚本中修改）：
- `model_name`: resnet18, resnet50, efficientnet_b0, mobilenet_v2
- `epochs`: 训练轮数，默认50
- `img_size`: 图像尺寸，默认224
- `batch_size`: 批次大小，默认32

### 4. 推理测试

#### 单张图像测试

```bash
python inference.py single data/flys/MVIMG_20251202_153157.jpg
```

#### 批量处理

```bash
python inference.py batch processed_data/detection/val/images
```

结果保存在 `inference_results/` 目录：
- `visualizations/` - 可视化结果
- `crops/` - 裁剪的果蝇图像（按类别分类）
- `results.json` - 详细检测结果
- `statistics.json` - 统计信息

## 项目结构

```
FlyCropper/
├── data/                          # 原始数据
│   ├── result.json               # COCO标注文件
│   ├── flys/                     # 图像目录1
│   ├── flys2/                    # 图像目录2
│   └── ...
├── processed_data/               # 处理后的数据
│   ├── detection/               # 检测数据（YOLO格式）
│   └── classification/          # 分类数据（裁剪图像）
├── runs/                         # 训练输出
│   ├── detect/                  # 检测模型
│   └── classification/          # 分类模型
├── inference_results/           # 推理结果
├── data_preparation.py          # 数据准备脚本
├── train_detection.py           # 检测训练脚本
├── train_classification.py      # 分类训练脚本
├── inference.py                 # 推理脚本
├── requirements.txt             # 依赖列表
├── pyproject.toml              # 项目配置
└── README.md                    # 本文件
```

## 数据集格式

### 输入格式（COCO）

```json
{
  "images": [
    {"id": 0, "file_name": "flys/xxx.jpg", "width": 3072, "height": 4096}
  ],
  "annotations": [
    {"id": 0, "image_id": 0, "category_id": 1, "bbox": [x, y, w, h]}
  ],
  "categories": [
    {"id": 0, "name": "Ambiguous"},
    {"id": 1, "name": "Long"},
    {"id": 2, "name": "Short"}
  ]
}
```

## 模型性能

### 检测模型（YOLOv8）
- 输入尺寸：640×640
- 检测类别：fly（所有果蝇）
- 评估指标：mAP50, mAP50-95, Precision, Recall

### 分类模型（ResNet50）
- 输入尺寸：224×224
- 分类类别：Long（长翅）、Short（短翅）
- 评估指标：Accuracy, Precision, Recall, F1-Score

## 常见问题

### 1. CUDA内存不足

减小 batch_size：
- 检测：16 → 8 或 4
- 分类：32 → 16 或 8

或使用更小的模型：
- 检测：yolov8n.pt (最小)
- 分类：resnet18 或 mobilenet_v2

### 2. 训练速度慢

- 确保安装了GPU版本的PyTorch
- 减小图像尺寸
- 使用更小的模型

### 3. 准确率不高

- 增加训练轮数
- 调整数据增强参数
- 尝试不同的模型架构
- 检查数据集质量和标注准确性

## 进阶使用

### 自定义训练参数

修改训练脚本中的参数，例如：

```python
# train_detection.py
trainer = FlyDetectionTrainer(
    data_yaml=data_yaml,
    model_name='yolov8m.pt',  # 使用中等模型
    epochs=150,               # 增加训练轮数
    img_size=1024,            # 增大图像尺寸
    batch_size=8,             # 减小批次
)
```

### 使用自定义数据集

修改 `data_preparation.py` 中的路径：

```python
coco_json_path = "your_data/annotations.json"
data_root = "your_data"
output_root = "your_processed_data"
```

### 导出ONNX模型

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/fly_detector/weights/best.pt')

# 导出为ONNX
model.export(format='onnx')
```

## 开发

### 代码格式化

```bash
pip install black isort
black .
isort .
```

### 运行测试

```bash
pip install pytest pytest-cov
pytest
```

## 许可证

MIT License

## 作者

FlyCropper Team

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://github.com/pytorch/vision)
