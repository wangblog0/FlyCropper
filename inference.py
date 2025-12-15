import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FlyDetectionClassificationPipeline:
    """果蝇检测和分类端到端流程"""

    def __init__(self, detection_model_path, classification_model_path,
                 classification_model_name='resnet50', num_classes=2, conf_threshold=0.25):
        """
        Args:
            detection_model_path: 检测模型路径
            classification_model_path: 分类模型路径
            classification_model_name: 分类模型名称
            num_classes: 分类类别数
            conf_threshold: 检测置信度阈值
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold

        # 加载检测模型
        print("加载检测模型...")
        self.detection_model = YOLO(detection_model_path)

        # 加载分类模型
        print("加载分类模型...")
        self.classification_model = self._load_classification_model(
            classification_model_path, classification_model_name, num_classes
        )

        # 分类类别
        self.class_names = ['Long', 'Short']

        # 分类图像转换
        self.classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"✅ 模型加载完成！设备: {self.device}")

    def _load_classification_model(self, model_path, model_name, num_classes):
        """加载分类模型"""
        # 创建模型架构
        if model_name == 'resnet18':
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def detect_flies(self, image):
        """检测图像中的果蝇

        Args:
            image: PIL Image 或 numpy array

        Returns:
            boxes: 检测框 [[x1, y1, x2, y2, conf], ...]
        """
        results = self.detection_model(image, conf=self.conf_threshold, verbose=False)

        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2, conf])

        return np.array(boxes) if boxes else np.array([]).reshape(0, 5)

    def classify_fly(self, image, box):
        """对检测到的果蝇进行分类

        Args:
            image: PIL Image
            box: [x1, y1, x2, y2, conf]

        Returns:
            class_name: 类别名称
            confidence: 分类置信度
        """
        x1, y1, x2, y2 = map(int, box[:4])

        # 裁剪果蝇区域
        cropped = image.crop((x1, y1, x2, y2))

        # 转换图像
        img_tensor = self.classification_transform(cropped).unsqueeze(0).to(self.device)

        # 分类
        with torch.no_grad():
            outputs = self.classification_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_name = self.class_names[predicted.item()]
        confidence = confidence.item()

        return class_name, confidence

    def process_image(self, image_path, save_path=None, save_crops=False, crops_dir=None):
        """处理单张图像

        Args:
            image_path: 图像路径
            save_path: 结果保存路径
            save_crops: 是否保存裁剪的果蝇图像
            crops_dir: 裁剪图像保存目录

        Returns:
            results: [{'box': [x1, y1, x2, y2], 'det_conf': conf,
                      'class': class_name, 'cls_conf': conf}, ...]
        """
        # 读取图像
        image = Image.open(image_path).convert('RGB')

        # 检测果蝇
        boxes = self.detect_flies(image)

        results = []
        for i, box in enumerate(boxes):
            # 分类
            class_name, cls_conf = self.classify_fly(image, box)

            result = {
                'box': box[:4].tolist(),
                'det_conf': float(box[4]),
                'class': class_name,
                'cls_conf': float(cls_conf)
            }
            results.append(result)

            # 保存裁剪图像
            if save_crops and crops_dir:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = image.crop((x1, y1, x2, y2))
                crop_path = Path(crops_dir) / class_name / f"{Path(image_path).stem}_{i}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(crop_path)

        # 可视化并保存
        if save_path:
            self.visualize_results(image_path, results, save_path)

        return results

    def process_batch(self, image_dir, output_dir, save_crops=False):
        """批量处理图像

        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            save_crops: 是否保存裁剪图像
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        crops_dir = output_dir / 'crops' if save_crops else None
        if crops_dir:
            for class_name in self.class_names:
                (crops_dir / class_name).mkdir(parents=True, exist_ok=True)

        # 获取所有图像
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        all_results = {}
        stats = {'Long': 0, 'Short': 0}

        print(f"\n处理 {len(image_paths)} 张图像...")
        for img_path in tqdm(image_paths):
            save_path = vis_dir / img_path.name
            results = self.process_image(img_path, save_path, save_crops, crops_dir)

            all_results[img_path.name] = results

            # 统计
            for result in results:
                stats[result['class']] += 1

        # 保存结果
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print(f"\n✅ 处理完成！")
        print(f"结果保存在: {output_dir}")
        print(f"\n统计信息:")
        total = sum(stats.values())
        for class_name, count in stats.items():
            percentage = 100 * count / total if total > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # 保存统计
        with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return all_results, stats

    def visualize_results(self, image_path, results, save_path):
        """可视化检测和分类结果"""
        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)

        colors = {'Long': 'green', 'Short': 'red'}

        for result in results:
            x1, y1, x2, y2 = result['box']
            class_name = result['class']
            det_conf = result['det_conf']
            cls_conf = result['cls_conf']

            color = colors.get(class_name, 'blue')

            # 绘制检测框
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # 添加标签
            label = f"{class_name} {cls_conf:.2f}"
            ax.text(x1, y1-10, label, color='white', fontsize=10,
                   bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # 配置路径
    detection_model_path = "runs/detect/fly_detector/weights/best.pt"
    classification_model_path = "runs/classification/best_model.pth"

    # 创建推理流程
    pipeline = FlyDetectionClassificationPipeline(
        detection_model_path=detection_model_path,
        classification_model_path=classification_model_path,
        classification_model_name='resnet50',
        num_classes=2,
        conf_threshold=0.25
    )

    # 选择处理模式
    # 移除sys.argv相关逻辑，直接使用默认值，确保路径正确设置
    mode = 'batch' # 强制设置为批量处理模式

    if mode == 'single':
        # 单张图像测试
        image_path = "data/flys/MVIMG_20251202_153157.jpg" # 示例路径，可能需要根据实际情况修改
        output_path = "test_output.jpg"
        results = pipeline.process_image(image_path, save_path=output_path,
                                        save_crops=True, crops_dir='test_crops')

        print(f"\n检测到 {len(results)} 只果蝇:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['class']} (分类置信度: {result['cls_conf']:.3f}, "
                  f"检测置信度: {result['det_conf']:.3f})")

    else:
        # 批量处理
        image_dir = "/content/processed_data/detection/val/images"
        print(f"正在使用以下目录进行推理: {image_dir}")

        output_dir = "inference_results"

        all_results, stats = pipeline.process_batch(
            image_dir=image_dir,
            output_dir=output_dir,
            save_crops=True
        )


if __name__ == "__main__":
    main()
