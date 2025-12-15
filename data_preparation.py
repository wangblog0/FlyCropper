"""
数据准备脚本
功能：
1. 从YOLO格式数据集读取并划分训练集和验证集
2. 为目标检测准备数据（所有类别统一为"fly"）
3. 为分类准备裁剪后的图像数据集（Long/Short/Ambiguous）
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
from tqdm import tqdm


class DataPreparation:
    def __init__(self, yolo_data_root, output_root, train_ratio=0.8, exclude_ambiguous=True):
        """
        Args:
            yolo_data_root: YOLO格式数据根目录（包含images/和labels/子目录）
            output_root: 输出目录
            train_ratio: 训练集比例
            exclude_ambiguous: 是否排除Ambiguous类别
        """
        self.yolo_data_root = Path(yolo_data_root)
        self.images_dir = self.yolo_data_root / 'images'
        self.labels_dir = self.yolo_data_root / 'labels'
        self.classes_file = self.yolo_data_root / 'classes.txt'
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.exclude_ambiguous = exclude_ambiguous
        
        # 读取类别
        with open(self.classes_file, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f if line.strip()]
        
        print(f"类别: {self.classes}")
        
        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png')))
        
        # 过滤出有对应标注文件的图像
        self.valid_images = []
        for img_path in self.image_files:
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                self.valid_images.append(img_path)
        
        print(f"加载数据集: {len(self.valid_images)} 张有效图片（共 {len(self.image_files)} 张）")
        
    def parse_yolo_label(self, label_path, img_width, img_height):
        """解析YOLO格式标注文件
        
        Args:
            label_path: 标注文件路径
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            boxes: [(class_id, x1, y1, x2, y2), ...]
        """
        boxes = []
        
        if not label_path.exists():
            return boxes
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为绝对坐标
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                
                boxes.append((class_id, x1, y1, x2, y2))
        
        return boxes
        
    def prepare_detection_data(self):
        """准备目标检测数据（YOLO格式，所有类别统一为fly）"""
        print("\n=== 准备目标检测数据 ===")
        
        # 创建输出目录
        detection_root = self.output_root / "detection"
        for split in ['train', 'val']:
            (detection_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (detection_root / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 划分训练集和验证集
        images = self.valid_images.copy()
        random.shuffle(images)
        train_size = int(len(images) * self.train_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]
        
        print(f"训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")
        
        # 处理训练集和验证集
        for split, img_list in [('train', train_images), ('val', val_images)]:
            print(f"\n处理{split}集...")
            for img_path in tqdm(img_list):
                # 复制图片
                output_img_path = detection_root / split / 'images' / img_path.name
                shutil.copy(img_path, output_img_path)
                
                # 读取并转换标注（所有类别统一为0）
                label_path = self.labels_dir / (img_path.stem + '.txt')
                output_label_path = detection_root / split / 'labels' / (img_path.stem + '.txt')
                
                with open(label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # 将所有类别改为0（统一为fly）
                            f_out.write(f"0 {' '.join(parts[1:])}\n")
        
        # 生成data.yaml配置文件
        yaml_content = f"""path: {detection_root.absolute()}
train: train/images
val: val/images

nc: 1
names: ['fly']
"""
        with open(detection_root / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ 目标检测数据准备完成！保存在: {detection_root}")
        return detection_root / 'data.yaml'
        
    def prepare_classification_data(self):
        """准备分类数据（裁剪后的图像，按类别分类）"""
        print("\n=== 准备分类数据 ===")
        
        # 创建输出目录
        classification_root = self.output_root / "classification"
        
        # 确定要处理的类别
        if self.exclude_ambiguous:
            target_classes = [c for c in self.classes if c != 'Ambiguous']
        else:
            target_classes = self.classes
        
        print(f"处理类别: {target_classes}")
        
        for split in ['train', 'val']:
            for category in target_classes:
                (classification_root / split / category).mkdir(parents=True, exist_ok=True)
        
        # 划分训练集和验证集（使用相同的划分）
        images = self.valid_images.copy()
        random.shuffle(images)
        train_size = int(len(images) * self.train_ratio)
        train_images = set(images[:train_size])
        
        # 处理所有图像
        print("裁剪并保存图像...")
        crop_counts = {'train': {}, 'val': {}}
        
        for img_path in tqdm(self.valid_images):
            # 确定是训练集还是验证集
            split = 'train' if img_path in train_images else 'val'
            
            # 读取图像
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"无法打开图像 {img_path}: {e}")
                continue
            
            # 读取标注
            label_path = self.labels_dir / (img_path.stem + '.txt')
            boxes = self.parse_yolo_label(label_path, img.width, img.height)
            
            # 裁剪每个标注框
            for idx, (class_id, x1, y1, x2, y2) in enumerate(boxes):
                category_name = self.classes[class_id]
                
                # 如果排除Ambiguous类别
                if self.exclude_ambiguous and category_name == 'Ambiguous':
                    continue
                
                # 确保边界在图像内
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img.width, int(x2))
                y2 = min(img.height, int(y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 裁剪
                cropped = img.crop((x1, y1, x2, y2))
                
                # 保存
                if category_name not in crop_counts[split]:
                    crop_counts[split][category_name] = 0
                crop_counts[split][category_name] += 1
                
                crop_filename = f"{img_path.stem}_{idx}_{category_name}.jpg"
                output_path = classification_root / split / category_name / crop_filename
                cropped.save(output_path, quality=95)
        
        # 打印统计信息
        print(f"\n✅ 分类数据准备完成！保存在: {classification_root}")
        print("\n数据集统计:")
        for split in ['train', 'val']:
            print(f"\n{split}集:")
            total = sum(crop_counts[split].values())
            for category, count in sorted(crop_counts[split].items()):
                percentage = 100 * count / total if total > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        return classification_root


def main():
    # 设置随机种子
    random.seed(42)
    
    # 配置路径
    yolo_data_root = "data_yolo"  # YOLO格式数据目录
    output_root = "processed_data"
    
    # 创建数据准备对象
    data_prep = DataPreparation(
        yolo_data_root=yolo_data_root,
        output_root=output_root,
        train_ratio=0.8,
        exclude_ambiguous=True  # 排除Ambiguous类别
    )
    
    # 准备目标检测数据
    yaml_path = data_prep.prepare_detection_data()
    
    # 准备分类数据
    classification_root = data_prep.prepare_classification_data()
    
    print("\n" + "="*60)
    print("✅ 数据准备完成！")
    print(f"检测数据配置: {yaml_path}")
    print(f"分类数据目录: {classification_root}")
    print("="*60)


if __name__ == "__main__":
    main()
