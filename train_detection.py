"""
果蝇目标检测训练脚本（YOLOv8）
使用YOLOv8进行果蝇检测（不区分类别）
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch


class FlyDetectionTrainer:
    def __init__(self, data_yaml, model_name='yolov8n.pt', epochs=100, img_size=640, batch_size=16):
        """
        Args:
            data_yaml: 数据配置文件路径
            model_name: 预训练模型名称
            epochs: 训练轮数
            img_size: 图像尺寸
            batch_size: 批次大小
        """
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        
    def train(self, project='runs/detect', name='fly_detector'):
        """训练目标检测模型"""
        print("=== 开始训练果蝇检测模型 ===")
        print(f"模型: {self.model_name}")
        print(f"数据配置: {self.data_yaml}")
        print(f"训练轮数: {self.epochs}")
        print(f"图像尺寸: {self.img_size}")
        print(f"批次大小: {self.batch_size}")
        print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # 加载预训练模型
        model = YOLO(self.model_name)
        
        # 训练参数
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            project=project,
            name=name,
            patience=20,  # 早停耐心值
            save=True,
            save_period=10,  # 每10个epoch保存一次
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            # 优化器
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # 其他
            cos_lr=True,
            label_smoothing=0.0,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            verbose=True,
        )
        
        print("\n✅ 训练完成！")
        print(f"最佳模型保存在: {Path(project) / name / 'weights' / 'best.pt'}")
        
        return results
    
    def validate(self, model_path, project='runs/detect', name='val'):
        """验证模型"""
        print("\n=== 验证模型 ===")
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            imgsz=self.img_size,
            batch=self.batch_size,
            project=project,
            name=name,
            plots=True,
        )
        
        print("\n验证结果:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        
        return results


def main():
    # 配置参数
    data_yaml = "processed_data/detection/data.yaml"
    
    # 创建训练器
    trainer = FlyDetectionTrainer(
        data_yaml=data_yaml,
        model_name='yolov8n.pt',  # 可选: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        epochs=100,
        img_size=640,
        batch_size=16,
    )
    
    # 训练模型
    results = trainer.train(project='runs/detect', name='fly_detector')
    
    # 验证最佳模型
    best_model_path = 'runs/detect/fly_detector/weights/best.pt'
    if Path(best_model_path).exists():
        trainer.validate(best_model_path, project='runs/detect', name='val_best')


if __name__ == "__main__":
    main()
