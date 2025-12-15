"""
果蝇翅膀分类训练脚本
使用ResNet进行长翅/短翅分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class FlyClassificationDataset(Dataset):
    """果蝇分类数据集"""
    def __init__(self, data_root, split='train', transform=None, exclude_ambiguous=True):
        """
        Args:
            data_root: 数据根目录
            split: 'train' 或 'val'
            transform: 数据转换
            exclude_ambiguous: 是否排除Ambiguous类别
        """
        self.data_root = Path(data_root) / split
        self.transform = transform
        
        # 类别映射
        if exclude_ambiguous:
            self.class_to_idx = {'Long': 0, 'Short': 1}
            self.classes = ['Long', 'Short']
        else:
            self.class_to_idx = {'Ambiguous': 0, 'Long': 1, 'Short': 2}
            self.classes = ['Ambiguous', 'Long', 'Short']
        
        # 加载图像路径
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_root / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"{split}集: {len(self.samples)} 个样本")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FlyClassifier:
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True):
        """
        Args:
            num_classes: 类别数量
            model_name: 模型名称 ('resnet18', 'resnet50', 'efficientnet_b0', etc.)
            pretrained: 是否使用预训练权重
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = self._create_model(pretrained)
        self.model = self.model.to(self.device)
        
        print(f"模型: {model_name}, 设备: {self.device}, 类别数: {num_classes}")
        
    def _create_model(self, pretrained):
        """创建模型"""
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
        
        return model
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, save_dir='runs/classification'):
        """训练模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print("\n=== 开始训练 ===")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                 'acc': f'{100.*train_correct/train_total:.2f}%'})
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # 验证阶段
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'model_name': self.model_name,
                    'num_classes': self.num_classes,
                }, save_dir / 'best_model.pth')
                print(f"✅ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        # 保存训练历史
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_history(history, save_dir)
        
        print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
        print(f"模型保存在: {save_dir / 'best_model.pth'}")
        
        return history
    
    def validate(self, val_loader, criterion=None):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def evaluate(self, test_loader, class_names, save_dir='runs/classification'):
        """详细评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        print("\n=== 评估模型 ===")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="评估中"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_dir = Path(save_dir)
        plt.savefig(save_dir / 'confusion_matrix.png')
        print(f"\n混淆矩阵保存在: {save_dir / 'confusion_matrix.png'}")
        
        return all_preds, all_labels
    
    def plot_history(self, history, save_dir):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'training_history.png')
        print(f"训练曲线保存在: {Path(save_dir) / 'training_history.png'}")
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {model_path}")
        print(f"最佳准确率: {checkpoint['best_acc']:.2f}%")


def main():
    # 配置参数
    data_root = "processed_data/classification"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    img_size = 224
    
    # 数据转换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = FlyClassificationDataset(data_root, 'train', train_transform, exclude_ambiguous=True)
    val_dataset = FlyClassificationDataset(data_root, 'val', val_transform, exclude_ambiguous=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建分类器
    classifier = FlyClassifier(num_classes=2, model_name='resnet50', pretrained=True)
    
    # 训练
    history = classifier.train(train_loader, val_loader, epochs=num_epochs, lr=learning_rate)
    
    # 评估
    classifier.evaluate(val_loader, class_names=['Long', 'Short'])


if __name__ == "__main__":
    main()
