"""
GPU性能检测和推荐配置脚本
用于检测GPU配置并推荐合适的训练参数
"""

import torch
import platform


def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return None
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # GB
        gpu_info.append({
            'index': i,
            'name': props.name,
            'total_memory': total_memory,
            'compute_capability': f"{props.major}.{props.minor}",
        })
    
    return gpu_info


def recommend_yolo_config(gpu_memory_gb):
    """根据GPU内存推荐YOLO配置"""
    if gpu_memory_gb >= 24:
        return {
            'model': 'yolov8l.pt 或 yolov8x.pt',
            'batch_size': 16,
            'img_size': 1024,
            'note': '顶级配置，适合获得最佳效果'
        }
    elif gpu_memory_gb >= 16:
        return {
            'model': 'yolov8m.pt',
            'batch_size': 16,
            'img_size': 640,
            'note': '平衡配置，推荐用于生产环境'
        }
    elif gpu_memory_gb >= 8:
        return {
            'model': 'yolov8s.pt',
            'batch_size': 16,
            'img_size': 640,
            'note': '推荐配置，适合大多数任务'
        }
    elif gpu_memory_gb >= 4:
        return {
            'model': 'yolov8n.pt',
            'batch_size': 8,
            'img_size': 640,
            'note': '轻量配置，速度快'
        }
    else:
        return {
            'model': 'yolov8n.pt',
            'batch_size': 4,
            'img_size': 416,
            'note': '最小配置'
        }


def recommend_classification_config(gpu_memory_gb):
    """根据GPU内存推荐分类器配置"""
    if gpu_memory_gb >= 24:
        return {
            'model': 'resnet101 或 efficientnet_b4',
            'batch_size': 64,
            'img_size': 384,
            'note': '大模型，更高精度'
        }
    elif gpu_memory_gb >= 16:
        return {
            'model': 'resnet50',
            'batch_size': 48,
            'img_size': 256,
            'note': '平衡配置'
        }
    elif gpu_memory_gb >= 8:
        return {
            'model': 'resnet50',
            'batch_size': 32,
            'img_size': 256,
            'note': '推荐配置'
        }
    elif gpu_memory_gb >= 4:
        return {
            'model': 'resnet18 或 mobilenet_v2',
            'batch_size': 16,
            'img_size': 224,
            'note': '轻量配置'
        }
    else:
        return {
            'model': 'mobilenet_v2',
            'batch_size': 8,
            'img_size': 224,
            'note': '最小配置'
        }


def main():
    print("="*70)
    print("🔍 FlyCropper GPU性能检测和配置推荐")
    print("="*70)
    
    # 系统信息
    print(f"\n📌 系统信息:")
    print(f"  - 操作系统: {platform.system()} {platform.release()}")
    print(f"  - Python版本: {platform.python_version()}")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - cuDNN版本: {torch.backends.cudnn.version()}")
        
        gpu_info = get_gpu_info()
        print(f"\n🎮 GPU信息:")
        for gpu in gpu_info:
            print(f"  [{gpu['index']}] {gpu['name']}")
            print(f"      显存: {gpu['total_memory']:.2f} GB")
            print(f"      计算能力: {gpu['compute_capability']}")
            
            # 当前GPU状态
            torch.cuda.set_device(gpu['index'])
            allocated = torch.cuda.memory_allocated(gpu['index']) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu['index']) / (1024**3)
            print(f"      已分配: {allocated:.2f} GB")
            print(f"      已预留: {reserved:.2f} GB")
        
        # 配置推荐
        main_gpu = gpu_info[0]
        gpu_memory = main_gpu['total_memory']
        
        print(f"\n📊 针对 {main_gpu['name']} ({gpu_memory:.2f} GB) 的配置推荐:")
        
        # YOLO检测配置
        yolo_config = recommend_yolo_config(gpu_memory)
        print(f"\n  🎯 目标检测 (YOLO):")
        print(f"     模型: {yolo_config['model']}")
        print(f"     批次大小: {yolo_config['batch_size']}")
        print(f"     图像尺寸: {yolo_config['img_size']}")
        print(f"     说明: {yolo_config['note']}")
        
        # 分类器配置
        cls_config = recommend_classification_config(gpu_memory)
        print(f"\n  🏷️  分类器:")
        print(f"     模型: {cls_config['model']}")
        print(f"     批次大小: {cls_config['batch_size']}")
        print(f"     图像尺寸: {cls_config['img_size']}")
        print(f"     说明: {cls_config['note']}")
        
        # 性能建议
        print(f"\n💡 性能优化建议:")
        if gpu_memory >= 8:
            print("  ✅ GPU性能充足，可以使用推荐配置")
            print("  ✅ 建议启用混合精度训练 (AMP) 以进一步提升速度")
        elif gpu_memory >= 4:
            print("  ⚠️  GPU性能一般，建议使用轻量模型")
            print("  ⚠️  如果显存不足，可以减小批次大小")
        else:
            print("  ⚠️  GPU性能较弱，建议使用CPU训练或云端GPU")
        
        # YOLO模型对比
        print(f"\n📈 YOLO模型性能对比:")
        print("  ┌─────────────┬──────────┬───────────┬─────────┬──────────┐")
        print("  │ 模型        │ 参数量   │ mAP50-95  │ 速度    │ 推荐场景 │")
        print("  ├─────────────┼──────────┼───────────┼─────────┼──────────┤")
        print("  │ YOLOv8n     │ 3.2M     │ 37.3      │ 最快    │ 快速验证 │")
        print("  │ YOLOv8s     │ 11.2M    │ 44.9      │ 快      │ ⭐推荐   │")
        print("  │ YOLOv8m     │ 25.9M    │ 50.2      │ 中等    │ 高精度   │")
        print("  │ YOLOv8l     │ 43.7M    │ 52.9      │ 慢      │ 最佳效果 │")
        print("  │ YOLOv8x     │ 68.2M    │ 53.9      │ 最慢    │ 研究用   │")
        print("  └─────────────┴──────────┴───────────┴─────────┴──────────┘")
        
        print(f"\n📌 果蝇检测任务建议:")
        print("  - 数据集规模: ~400张图像")
        print("  - 任务难度: 中等（单类别检测）")
        if gpu_memory >= 8:
            print("  - 最佳选择: YOLOv8s (平衡速度和精度)")
            print("  - 备选方案: YOLOv8m (追求更高精度)")
        else:
            print("  - 最佳选择: YOLOv8n (轻量快速)")
            print("  - 备选方案: YOLOv8s (如果显存允许)")
        
    else:
        print(f"\n⚠️  未检测到可用GPU")
        print(f"\n💡 建议:")
        print("  1. 安装CUDA和cuDNN")
        print("  2. 安装GPU版本的PyTorch:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  3. 或使用CPU训练（速度较慢）:")
        print("     - 推荐使用小模型: YOLOv8n, MobileNetV2")
        print("     - 减小批次大小: 4-8")
    
    print("\n" + "="*70)
    print("📝 使用方法:")
    print("  1. 根据推荐配置修改训练脚本中的参数")
    print("  2. 如果训练时显存不足，逐步减小批次大小")
    print("  3. 可以通过 watch -n 1 nvidia-smi 实时监控GPU使用情况")
    print("="*70)


if __name__ == "__main__":
    main()