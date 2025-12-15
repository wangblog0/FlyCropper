"""
Label Studio标注图像裁切程序
根据JSON标注文件裁切果蝇图像，并按分类保存到不同文件夹
"""

import json
import os
from pathlib import Path
from PIL import Image


def load_annotations(json_path):
    """加载Label Studio导出的JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_folders(base_path, labels):
    """创建输出文件夹"""
    for label in labels:
        folder_path = os.path.join(base_path, label)
        os.makedirs(folder_path, exist_ok=True)
        print(f"创建文件夹: {folder_path}")


def crop_and_save_images(annotations, base_path, project_root):
    """
    裁切图像并保存到对应分类文件夹
    
    参数:
        annotations: 标注数据列表
        base_path: 输出基础路径
        project_root: 项目根目录路径
    """
    # 统计信息
    stats = {}
    processed_count = 0
    
    # 跟踪每个基础文件名的裁切计数（用于处理重名文件）
    filename_counters = {}
    
    for task in annotations:
        # 获取图像路径（包含源文件夹信息）
        image_path = task['data']['image']
        
        # 构建完整的源图像路径
        full_image_path = os.path.join(project_root, image_path)
        
        # 检查图像是否存在
        if not os.path.exists(full_image_path):
            print(f"警告: 图像不存在 - {full_image_path}")
            continue
        
        # 提取纯文件名（不含路径）
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]
        
        # 加载原始图像
        try:
            img = Image.open(full_image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"错误: 无法打开图像 {full_image_path} - {e}")
            continue
        
        # 处理每个标注
        for annotation in task['annotations']:
            if annotation['was_cancelled']:
                continue
            
            # 处理标注中的每个矩形框
            for result in annotation['result']:
                if result['type'] != 'rectanglelabels':
                    continue
                
                # 获取标签
                labels = result['value']['rectanglelabels']
                if not labels:
                    continue
                
                label = labels[0]  # 取第一个标签
                
                # 获取矩形框坐标（百分比形式）
                x_percent = result['value']['x']
                y_percent = result['value']['y']
                width_percent = result['value']['width']
                height_percent = result['value']['height']
                
                # 转换为像素坐标
                x1 = int((x_percent / 100) * img_width)
                y1 = int((y_percent / 100) * img_height)
                x2 = int(((x_percent + width_percent) / 100) * img_width)
                y2 = int(((y_percent + height_percent) / 100) * img_height)
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)
                
                # 裁切图像
                try:
                    cropped_img = img.crop((x1, y1, x2, y2))
                    
                    # 获取或初始化该文件名的计数器
                    if base_filename not in filename_counters:
                        filename_counters[base_filename] = 0
                    
                    # 递增计数器
                    filename_counters[base_filename] += 1
                    crop_number = filename_counters[base_filename]
                    
                    # 生成输出文件名（使用全局递增的序号）
                    output_filename = f"{base_filename}_crop_{crop_number}.jpg"
                    output_path = os.path.join(base_path, label, output_filename)
                    
                    # 保存裁切后的图像
                    cropped_img.save(output_path, 'JPEG', quality=95)
                    
                    # 更新统计信息
                    stats[label] = stats.get(label, 0) + 1
                    processed_count += 1
                    
                    # 显示来源信息
                    source_folder = os.path.dirname(image_path)
                    print(f"已保存: {output_filename} (来源: {source_folder}/{image_filename}, 尺寸: {x2-x1}x{y2-y1})")
                    
                except Exception as e:
                    print(f"错误: 裁切图像失败 - {e}")
        
        img.close()
    
    return stats, processed_count


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent
    json_file = project_root / "project-1-at-2025-12-02-18-36-aa5ef72e.json"
    output_folder = project_root / "cropped_flies"
    
    print("=" * 60)
    print("果蝇图像裁切程序")
    print("=" * 60)
    print(f"JSON文件: {json_file}")
    print(f"项目根目录: {project_root}")
    print(f"输出文件夹: {output_folder}")
    print()
    
    # 加载标注数据
    print("正在加载标注数据...")
    annotations = load_annotations(json_file)
    print(f"共加载 {len(annotations)} 个标注任务")
    print()
    
    # 收集所有标签
    labels = set()
    for task in annotations:
        for annotation in task['annotations']:
            for result in annotation['result']:
                if result['type'] == 'rectanglelabels':
                    labels.update(result['value']['rectanglelabels'])
    
    print(f"发现的分类标签: {', '.join(sorted(labels))}")
    print()
    
    # 创建输出文件夹
    print("创建输出文件夹...")
    create_output_folders(output_folder, labels)
    print()
    
    # 裁切并保存图像
    print("开始处理图像...")
    print("-" * 60)
    stats, processed_count = crop_and_save_images(annotations, output_folder, project_root)
    print("-" * 60)
    print()
    
    # 显示统计信息
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总共处理: {processed_count} 个裁切区域")
    print("\n各分类统计:")
    for label, count in sorted(stats.items()):
        print(f"  {label}: {count} 张图像")
    print(f"\n输出目录: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
