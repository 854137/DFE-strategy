import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
from pycocotools.coco import COCO


class DatasetAnalyzer:
    def __init__(self, yaml_path, use_val_set=False):
        """
        初始化数据集分析器，仅基于YAML配置文件
        """
        self.yaml_path = Path(yaml_path)
        self.dataset_path = self.yaml_path.parent  # YAML文件所在目录作为数据集根目录
        self.class_names = {}
        self.stats = {}
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.use_val_set = use_val_set  # 是否使用验证集
        self.invalid_bboxes = []  # 记录无效边界框信息

        # 从YAML中读取路径和类别信息
        self._load_yaml_config()

    def _load_yaml_config(self):
        """
        从YAML文件中加载配置信息
        """
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # 解析数据集根目录
        self.root_path = Path(yaml_data.get('path', self.yaml_path.parent))
        if not self.root_path.is_absolute():
            self.root_path = self.dataset_path / self.root_path

        # 解析训练、验证、测试路径
        self.train_path = self.root_path / yaml_data.get('train', 'train/images')
        self.test_path = self.root_path / yaml_data.get('test', 'test/images')
        self.val_path = self.root_path / yaml_data.get('val', 'val/images') if 'val' in yaml_data else self.test_path

        # 解析类别信息
        if 'names' in yaml_data:
            if isinstance(yaml_data['names'], list):
                self.class_names = {i: name for i, name in enumerate(yaml_data['names'])}
            elif isinstance(yaml_data['names'], dict):
                self.class_names = yaml_data['names']
        elif 'nc' in yaml_data and 'names' in yaml_data:
            # YOLOv5/v8格式: nc: 80, names: [class1, class2, ...]
            if isinstance(yaml_data['names'], list):
                self.class_names = {i: name for i, name in enumerate(yaml_data['names'])}

        # 如果没有找到类别名称，尝试从其他键获取
        if not self.stats:
            for key in ['names', 'classes', 'class_names']:
                if key in yaml_data:
                    if isinstance(yaml_data[key], list):
                        self.class_names = {i: name for i, name in enumerate(yaml_data[key])}
                    elif isinstance(yaml_data[key], dict):
                        self.class_names = yaml_data[key]
                    break

        print(f"数据集根目录: {self.root_path}")
        print(f"训练图像路径: {self.train_path}")
        print(f"验证图像路径: {self.val_path}")
        print(f"测试图像路径: {self.test_path}")
        print(f"类别数: {len(self.class_names)}")
        print(f"是否分析验证集: {self.use_val_set}")

    def _find_image_files(self, img_dir):
        """查找指定目录下的所有图像文件"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(list(img_dir.rglob(f'*{ext}')))
            image_files.extend(list(img_dir.rglob(f'*{ext.upper()}')))
        return list(set(image_files))  # 去重

    def _find_label_files(self, img_dir):
        """查找与指定图像目录对应的标签目录中的标签文件"""
        # 查找同级的labels目录
        if 'train' in str(img_dir):
            label_dir = img_dir.parent.parent / 'labels' / 'train'
            if not label_dir.exists():
                label_dir = img_dir.parent / 'labels'  # 尝试 images/labels
        elif 'val' in str(img_dir) or 'valid' in str(img_dir):
            label_dir = img_dir.parent.parent / 'labels' / 'val'
            if not label_dir.exists():
                label_dir = img_dir.parent / 'labels'  # 尝试 images/labels
        elif 'test' in str(img_dir):
            label_dir = img_dir.parent.parent / 'labels' / 'test'
            if not label_dir.exists():
                label_dir = img_dir.parent / 'labels'  # 尝试 images/labels
        else:
            # 如果图像目录不是train/val/test，直接查找同级labels
            label_dir = img_dir.parent / 'labels'

        # 检查标准的labels目录（与images同级）
        standard_label_dir = self.root_path / 'labels'
        if standard_label_dir.exists():
            # 查找train/val/test子目录
            if 'train' in str(img_dir):
                actual_label_dir = standard_label_dir / 'train'
            elif 'val' in str(img_dir) or 'valid' in str(img_dir):
                actual_label_dir = standard_label_dir / 'val'
            elif 'test' in str(img_dir):
                actual_label_dir = standard_label_dir / 'test'
            else:
                actual_label_dir = standard_label_dir

            if actual_label_dir.exists():
                return list(actual_label_dir.rglob('*.txt'))

        # 如果标准目录不存在，尝试其他路径
        if not standard_label_dir.exists() or not any(list(standard_label_dir.rglob('*.txt'))):
            if label_dir.exists():
                return list(label_dir.rglob('*.txt'))

        return []

    def analyze_dataset(self):
        """
        分析数据集，统计基本信息
        """
        print("开始分析数据集...")

        # 获取训练集图像和标签文件
        train_image_files = self._find_image_files(self.train_path)
        train_label_files = self._find_label_files(self.train_path)

        # 创建标签文件名到路径的映射
        label_name_to_path = {label_file.stem: label_file for label_file in train_label_files}

        print(f"训练集找到 {len(train_image_files)} 张图片")
        print(f"训练集找到 {len(train_label_files)} 个标签文件")

        # 统计训练集信息
        train_instances = 0
        train_class_counts = Counter()
        train_image_sizes = []
        train_aspect_ratios = []
        train_bbox_sizes = []  # 存储边界框尺寸
        processed_train_images = set()  # 记录已处理的训练图像

        for img_file in train_image_files:
            img_name = img_file.stem

            # 查找对应的标签文件
            if img_name in label_name_to_path:
                label_file = label_name_to_path[img_name]

                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    # 读取图像尺寸
                    try:
                        img = Image.open(img_file)
                        train_image_sizes.append(img.size)
                    except Exception as e:
                        print(f"无法读取图片 {img_file}: {e}")
                        continue

                    # 处理标签文件中的每一行（每个实例）
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO格式: class x_center y_center width height
                            class_id = int(float(parts[0]))

                            # 解析边界框坐标
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])

                            # 验证边界框坐标是否在合理范围内
                            if not (
                                    0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                                print(f"警告: 图像 {img_file} 中第 {line_idx + 1} 行的边界框坐标超出范围: {parts}")
                                continue

                            # 验证边界框宽高是否为正
                            if bbox_width <= 0 or bbox_height <= 0:
                                print(f"警告: 图像 {img_file} 中第 {line_idx + 1} 行的边界框宽高非正: {parts}")
                                continue

                            train_class_counts[class_id] += 1
                            train_instances += 1

                            # 计算宽高比
                            if bbox_height > 0:
                                train_aspect_ratios.append(bbox_width / bbox_height)

                            # 存储边界框尺寸 (归一化)
                            train_bbox_sizes.append((bbox_width, bbox_height))

                    # 记录已处理的图像（有标签的图像）
                    processed_train_images.add(img_file)

                except Exception as e:
                    print(f"处理标签文件 {label_file} 时出错: {e}")
                    continue
            else:
                # 图像没有对应的标签文件，但仍计入统计（作为背景拟合的图像）
                try:
                    img = Image.open(img_file)
                    train_image_sizes.append(img.size)
                    # 记录没有标签的图像（背景图像）
                    processed_train_images.add(img_file)
                    print(f"训练集图像 {img_file} 没有标签文件，作为背景图像计入统计")
                except Exception as e:
                    print(f"无法读取图片 {img_file}: {e}")
                    continue

        # 如果需要，也分析验证集
        val_instances = 0
        val_class_counts = Counter()
        val_image_sizes = []
        val_aspect_ratios = []
        val_bbox_sizes = []
        processed_val_images = set()

        if self.use_val_set:
            val_image_files = self._find_image_files(self.val_path)
            val_label_files = self._find_label_files(self.val_path)

            val_label_name_to_path = {label_file.stem: label_file for label_file in val_label_files}

            print(f"验证集找到 {len(val_image_files)} 张图片")
            print(f"验证集找到 {len(val_label_files)} 个标签文件")

            for img_file in val_image_files:
                img_name = img_file.stem

                if img_name in val_label_name_to_path:
                    label_file = val_label_name_to_path[img_name]

                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()

                        try:
                            img = Image.open(img_file)
                            val_image_sizes.append(img.size)
                        except Exception as e:
                            print(f"无法读取验证集图片 {img_file}: {e}")
                            continue

                        for line_idx, line in enumerate(lines):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))

                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                bbox_width = float(parts[3])
                                bbox_height = float(parts[4])

                                if not (
                                        0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                                    print(
                                        f"警告: 验证集图像 {img_file} 中第 {line_idx + 1} 行的边界框坐标超出范围: {parts}")
                                    continue

                                if bbox_width <= 0 or bbox_height <= 0:
                                    print(
                                        f"警告: 验证集图像 {img_file} 中第 {line_idx + 1} 行的边界框宽高非正: {parts}")
                                    continue

                                val_class_counts[class_id] += 1
                                val_instances += 1

                                if bbox_height > 0:
                                    val_aspect_ratios.append(bbox_width / bbox_height)
                                val_bbox_sizes.append((bbox_width, bbox_height))

                        processed_val_images.add(img_file)

                    except Exception as e:
                        print(f"处理验证集标签文件 {label_file} 时出错: {e}")
                        continue
                else:
                    # 验证集图像没有标签文件，但仍计入统计
                    try:
                        img = Image.open(img_file)
                        val_image_sizes.append(img.size)
                        # 记录没有标签的图像（背景图像）
                        processed_val_images.add(img_file)
                        print(f"验证集图像 {img_file} 没有标签文件，作为背景图像计入统计")
                    except Exception as e:
                        print(f"无法读取验证集图片 {img_file}: {e}")
                        continue

        # 合并统计信息
        total_images = len(processed_train_images) + len(processed_val_images)
        total_instances = train_instances + val_instances
        total_class_counts = train_class_counts
        for k, v in val_class_counts.items():
            total_class_counts[k] += v
        total_image_sizes = train_image_sizes + val_image_sizes
        total_aspect_ratios = train_aspect_ratios + val_aspect_ratios
        total_bbox_sizes = train_bbox_sizes + val_bbox_sizes

        # 保存统计结果
        self.stats = {
            'total_images': total_images,
            'total_instances': total_instances,
            'class_counts': dict(total_class_counts),
            'image_sizes': total_image_sizes,
            'aspect_ratios': total_aspect_ratios,
            'bbox_sizes': total_bbox_sizes,
            # 分别记录训练集和验证集的统计
            'train_images': len(processed_train_images),
            'train_instances': train_instances,
            'val_images': len(processed_val_images),
            'val_instances': val_instances
        }

        print("数据集分析完成!")
        return self.stats

    def print_statistics(self):
        """
        打印统计信息
        """
        if not self.stats:
            print("请先运行 analyze_dataset() 方法")
            return

        print("\n" + "=" * 50)
        print("数据集统计信息")
        print("=" * 50)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"总实例数: {self.stats['total_instances']}")

        # 详细分割统计
        print(f"\n详细分割统计:")
        print(f"  训练集图片数: {self.stats['train_images']}")
        print(f"  训练集实例数: {self.stats['train_instances']}")
        if self.use_val_set:
            print(f"  验证集图片数: {self.stats['val_images']}")
            print(f"  验证集实例数: {self.stats['val_instances']}")

        print("\n各类别实例数:")
        for class_id, count in sorted(self.stats['class_counts'].items()):
            class_name = self.class_names.get(class_id, f"类别 {class_id}")
            print(f"  {class_name}: {count}")

        if self.stats['image_sizes']:
            widths = [size[0] for size in self.stats['image_sizes']]
            heights = [size[1] for size in self.stats['image_sizes']]
            print(f"\n图像尺寸范围: {min(self.stats['image_sizes'])} - {max(self.stats['image_sizes'])}")
            print(f"平均图像尺寸: {np.mean(widths):.1f} x {np.mean(heights):.1f}")

        if self.stats['aspect_ratios']:
            print(f"宽高比统计: 平均值={np.mean(self.stats['aspect_ratios']):.2f}, "
                  f"标准差={np.std(self.stats['aspect_ratios']):.2f}")

        if self.stats['bbox_sizes']:
            bbox_widths = [w for w, h in self.stats['bbox_sizes']]
            bbox_heights = [h for w, h in self.stats['bbox_sizes']]
            print(f"边界框尺寸: 平均宽度={np.mean(bbox_widths):.3f}, 平均高度={np.mean(bbox_heights):.3f}")

    def visualize_dataset(self, output_dir="visualization_results"):
        """
        生成数据集可视化图表 - 重点展示按类别划分的小、中、大对象数量
        """
        if not self.stats:
            print("请先运行 analyze_dataset() 方法")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # --- 1. 原有的综合统计图 ---
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('数据集统计分析可视化', fontsize=16, fontweight='bold')

        # 1. 图像尺寸分布 (散点图)
        ax1 = axes[0, 0]
        if self.stats['image_sizes']:
            widths = [size[0] for size in self.stats['image_sizes']]
            heights = [size[1] for size in self.stats['image_sizes']]

            scatter = ax1.scatter(widths, heights, alpha=0.6, c='#4682B4', s=50, edgecolors='#000080')
            ax1.set_title('图像尺寸分布', fontsize=14, fontweight='bold')
            ax1.set_xlabel('宽度 (像素)')
            ax1.set_ylabel('高度 (像素)')
        else:
            ax1.text(0.5, 0.5, '无图像数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('图像尺寸分布')

        # 2. 实例宽高比分布 (直方图)
        ax2 = axes[0, 1]
        if self.stats['aspect_ratios']:
            ax2.hist(self.stats['aspect_ratios'], bins='auto', color='#2F4F4F',
                     edgecolor='#191970', alpha=0.7)
            ax2.set_title('实例宽高比分布', fontsize=14, fontweight='bold')
            ax2.set_xlabel('宽高比 (宽度/高度)')
            ax2.set_xlim(0, 5)
            ax2.set_ylabel('频次')

            # 添加统计信息
            mean_ratio = np.mean(self.stats['aspect_ratios'])
            std_ratio = np.std(self.stats['aspect_ratios'])
            ax2.axvline(mean_ratio, color='#FF4500', linestyle='--', label=f'均值: {mean_ratio:.2f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无宽高比数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('实例宽高比分布')

        # 3. 边界框尺寸分布
        ax3 = axes[1, 0]
        if self.stats['bbox_sizes']:
            bbox_widths = [w for w, h in self.stats['bbox_sizes']]
            bbox_heights = [h for w, h in self.stats['bbox_sizes']]

            scatter = ax3.scatter(bbox_widths, bbox_heights, alpha=0.6, c='#2F4F4F', s=30, edgecolors='#191970')
            ax3.set_title('边界框尺寸分布 (归一化)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('宽度 (归一化)')
            ax3.set_ylabel('高度 (归一化)')

            # 添加对角线 y=x，表示正方形边界框
            ax3.plot([0, 1], [0, 1], '#FF4500', alpha=0.5, label='正方形 (宽高比=1)')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, '无边界框数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('边界框尺寸分布')

        # 4. 组合图：各类别实例数分布 + 数据集分割分布 + 数据集总体统计
        ax4 = axes[1, 1]

        # 准备数据
        if self.stats['class_counts']:
            class_ids = list(self.stats['class_counts'].keys())
            class_names = [self.class_names.get(cid, f"类别 {cid}") for cid in class_ids]

            # 准备训练集和验证集数据
            train_counts = []
            val_counts = []

            for cid in class_ids:
                train_counts.append(self.stats['class_counts'].get(cid, 0))
                val_counts.append(0)  # 实际中需要统计验证集中的类别数量

            # 如果启用了验证集，重新统计验证集中的类别数量
            if self.use_val_set:
                # 重新分析验证集以获取类别分布
                val_label_files = self._find_label_files(self.val_path)
                val_label_name_to_path = {label_file.stem: label_file for label_file in val_label_files}

                val_class_counts = Counter()
                val_image_files = self._find_image_files(self.val_path)

                for img_file in val_image_files:
                    img_name = img_file.stem
                    if img_name in val_label_name_to_path:
                        label_file = val_label_name_to_path[img_name]
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(float(parts[0]))
                                    val_class_counts[class_id] += 1
                        except Exception as e:
                            print(f"处理验证集标签文件 {label_file} 时出错: {e}")
                            continue

                # 更新验证集计数
                for i, cid in enumerate(class_ids):
                    val_counts[i] = val_class_counts.get(cid, 0)

            # 计算总实例数
            total_instances = sum(train_counts) + sum(val_counts)

            # 如果有验证集数据，重新计算训练集数据（排除验证集部分）
            if self.use_val_set:
                for i, cid in enumerate(class_ids):
                    train_counts[i] = self.stats['class_counts'].get(cid, 0) - val_counts[i]
                    if train_counts[i] < 0:
                        train_counts[i] = 0  # 防止负数

            # 创建颜色列表（更丰富的深色系）
            colors = ['#2F4F4F', '#4682B4', '#8B0000', '#006400', '#4B0082', '#8B008B', '#B22222', '#228B22', '#483D8B',
                      '#9932CC']

            # 设置柱状图的宽度为原来的1/2
            bar_width = 0.2  # 使用固定的小宽度

            # 计算x轴位置
            x = np.arange(len(class_names))

            # 绘制训练集柱状图
            if total_instances > 0:
                train_ratios = [c / total_instances for c in train_counts]
                bars1 = ax4.bar(x - bar_width / 2, train_ratios, bar_width,
                                label='训练集', color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
            else:
                train_ratios = [0] * len(train_counts)
                bars1 = ax4.bar(x - bar_width / 2, train_ratios, bar_width,
                                label='训练集', color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)

            # 绘制验证集柱状图（如果启用）
            if self.use_val_set and total_instances > 0:
                val_ratios = [c / total_instances for c in val_counts]
                bars2 = ax4.bar(x + bar_width / 2, val_ratios, bar_width,
                                label='验证集', color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)

            # 设置纵坐标为比例
            ax4.set_ylabel('实例数比例', fontsize=12)
            ax4.set_xlabel('类别', fontsize=12)

            # 设置y轴范围
            max_ratio = max(train_ratios + val_ratios) if self.use_val_set else max(train_ratios)
            ax4.set_ylim(0, max_ratio * 1.2 if max_ratio > 0 else 0.1)

            # 在柱子上显示数值（比例）
            for i, (train_ratio, val_ratio) in enumerate(
                    zip(train_ratios, val_ratios if self.use_val_set else [0] * len(train_ratios))):
                if train_ratio > 0:
                    ax4.text(i - bar_width / 2, train_ratio, f'{train_ratio:.3f}',
                             ha='center', va='bottom', fontsize=8)

                if self.use_val_set and val_ratio > 0:
                    ax4.text(i + bar_width / 2, val_ratio, f'{val_ratio:.3f}',
                             ha='center', va='bottom', fontsize=8)

            # 设置x轴标签
            ax4.set_xticks(x)
            ax4.set_xticklabels(class_names, rotation=45, ha='right')

            # 添加图例
            ax4.legend(loc='upper right')

            # 在图中添加总体统计信息
            stats_text = f"总图片数: {self.stats['total_images']}\n总实例数: {self.stats['total_instances']}"
            ax4.text(0.02, 0.98, stats_text,
                     transform=ax4.transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontsize=10,
                     fontweight='bold')

        ax4.set_title('各类别实例数分布 & 总体统计', fontsize=14, fontweight='bold')

        # 调整布局，解决警告
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间

        # 保存图表
        output_path = os.path.join(output_dir, 'dataset_analysis.png')
        # 修复错误：figsize参数应放在figure创建时，而不是savefig时
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"主图表已保存至: {output_path}")
        plt.show()

        # --- 2. 新增：按类别统计小、中、大对象数量的柱状图 ---
        self._generate_size_by_category_chart(output_dir)

    def _generate_size_by_category_chart(self, output_dir):
        """
        生成按类别统计小、中、大对象数量的柱状图
        """
        # 首先需要将YOLO格式转换为COCO格式，以便复用原有的逻辑
        print("\n正在将YOLO格式转换为COCO格式以进行尺寸分析...")
        coco_path = os.path.join(output_dir, "temp_coco.json")

        # 重用yolo_to_coco逻辑 - 仅处理训练集
        train_image_files = self._find_image_files(self.train_path)
        train_label_files = self._find_label_files(self.train_path)

        if not train_image_files or not train_label_files:
            print("训练集没有找到图像或标签文件，无法生成尺寸分析图。")
            return

        # 创建标签文件名到路径的映射
        label_name_to_path = {label_file.stem: label_file for label_file in train_label_files}

        # 从YAML获取类别
        class_names_list = list(self.class_names.values())

        categories = [{"name": name, "id": idx + 1} for idx, name in enumerate(class_names_list)]

        images = []
        annotations = []
        annotation_id = 1
        processed_img_count = 0

        for img_path in sorted(train_image_files):
            img_name = img_path.stem

            # 查找对应的标签文件
            if img_name in label_name_to_path:
                label_file = label_name_to_path[img_name]

                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
                    continue

                image_id = processed_img_count + 1
                images.append({
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(img_path)
                })

                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0])) + 1
                            if class_id > len(class_names_list):
                                continue

                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])

                            # 验证边界框坐标范围
                            if not (
                                    0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                                print(
                                    f"警告: 训练集图像 {img_path} 第 {line_idx + 1} 行边界框坐标超出范围，跳过: {parts}")
                                continue

                            # 验证边界框宽高是否为正
                            if bbox_width <= 0 or bbox_height <= 0:
                                print(f"警告: 训练集图像 {img_path} 第 {line_idx + 1} 行边界框宽高非正，跳过: {parts}")
                                continue

                            x = (x_center - bbox_width / 2) * width
                            y = (y_center - bbox_height / 2) * height
                            w = bbox_width * width
                            h = bbox_height * height

                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, width - x)
                            h = min(h, height - y)

                            # 计算面积
                            area = w * h

                            # 检查面积是否有效
                            if area <= 0:
                                print(
                                    f"警告: 训练集图像 {img_path} 第 {line_idx + 1} 行计算出的面积为0或负数，跳过: {parts}")
                                continue

                            annotations.append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": [x, y, w, h],
                                "area": area,
                                "iscrowd": 0
                            })
                            annotation_id += 1
                except Exception as e:
                    print(f"Error reading YOLO file {label_file}: {e}")
                    continue

                processed_img_count += 1
            else:
                # 图像没有对应的标签文件，但仍计入统计（作为背景拟合的图像）
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
                    continue

                image_id = processed_img_count + 1
                images.append({
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(img_path)
                })
                processed_img_count += 1
                print(f"训练集图像 {img_path} 没有标签文件，作为背景图像计入统计")

        # 如果启用了验证集，也处理验证集
        if self.use_val_set:
            val_image_files = self._find_image_files(self.val_path)
            val_label_files = self._find_label_files(self.val_path)
            val_label_name_to_path = {label_file.stem: label_file for label_file in val_label_files}

            for img_path in sorted(val_image_files):
                img_name = img_path.stem

                if img_name in val_label_name_to_path:
                    label_file = val_label_name_to_path[img_name]

                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception as e:
                        print(f"Error reading validation image {img_path}: {e}")
                        continue

                    image_id = processed_img_count + 1
                    images.append({
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": os.path.basename(img_path)
                    })

                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()

                        for line_idx, line in enumerate(lines):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0])) + 1
                                if class_id > len(class_names_list):
                                    continue

                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                bbox_width = float(parts[3])
                                bbox_height = float(parts[4])

                                if not (
                                        0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                                    print(
                                        f"警告: 验证集图像 {img_path} 第 {line_idx + 1} 行边界框坐标超出范围，跳过: {parts}")
                                    continue

                                if bbox_width <= 0 or bbox_height <= 0:
                                    print(
                                        f"警告: 验证集图像 {img_path} 第 {line_idx + 1} 行边界框宽高非正，跳过: {parts}")
                                    continue

                                x = (x_center - bbox_width / 2) * width
                                y = (y_center - bbox_height / 2) * height
                                w = bbox_width * width
                                h = bbox_height * height

                                x = max(0, x)
                                y = max(0, y)
                                w = min(w, width - x)
                                h = min(h, height - y)

                                area = w * h

                                if area <= 0:
                                    print(
                                        f"警告: 验证集图像 {img_path} 第 {line_idx + 1} 行计算出的面积为0或负数，跳过: {parts}")
                                    continue

                                annotations.append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": class_id,
                                    "bbox": [x, y, w, h],
                                    "area": area,
                                    "iscrowd": 0
                                })
                                annotation_id += 1
                    except Exception as e:
                        print(f"Error reading validation YOLO file {label_file}: {e}")
                        continue

                    processed_img_count += 1
                else:
                    # 验证集图像没有标签文件，但仍计入统计
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception as e:
                        print(f"Error reading validation image {img_path}: {e}")
                        continue

                    image_id = processed_img_count + 1
                    images.append({
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": os.path.basename(img_path)
                    })
                    processed_img_count += 1
                    print(f"验证集图像 {img_path} 没有标签文件，作为背景图像计入统计")

        coco_data = {
            "info": {"description": "Converted from YOLO", "version": "1.0", "year": 2024},
            "licenses": [{"id": 1, "name": "Unknown"}],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)

        # 2. 加载COCO并分析尺寸
        coco = COCO(coco_path)
        img_ids = coco.getImgIds()

        # 获取所有类别ID和名称
        cats = coco.loadCats(coco.getCatIds())
        cat_names = [cat['name'] for cat in cats]
        cat_ids = [cat['id'] for cat in cats]

        print(f"Found categories: {dict(zip(cat_ids, cat_names))}")

        # 初始化统计变量
        small_objects = 0
        medium_objects = 0
        large_objects = 0
        cat_small_counts = {cat_id: 0 for cat_id in cat_ids}
        cat_medium_counts = {cat_id: 0 for cat_id in cat_ids}
        cat_large_counts = {cat_id: 0 for cat_id in cat_ids}

        for id in img_ids:
            img_info = coco.loadImgs(id)[0]
            height = img_info["height"]
            width = img_info["width"]

            ann_ids = coco.getAnnIds(imgIds=id)
            ann_info = coco.loadAnns(ann_ids)

            for _, ann in enumerate(ann_info):
                category_id = ann["category_id"]
                x1, y1, w, h = ann["bbox"]
                scale_ratio = min(640 / height, 640 / width)
                scale_w = w * scale_ratio
                scale_h = h * scale_ratio
                area = scale_w * scale_h

                # 检查面积是否有效（理论上经过前面的过滤，这里应该都是有效的）
                if area <= 0:
                    print(
                        f"警告: 在COCO格式中发现面积为0或负数的边界框，这不应该发生: [x={x1}, y={y1}, w={w}, h={h}], 原始面积={ann['area']}")
                    continue

                if area <= 32 * 32:
                    small_objects += 1
                    cat_small_counts[category_id] += 1
                elif area <= 96 * 96:
                    medium_objects += 1
                    cat_medium_counts[category_id] += 1
                else:
                    large_objects += 1
                    cat_large_counts[category_id] += 1

        print("小对象总数:", small_objects)
        print("中对象总数:", medium_objects)
        print("大对象总数:", large_objects)

        # 打印每个类别的详细统计
        print("\n各类型对象按类别统计:")
        for cat_id, cat_name in zip(cat_ids, cat_names):
            print(
                f"{cat_name} - 小: {cat_small_counts[cat_id]}, 中: {cat_medium_counts[cat_id]}, 大: {cat_large_counts[cat_id]}")

        # 计算所有类别的总数量
        labels = ["Small", "Medium", "Large"]
        data_by_category = []
        category_names = []

        for cat_id, cat_name in zip(cat_ids, cat_names):
            data_by_category.append([
                cat_small_counts[cat_id],
                cat_medium_counts[cat_id],
                cat_large_counts[cat_id]
            ])
            category_names.append(cat_name)

        index = np.arange(len(labels))
        width = 0.15


        # plots - 使用深色系风格
        fig, ax = plt.subplots(figsize=(max(8, len(cat_names) * 0.8), 5), dpi=200)

        # 修改前
        # 为每个类别绘制柱状图
        for i, (data, cat_name) in enumerate(zip(data_by_category, category_names)):
            offset = (i - len(data_by_category) // 2) * width
            color_idx = i % 10  # 循环使用颜色
            colors = ["#130074", "#CB181B", "#008B45", "#FDBF6F", "#6A3D9A", "#FF7F00", "#33A02C", "#E31A1C", "#1F78B4",
                      "#FB9A99"]
            color = colors[color_idx]

            bars = ax.bar(
                index + offset,
                data,
                width,
                label=cat_name,
                color=color,
                ec="black",
                lw=0.5,
            )


        # # 修改开始
        # # --- 修改开始：定义网格样式 ---
        # # ！颜色改网格！
        # # 定义不同的网格样式
        # hatch_patterns = ['///', 'xxx', '...', '+++', '|||', '---', '***', 'OOO', '...']
        # # 统一的颜色（灰色系，突出网格）
        # gray_color = '#D3D3D3'  # 浅灰色
        #
        # # 为每个类别绘制柱状图
        # for i, (data, cat_name) in enumerate(zip(data_by_category, category_names)):
        #     offset = (i - len(data_by_category) // 2) * width
        #     # 使用灰色填充
        #     # 使用不同的网格图案
        #     hatch = hatch_patterns[i % len(hatch_patterns)]
        #
        #     bars = ax.bar(
        #         index + offset,
        #         data,
        #         width,
        #         label=cat_name,
        #         color=gray_color,  # 填充色改为灰色
        #         edgecolor='black',  # 边框保持黑色
        #         hatch=hatch,  # 关键：添加网格图案
        #         linewidth=0.5,  # 边框宽度
        #         zorder=3  # 确保在网格线上层
        #     )
        #
        # # --- 修改结束 ---

        # 柱子上的数字显示 - 按类别统计：每个类别在三种尺度中的比例相加为1
        # 计算每个类别的总数，然后计算在各尺度中的占比
        total_by_category = [sum(data) for data in data_by_category]
        percentages_by_category = []

        for i, (data, total) in enumerate(zip(data_by_category, total_by_category)):
            if total > 0:
                per_small = data[0] / total
                per_medium = data[1] / total
                per_large = data[2] / total
            else:
                per_small = 0
                per_medium = 0
                per_large = 0

            percentages_by_category.append([per_small, per_medium, per_large])

        # 在柱子上显示百分比
        for i, (data, per) in enumerate(zip(data_by_category, percentages_by_category)):
            offset = (i - len(data_by_category) // 2) * width
            for a, b, c in zip(index + offset, data, per):
                if b > 0:  # 只在有数据时显示
                    ax.text(a, b, '%.3f' % c, ha="center", va="bottom", fontsize=6)

        # 定制化设计 - 保持multibbox风格
        ax.tick_params(axis="x", direction="in", bottom=False)
        ax.tick_params(axis="y", direction="out", labelsize=8, length=3)
        ax.set_ylabel("Number of objects", fontsize='small')
        ax.set_xticks(index)
        ax.set_xticklabels([f"Small\n", f"Medium\n", f"Large\n"], fontsize='small')

        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")
        # 修改前：
        # ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        # 修改后：
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1, 1),
            frameon=True,
            framealpha=0.7,
            fontsize=8
        )

        # 解决tight_layout警告
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'size_distribution_by_category.png')
        plt.savefig(output_path, dpi=900, bbox_inches="tight")
        print(f"尺寸分布图已保存至: {output_path}")
        plt.show()

        # 删除临时COCO文件
        os.remove(coco_path)


def main():
    """
    主函数 - 运行数据集分析
    """
    print("基于YAML配置的YOLO数据集统计分析工具")
    print("=" * 50)
    print("根据YAML配置文件自动解析数据集路径和结构")
    print("标签文件目录与images目录同级")
    print("=" * 50)

    # --- 自动获取YAML路径，无需键盘输入 ---
    # 你可以在这里直接指定YAML文件路径，例如：
    yaml_path = "../datayaml/URPC_trainset.yaml"  # 修改为你实际的YAML文件路径
    # 或者在代码中搜索当前目录下的data.yaml
    # yaml_path = next(Path(".").glob("data.yaml"), None)
    # if not yaml_path:
    #     print("未找到 data.yaml 文件")
    #     return
    # yaml_path = str(yaml_path)
    # -----------------------------------------

    # 默认不使用验证集
    use_validation = True  # 设为True如果需要分析验证集

    if not os.path.exists(yaml_path):
        print(f"错误: YAML文件 '{yaml_path}' 不存在")
        return

    # 创建分析器
    analyzer = DatasetAnalyzer(yaml_path, use_val_set=use_validation)

    # 分析数据集
    stats = analyzer.analyze_dataset()

    # 打印统计信息
    analyzer.print_statistics()

    # 生成可视化图表
    # --- 自动设置输出目录，无需键盘输入 ---
    output_dir = "datasets_visualization_results"  # 修改为你想要的输出目录
    datayaml_name = Path(yaml_path).stem
    output_dir = os.path.join(output_dir, datayaml_name)
    # -----------------------------------------

    analyzer.visualize_dataset(output_dir)

    print(f"\n分析完成! 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()



