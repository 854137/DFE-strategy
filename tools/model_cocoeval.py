import json
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml
import cv2


def create_output_directories(weights_path):
    """
    自动创建输出目录：在权重文件同级创建 evaluate_yolo_multiscale 目录
    """
    weights_dir = Path(weights_path).parent
    # 在权重目录的同级创建 evaluate_yolo_multiscale 目录
    output_dir = weights_dir.parent / "evaluate_yolo_multiscale"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件路径
    gt_json_path = output_dir / "val_coco.json"
    pred_json_path = output_dir / "pred.json"
    metrics_output = output_dir / "metrics.txt"

    return str(gt_json_path), str(pred_json_path), str(metrics_output), str(output_dir)


def load_yaml_config(yaml_path):
    """
    从YAML文件加载验证集路径和类别信息
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    # 获取数据集根目录
    dataset_root = Path(yaml_data.get('path', Path(yaml_path).parent))
    print(f"📊 数据集根目录: {dataset_root}")

    # 获取验证集图像路径
    val_images_subdir = yaml_data.get('val', 'images/test')  # 默认使用test，如果val不存在
    val_images_path = dataset_root / val_images_subdir
    print(f"📊 验证集图像路径: {val_images_path}")

    # 推导验证集标签路径
    val_labels_path = dataset_root / val_images_subdir.replace('images', 'labels')
    print(f"📊 验证集标签路径: {val_labels_path}")

    # 检查路径是否存在
    if not val_images_path.exists():
        raise FileNotFoundError(f"验证集图像目录不存在: {val_images_path}")
    if not val_labels_path.exists():
        raise FileNotFoundError(f"验证集标签目录不存在: {val_labels_path}")

    # 获取类别名称
    if 'names' in yaml_data:
        if isinstance(yaml_data['names'], list):
            class_names = yaml_data['names']
        elif isinstance(yaml_data['names'], dict):
            class_names = [yaml_data['names'][i] for i in sorted(yaml_data['names'].keys())]
    elif 'nc' in yaml_data and 'names' in yaml_data:
        if isinstance(yaml_data['names'], list):
            class_names = yaml_data['names']
    else:
        raise ValueError("YAML文件中未找到类别名称信息")

    return str(val_images_path), str(val_labels_path), class_names


def yolo_to_coco(images_dir, labels_dir, class_names, output_json):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # 验证路径是否存在
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    print(f"🔍 验证图像目录: {images_dir}")
    print(f"🔍 验证标签目录: {labels_dir}")

    # 检查标签目录中是否有标签文件
    label_files = list(labels_dir.glob("*.txt"))
    print(f"🔍 在标签目录中找到 {len(label_files)} 个标签文件")

    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for img_path in sorted(images_dir.iterdir()):
        # 严格检查图像文件扩展名
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            continue

        # 只处理有标签的图像
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 跳过无法读取的图像: {img_path}")
            continue
        height, width = img.shape[:2]

        images.append({
            "id": image_id,
            "file_name": img_path.name,
            "height": height,
            "width": width
        })

        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(f"⚠️ 标签文件 {label_path} 为空")
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"⚠️ 标签文件 {label_path} 第 {line_idx + 1} 行格式不正确: {line}")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # 验证坐标范围
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"⚠️ 标签文件 {label_path} 第 {line_idx + 1} 行坐标超出范围: {parts}")
                        continue

                    # 转换为像素坐标
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height

                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    # 确保边界框在图像范围内
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    w = min(w, width - x_min)
                    h = min(h, height - y_min)

                    area = w * h

                    if w <= 0 or h <= 0:
                        print(f"⚠️ 标签文件 {label_path} 第 {line_idx + 1} 行边界框宽高无效: {parts}")
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id,
                        "bbox": [x_min, y_min, w, h],
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
                except (ValueError, IndexError) as e:
                    print(f"⚠️ 标签文件 {label_path} 第 {line_idx + 1} 行解析错误: {line} - {e}")
                    continue

        image_id += 1

    print(f"📊 转换完成: {len(images)} 张图像, {len(annotations)} 个标注")

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    # ✅ 补全 COCO 格式必需的顶层字段
    coco_format = {
        "info": {
            "description": "Converted from YOLO format",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-10-26"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"✅ 已生成 COCO 格式标注文件: {output_json}")


def run_evaluation(
        weights_path,
        yaml_path,
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        single_cls=False,
        save_json=True,
        project="runs/val",
        name="exp",
        exist_ok=False,
):
    """
    执行YOLOv8模型评估
    """
    # 从YAML获取验证集路径和类别信息
    val_images_dir, val_labels_dir, class_names = load_yaml_config(yaml_path)

    print(f"📊 从YAML加载的配置:")
    print(f"   验证集图像目录: {val_images_dir}")
    print(f"   验证集标签目录: {val_labels_dir}")
    print(f"   类别数量: {len(class_names)}")
    print(f"   类别名称: {class_names}")

    # 自动创建输出目录并获取文件路径
    gt_json_path, pred_json_path, metrics_output, output_dir = create_output_directories(weights_path)

    print(f"📁 输出目录: {output_dir}")

    # 强制重新生成JSON文件（删除已存在的文件）
    for json_path in [gt_json_path, pred_json_path]:
        if Path(json_path).exists():
            print(f"🗑️ 删除已存在的JSON文件: {json_path}")
            Path(json_path).unlink()

    # 重新生成 ground truth JSON
    print("🔄 步骤 1/3: YOLO 格式 → COCO 格式 (ground truth)...")
    try:
        yolo_to_coco(val_images_dir, val_labels_dir, class_names, gt_json_path)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return

    # 加载模型
    print("🔄 加载模型...")
    model = YOLO(weights_path)

    # 获取图像文件列表（只包含有标签的图像，严格过滤）
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    all_files = []
    for ext in image_extensions:
        all_files.extend(list(Path(val_images_dir).glob(f"*{ext}")))
        all_files.extend(list(Path(val_images_dir).glob(f"*{ext.upper()}")))

    # 移除重复项
    all_files = list(set(all_files))

    print(f"📊 在图像目录 {val_images_dir} 中找到 {len(all_files)} 个图像文件")

    # 显示一些文件名用于调试
    if len(all_files) > 0:
        print(f"📊 前10个图像文件: {[f.name for f in all_files[:10]]}")

    # 仅保留有标签的图像
    image_files_with_labels = []
    for p in all_files:
        label_path = Path(val_labels_dir) / (p.stem + ".txt")
        if label_path.exists():
            image_files_with_labels.append(p)
        else:
            print(f"⚠️ 图像 {p.name} 没有对应的标签文件 {label_path}")

    image_files = sorted(image_files_with_labels)
    print(f"📊 找到 {len(image_files)} 个有标签的图像")  # 应该输出 4200

    # 检查是否图像数量仍然不匹配
    if len(image_files) != 4200:
        print(f"⚠️ 警告: 预期4200个图像，但找到了{len(image_files)}个图像")
        print("请检查您的数据集目录结构是否正确")

        # 如果找到了8400个图像，但实际只有4200个标签，说明可能有重复文件
        if len(all_files) == 8400 and len(image_files_with_labels) == 4200:
            print("🔍 检测到可能的重复文件问题...")
            # 检查是否有重复的文件名（大小写不同）
            names = [f.name.lower() for f in all_files]
            unique_names = list(set(names))
            if len(names) != len(unique_names):
                print(f"⚠️ 发现重复文件名（大小写不同）: {len(names) - len(unique_names)} 个")
                # 只保留唯一文件名
                seen_names = set()
                unique_files = []
                for f in all_files:
                    if f.name.lower() not in seen_names:
                        unique_files.append(f)
                        seen_names.add(f.name.lower())
                all_files = unique_files
                print(f"📊 去重后找到 {len(all_files)} 个图像文件")

                # 重新筛选有标签的图像
                image_files_with_labels = []
                for p in all_files:
                    label_path = Path(val_labels_dir) / (p.stem + ".txt")
                    if label_path.exists():
                        image_files_with_labels.append(p)
                image_files = sorted(image_files_with_labels)
                print(f"📊 去重后找到 {len(image_files)} 个有标签的图像")

    # 构建 image_id 映射（只包含有标签的图像，ID从1开始）
    image_id_map = {p.name: idx + 1 for idx, p in enumerate(image_files)}

    predictions = []

    print("🔄 步骤 2/3: 模型推理 → COCO 预测格式...")

    # 使用tqdm显示进度
    pbar = tqdm(total=len(image_files), desc="模型推理")

    # 使用模型进行预测
    results = model.predict(
        source=val_images_dir,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        batch=batch_size,
        save=False,
        stream=True,
        verbose=False
    )

    for result in results:  # results 是生成器，不是列表
        img_name = Path(result.path).name
        image_id = image_id_map.get(img_name)
        if image_id is None:
            # 如果图像没有标签，跳过（只处理有标签的图像）
            continue

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf.item())
                cls = int(box.cls.item())

                # 转换为COCO格式 [x, y, width, height]
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                predictions.append({
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                    "score": round(conf, 5)
                })

        pbar.update(1)

    pbar.close()

    # 保存预测结果
    with open(pred_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"✅ 预测结果已保存: {pred_json_path}")

    # 调试：检查JSON文件内容
    print("\n🔍 调试信息 - 检查JSON文件内容:")

    # 检查ground truth JSON
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    print(f"   Ground Truth JSON:")
    print(f"     - 图像数量: {len(gt_data.get('images', []))}")
    print(f"     - 标注数量: {len(gt_data.get('annotations', []))}")
    print(f"     - 类别数量: {len(gt_data.get('categories', []))}")

    if gt_data.get('annotations'):
        print(f"     - 示例标注: {gt_data['annotations'][0] if gt_data['annotations'] else 'None'}")

    # 检查预测JSON
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    print(f"   预测 JSON:")
    print(f"     - 预测数量: {len(pred_data)}")

    if pred_data:
        print(f"     - 示例预测: {pred_data[0] if pred_data else 'None'}")

    # 检查是否有图像ID匹配
    gt_img_ids = set(ann['image_id'] for ann in gt_data.get('annotations', []))
    pred_img_ids = set(pred['image_id'] for pred in pred_data)

    print(f"   - Ground Truth中涉及的图像ID: {sorted(list(gt_img_ids)[:5])}...")  # 显示前5个
    print(f"   - 预测中涉及的图像ID: {sorted(list(pred_img_ids)[:5])}...")  # 显示前5个
    print(f"   - 共同的图像ID数量: {len(gt_img_ids & pred_img_ids)}")

    print("\n🔄 步骤 3/3: COCO 多尺度评估...")
    try:
        coco_gt = COCO(gt_json_path)
        coco_dt = coco_gt.loadRes(pred_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # 这会触发 stats 计算

        # 提取并展示 6 项指标
        print_and_save_metrics(coco_eval, output_file=metrics_output)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈


def print_and_save_metrics(coco_eval, output_file=None):
    """
    从 COCOeval 对象中提取 6 项标准指标并打印/保存
    coco_eval.stats 索引说明（来自 pycocotools 源码）：
      stats[0] = AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
      stats[1] = AP @[ IoU=0.50      | area=all | maxDets=100 ]
      stats[2] = AP @[ IoU=0.75      | area=all | maxDets=100 ]
      stats[3] = AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
      stats[4] = AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      stats[5] = AP @[ IoU=0.50:0.95 | area=large | maxDets=100 ]
    """
    stats = coco_eval.stats

    metrics = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "APS": stats[3],
        "APM": stats[4],
        "APL": stats[5]
    }

    # 打印到控制台
    print("\n" + "=" * 50)
    print("📊 COCO 评估结果 (6 项核心指标):")
    print("=" * 50)
    print(f"AP     (mAP@[0.5:0.95]) : {metrics['AP']:.4f}")
    print(f"AP50   (mAP@0.50)       : {metrics['AP50']:.4f}")
    print(f"AP75   (mAP@0.75)       : {metrics['AP75']:.4f}")
    print(f"APS    (small objects)  : {metrics['APS']:.4f}")
    print(f"APM    (medium objects) : {metrics['APM']:.4f}")
    print(f"APL    (large objects)  : {metrics['APL']:.4f}")
    print("=" * 50)

    # 可选：保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write("COCO Evaluation Metrics:\n")
            f.write(f"AP     = {metrics['AP']:.6f}\n")
            f.write(f"AP50   = {metrics['AP50']:.6f}\n")
            f.write(f"AP75   = {metrics['AP75']:.6f}\n")
            f.write(f"APS    = {metrics['APS']:.6f}\n")
            f.write(f"APM    = {metrics['APM']:.6f}\n")
            f.write(f"APL    = {metrics['APL']:.6f}\n")
        print(f"✅ 指标已保存到: {output_file}")


# 直接在这里设置参数
if __name__ == "__main__":
    # 请根据你的实际情况修改以下路径
    # WEIGHTS_PATH = "../runs/detect/selfstudy/attempt/fadc2f_ablation/standard_ciou_focus_scratch_ruod_300e/weights/best.pt"
    WEIGHTS_PATH = "../runs/detect/patent/patent1/Secondtime_yolov8s_patent1new_scratch_ruod_300e/weights/best.pt"
    # WEIGHTS_PATH = "../runs/detect/selfstudy/yolo/yolov5l(dfl=1)_scratch_urpc(t)_100e/weights/last.pt"
    # WEIGHTS_PATH = "../runs/detect/reproduce/bsr5/bsr5_detr_scratch_ruod_300e/weights/best.pt"
    # WEIGHTS_PATH = "../runs/detect/selfstudy/rtdetr/rtdetr_resnet50_scratch_ruod_300e/weights/best.pt"
    # WEIGHTS_PATH = "../runs/detect/selfstudy/rtmdet/rtmdet_l_scratch_urpc(t)_100e/weights/last.pt"
    # WEIGHTS_PATH = "../runs/detect/selfstudy/retinanet/retinanet101(dfl=1)_scratch_urpc(t)_100e/weights/last.pt"
    # WEIGHTS_PATH = "../runs/detect/reproduce/amsp_uod/yolov8n(dfl=1)_amsp_uod(p)_scratch_urpc(t)_100e/weights/last.pt"
    # 替换为你的YOLOv8模型文件路径
    # 注意：专利用last.pt，其他都用best.pt
    YAML_PATH = "../datayaml/RUOD.yaml"  # 替换为你的YAML数据集配置文件路径
    # YAML_PATH = "../datayaml/URPC_trainset.yaml"  # 替换为你的YAML数据集配置文件路径

    # 执行评估
    try:
        run_evaluation(
            weights_path=WEIGHTS_PATH,
            yaml_path=YAML_PATH,
            conf_thres=0.005,
            # pred.json文件太大则把置信度阈值调高，调至0.01
            iou_thres=0.6,
            batch_size=32,
            # 内存超出则调小batch_size,调至4
            # 或者尝试重启电脑！
            imgsz=640
        )
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()



