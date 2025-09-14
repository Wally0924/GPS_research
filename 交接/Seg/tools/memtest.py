import os
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from rich import print
from rich.progress import Progress, track
from rich.table import Table
from rich.console import Console

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset
from engine.metric import Metrics
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.geo_v2 import create_memory_enhanced_geo_segformer
from geotrain_v2 import MemoryEnhancedGeoSegDataset, setup_gps_normalization

console = Console()


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Memory-Enhanced GeoSegformer Testing")
    
    # 基本参数
    parser.add_argument("img_dir", type=str, help="Test images directory")
    parser.add_argument("ann_dir", type=str, help="Test annotations directory") 
    parser.add_argument("gps_csv", type=str, help="Test GPS CSV file")
    parser.add_argument("category_csv", type=str, help="Category definition CSV")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint path")
    
    # GPS正规化参数（必需，用于和训练时保持一致）
    parser.add_argument("train_gps_csv", type=str, help="Training GPS CSV for normalization")
    parser.add_argument("val_gps_csv", type=str, help="Validation GPS CSV for normalization")
    
    # 输出设置
    parser.add_argument("--save-dir", type=str, default="./test_results", help="Results save directory")
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction visualizations")
    
    # 测试参数
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--max-len", type=int, default=None, help="Max number of test samples")
    
    # 测试模式
    parser.add_argument("--test-mode", type=str, default="memory_aware", 
                       choices=["inference", "progressive", "fast", "benchmark", "memory_aware"],
                       help="Testing mode: inference/progressive/fast/benchmark/memory_aware")
    
    # 快速测试选项
    parser.add_argument("--fast-eval", action="store_true", help="Use faster evaluation (less detailed)")
    parser.add_argument("--skip-memory-update", action="store_true", help="Skip memory updates for speed")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval for progressive mode")
    
    # 模型参数（从checkpoint自动推断，这些是备用值）
    parser.add_argument("--model-size", type=str, default="b0")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--fusion-method", type=str, default="attention")
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--spatial-radius", type=float, default=0.00005)
    parser.add_argument("--gps-norm-method", type=str, default="minmax")
    
    return parser.parse_args()


def analyze_gps_overlap(test_gps_csv: str, train_gps_csv: str, val_gps_csv: str, spatial_radius: float = 0.00005):
    """分析测试数据与训练数据的GPS重叠情况"""
    console.print(f"\n[bold cyan]📍 分析GPS数据重叠情况...")
    
    # 读取数据
    test_gps = pd.read_csv(test_gps_csv)
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    
    # 合并训练数据
    train_val_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    console.print(f"  测试数据: {len(test_gps)} 个GPS点")
    console.print(f"  训练数据: {len(train_val_gps)} 个GPS点")
    
    # 分析重叠情况
    exact_matches = 0
    nearby_matches = 0
    no_matches = 0
    
    match_info = []
    
    for _, test_row in test_gps.iterrows():
        test_lat, test_lon = test_row['lat'], test_row['long']
        
        # 计算与所有训练点的距离
        distances = []
        for _, train_row in train_val_gps.iterrows():
            train_lat, train_lon = train_row['lat'], train_row['long']
            distance = ((test_lat - train_lat)**2 + (test_lon - train_lon)**2)**0.5
            distances.append(distance)
        
        min_distance = min(distances)
        
        if min_distance < spatial_radius:
            exact_matches += 1
            match_type = "exact"
        elif min_distance < spatial_radius * 3:
            nearby_matches += 1  
            match_type = "nearby"
        else:
            no_matches += 1
            match_type = "none"
        
        match_info.append({
            'filename': test_row['filename'],
            'lat': test_lat,
            'lon': test_lon,
            'min_distance': min_distance,
            'match_type': match_type
        })
    
    # 输出统计
    total = len(test_gps)
    console.print(f"\n📊 GPS匹配统计:")
    console.print(f"  精确匹配 (距离 < {spatial_radius:.5f}): {exact_matches} ({exact_matches/total*100:.1f}%)")
    console.print(f"  附近匹配 (距离 < {spatial_radius*3:.5f}): {nearby_matches} ({nearby_matches/total*100:.1f}%)")
    console.print(f"  无匹配 (距离 > {spatial_radius*3:.5f}): {no_matches} ({no_matches/total*100:.1f}%)")
    
    memory_coverage = (exact_matches + nearby_matches) / total * 100
    console.print(f"  记忆覆盖率: {memory_coverage:.1f}%")
    
    if memory_coverage > 70:
        console.print(f"  ✅ 记忆覆盖率良好，适合测试记忆功能")
    elif memory_coverage > 30:
        console.print(f"  ⚠️ 记忆覆盖率中等，部分位置能利用记忆")
    else:
        console.print(f"  ❌ 记忆覆盖率较低，大部分位置无法利用记忆")
    
    return match_info, {
        'exact_matches': exact_matches,
        'nearby_matches': nearby_matches, 
        'no_matches': no_matches,
        'memory_coverage': memory_coverage
    }
    """从checkpoint加载模型"""
    console.print(f"[bold blue]📥 加载模型checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint获取训练时的参数
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        model_size = getattr(saved_args, 'model_size', args.model_size)
        feature_dim = getattr(saved_args, 'feature_dim', args.feature_dim)
        fusion_method = getattr(saved_args, 'fusion_method', args.fusion_method)
        memory_size = getattr(saved_args, 'memory_size', args.memory_size)
        spatial_radius = getattr(saved_args, 'spatial_radius', args.spatial_radius)
        
        console.print(f"✅ 使用训练时的模型参数:")
        console.print(f"  模型大小: {model_size}")
        console.print(f"  特征维度: {feature_dim}")
        console.print(f"  融合方法: {fusion_method}")
        console.print(f"  记忆库大小: {memory_size}")
        console.print(f"  空间半径: {spatial_radius}")
    else:
        # 使用命令行参数作为备用
        model_size = args.model_size
        feature_dim = args.feature_dim
        fusion_method = args.fusion_method
        memory_size = args.memory_size
        spatial_radius = args.spatial_radius
        console.print(f"⚠️  使用默认参数（checkpoint中无保存的参数）")
    
    # 创建模型
    model = create_memory_enhanced_geo_segformer(
        num_classes=num_classes,
        model_size=model_size,
        feature_dim=feature_dim,
        fusion_method=fusion_method,
        memory_size=memory_size,
        spatial_radius=spatial_radius
    ).to(device)
    
    # 加载权重
    try:
        model.load_state_dict(checkpoint["model"])
        console.print(f"✅ 模型权重加载成功")
    except Exception as e:
        console.print(f"❌ 权重加载失败: {e}")
        raise e
    
    # 显示训练时的记忆库统计（如果有）
    if 'memory_stats' in checkpoint:
        stats = checkpoint['memory_stats']
        console.print(f"📊 训练时的记忆库状态:")
        console.print(f"  总位置数: {stats['total_locations']}")
        console.print(f"  总记忆数: {stats['total_memories']}")
        console.print(f"  命中率: {stats['hit_rate']:.4f}")
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"🔧 模型参数量: {total_params:.2f}M")
    
    return model


def inference_mode_test(model, dataloader, device: str, args: Namespace) -> Dict[str, Any]:
    """推理模式测试：使用训练时的记忆，不更新记忆库"""
    console.print(f"\n[bold green]🔮 推理模式测试 (使用训练时的记忆)")
    console.print(f"  测试样本数: {len(dataloader)}")
    
    num_categories = len(Category.load(args.category_csv))
    if any(cat.id == 255 for cat in Category.load(args.category_csv)):
        num_categories -= 1
        
    metrics = Metrics(num_categories, nan_to_num=0)
    
    model.eval()
    total_memory_weight = 0
    total_samples = 0
    predictions = []
    
    # 获取初始记忆库状态
    initial_memory_stats = model.get_memory_stats()
    console.print(f"📊 初始记忆库状态:")
    console.print(f"  位置数: {initial_memory_stats['total_locations']}")
    console.print(f"  记忆数: {initial_memory_stats['total_memories']}")
    console.print(f"  命中率: {initial_memory_stats['hit_rate']:.4f}")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(track(dataloader, description="推理测试")):
            img = data["img"].to(device)
            ann = data["ann"].to(device)[:, 0, :, :]
            gps = data["gps"].to(device)
            filename = data.get("filename", [f"sample_{batch_idx}"])
            
            # 推理（不更新记忆库）
            outputs = model(img, gps, return_embeddings=False, update_memory=False)
            pred = outputs['segmentation_logits']
            memory_weight = outputs.get('memory_weight', 0)
            
            # 计算指标
            pred_labels = pred.argmax(1)
            metrics.compute_and_accum(pred_labels, ann)
            
            total_memory_weight += memory_weight
            total_samples += img.shape[0]
            
            # 保存预测结果用于可视化
            if args.save_predictions:
                predictions.append({
                    'filename': filename[0] if isinstance(filename, list) else filename,
                    'pred': pred_labels.cpu(),
                    'ann': ann.cpu(),
                    'memory_weight': memory_weight
                })
    
    # 获取最终结果
    results = metrics.get_and_reset()
    avg_memory_weight = total_memory_weight / total_samples
    
    # 获取最终记忆库状态（应该和初始状态相同）
    final_memory_stats = model.get_memory_stats()
    
    console.print(f"\n📈 推理模式结果:")
    console.print(f"  mIoU: {results['IoU'].mean():.5f}")
    console.print(f"  mDice: {results['Dice'].mean():.5f}")
    console.print(f"  整体准确率: {results['aAcc']:.5f}")
    console.print(f"  平均记忆权重: {avg_memory_weight:.4f}")
    console.print(f"  记忆库状态: 无变化 (推理模式)")
    
    return {
        'mode': 'inference',
        'miou': results['IoU'].mean(),
        'dice': results['Dice'].mean(),
        'accuracy': results['aAcc'],
        'memory_weight': avg_memory_weight,
        'detailed_results': results,
        'predictions': predictions,
        'memory_stats': {
            'initial': initial_memory_stats,
            'final': final_memory_stats
        }
    }


def fast_mode_test(model, dataloader, device: str, args: Namespace) -> Dict[str, Any]:
    """快速测试模式：只测试核心功能，减少详细统计"""
    console.print(f"\n[bold cyan]⚡ 快速测试模式")
    console.print(f"  测试样本数: {len(dataloader)}")
    
    num_categories = len(Category.load(args.category_csv))
    if any(cat.id == 255 for cat in Category.load(args.category_csv)):
        num_categories -= 1
    
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    total_memory_weight = 0
    total_samples = 0
    
    # 简化的IoU计算
    class_intersect = torch.zeros(num_categories)
    class_union = torch.zeros(num_categories)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(track(dataloader, description="快速测试")):
            img = data["img"].to(device)
            ann = data["ann"].to(device)[:, 0, :, :]
            gps = data["gps"].to(device)
            
            # 前向传播（可选择是否更新记忆）
            update_memory = not args.skip_memory_update
            outputs = model(img, gps, return_embeddings=False, update_memory=update_memory)
            pred = outputs['segmentation_logits']
            memory_weight = outputs.get('memory_weight', 0)
            
            # 快速准确率计算
            pred_labels = pred.argmax(1)
            correct_pixels += (pred_labels == ann).sum().item()
            total_pixels += ann.numel()
            
            # 简化的类别IoU计算
            for c in range(num_categories):
                pred_c = (pred_labels == c)
                gt_c = (ann == c)
                class_intersect[c] += (pred_c & gt_c).sum().item()
                class_union[c] += (pred_c | gt_c).sum().item()
            
            total_memory_weight += memory_weight
            total_samples += img.shape[0]
    
    # 计算结果
    pixel_accuracy = correct_pixels / total_pixels
    class_iou = class_intersect / (class_union + 1e-8)
    mean_iou = class_iou.mean().item()
    avg_memory_weight = total_memory_weight / total_samples
    
    # 获取记忆库统计
    memory_stats = model.get_memory_stats()
    
    console.print(f"\n📈 快速测试结果:")
    console.print(f"  像素准确率: {pixel_accuracy:.5f}")
    console.print(f"  mIoU: {mean_iou:.5f}")
    console.print(f"  平均记忆权重: {avg_memory_weight:.4f}")
    console.print(f"  记忆库状态: {memory_stats['total_locations']} 位置, {memory_stats['total_memories']} 记忆")
    
    return {
        'mode': 'fast',
        'pixel_accuracy': pixel_accuracy,
        'miou': mean_iou,
        'memory_weight': avg_memory_weight,
        'memory_stats': {'final': memory_stats}
    }


def benchmark_mode_test(model, dataloader, device: str, args: Namespace) -> Dict[str, Any]:
    """基准测试模式：主要测试速度性能"""
    console.print(f"\n[bold magenta]🏃 基准测试模式 (测试速度)")
    console.print(f"  测试样本数: {len(dataloader)}")
    
    import time
    
    model.eval()
    
    # 预热
    console.print("🔥 预热模型...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 5:  # 预热5个批次
                break
            img = data["img"].to(device)
            gps = data["gps"].to(device)
            _ = model(img, gps, update_memory=False)
    
    # 正式基准测试
    times_inference = []
    times_memory = []
    memory_weights = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(track(dataloader, description="基准测试")):
            if batch_idx >= 100:  # 只测试100个样本
                break
                
            img = data["img"].to(device)
            gps = data["gps"].to(device)
            
            # 测试纯推理速度
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            outputs = model(img, gps, update_memory=False)
            torch.cuda.synchronize() if device == 'cuda' else None
            inference_time = time.time() - start_time
            times_inference.append(inference_time)
            
            # 测试包含记忆更新的速度
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            outputs = model(img, gps, update_memory=True)
            torch.cuda.synchronize() if device == 'cuda' else None
            memory_time = time.time() - start_time
            times_memory.append(memory_time)
            
            memory_weights.append(outputs.get('memory_weight', 0))
    
    # 计算统计
    avg_inference_time = np.mean(times_inference) * 1000  # ms
    avg_memory_time = np.mean(times_memory) * 1000  # ms
    memory_overhead = avg_memory_time - avg_inference_time
    avg_memory_weight = np.mean(memory_weights)
    
    fps_inference = 1000 / avg_inference_time
    fps_memory = 1000 / avg_memory_time
    
    console.print(f"\n⚡ 基准测试结果:")
    console.print(f"  纯推理时间: {avg_inference_time:.2f} ms ({fps_inference:.1f} FPS)")
    console.print(f"  记忆模式时间: {avg_memory_time:.2f} ms ({fps_memory:.1f} FPS)")
    console.print(f"  记忆开销: {memory_overhead:.2f} ms ({memory_overhead/avg_inference_time*100:.1f}%)")
    console.print(f"  平均记忆权重: {avg_memory_weight:.4f}")
    
    return {
        'mode': 'benchmark', 
        'inference_time_ms': avg_inference_time,
        'memory_time_ms': avg_memory_time,
        'memory_overhead_ms': memory_overhead,
        'fps_inference': fps_inference,
        'fps_memory': fps_memory,
        'memory_weight': avg_memory_weight
    }
def progressive_mode_test(model, dataloader, device: str, args: Namespace) -> Dict[str, Any]:
    """渐进式测试：边测试边建立新记忆（优化版）"""
    console.print(f"\n[bold yellow]🧪 渐进式测试 (建立新记忆)")
    console.print(f"  测试样本数: {len(dataloader)}")
    console.print(f"  评估间隔: {args.eval_interval}")
    
    num_categories = len(Category.load(args.category_csv))
    if any(cat.id == 255 for cat in Category.load(args.category_csv)):
        num_categories -= 1
    
    model.eval()
    total_memory_weight = 0
    total_samples = 0
    predictions = []
    
    # 记录进度（减少记录频率）
    progress_log = {
        'samples': [],
        'memory_locations': [],
        'memory_count': [],
        'hit_rates': [],
        'memory_weights': []
    }
    
    # 获取初始记忆库状态
    initial_memory_stats = model.get_memory_stats()
    console.print(f"📊 初始记忆库状态:")
    console.print(f"  位置数: {initial_memory_stats['total_locations']}")
    console.print(f"  记忆数: {initial_memory_stats['total_memories']}")
    console.print(f"  命中率: {initial_memory_stats['hit_rate']:.4f}")
    
    # 分批计算指标以节省内存和时间
    metrics = None
    if not args.fast_eval:
        metrics = Metrics(num_categories, nan_to_num=0)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(track(dataloader, description="渐进式测试")):
            img = data["img"].to(device)
            ann = data["ann"].to(device)[:, 0, :, :]
            gps = data["gps"].to(device)
            filename = data.get("filename", [f"sample_{batch_idx}"])
            
            # 前向传播并更新记忆（关键：update_memory=True）  
            outputs = model(img, gps, return_embeddings=False, update_memory=True)
            pred = outputs['segmentation_logits']
            memory_weight = outputs.get('memory_weight', 0)
            pred_labels = pred.argmax(1)
            
            # 只在需要时计算详细指标
            if not args.fast_eval and metrics is not None:
                metrics.compute_and_accum(pred_labels, ann)
            
            total_memory_weight += memory_weight
            total_samples += img.shape[0]
            
            # 减少记录频率
            if batch_idx % args.eval_interval == 0:
                current_stats = model.get_memory_stats()
                progress_log['samples'].append(batch_idx + 1)
                progress_log['memory_locations'].append(current_stats['total_locations'])
                progress_log['memory_count'].append(current_stats['total_memories'])
                progress_log['hit_rates'].append(current_stats['hit_rate'])
                progress_log['memory_weights'].append(memory_weight)
                
                console.print(f"  样本 {batch_idx + 1}: 记忆库 {current_stats['total_locations']} 位置, "
                             f"命中率 {current_stats['hit_rate']:.3f}, 记忆权重 {memory_weight:.3f}")
            
            # 只保存部分预测结果
            if args.save_predictions and len(predictions) < 50:
                predictions.append({
                    'filename': filename[0] if isinstance(filename, list) else filename,
                    'pred': pred_labels.cpu(),
                    'ann': ann.cpu(),
                    'memory_weight': memory_weight
                })
    
    # 获取结果
    results = {}
    if not args.fast_eval and metrics is not None:
        results = metrics.get_and_reset()
        miou = results['IoU'].mean()
        dice = results['Dice'].mean()
        accuracy = results['aAcc']
    else:
        # 快速评估模式
        miou = dice = accuracy = 0.0
    
    avg_memory_weight = total_memory_weight / total_samples
    
    # 获取最终记忆库状态
    final_memory_stats = model.get_memory_stats()
    
    console.print(f"\n📈 渐进式测试结果:")
    if not args.fast_eval:
        console.print(f"  mIoU: {miou:.5f}")
        console.print(f"  mDice: {dice:.5f}")
        console.print(f"  整体准确率: {accuracy:.5f}")
    console.print(f"  平均记忆权重: {avg_memory_weight:.4f}")
    console.print(f"📊 记忆库变化:")
    console.print(f"  位置数: {initial_memory_stats['total_locations']} → {final_memory_stats['total_locations']}")
    console.print(f"  记忆数: {initial_memory_stats['total_memories']} → {final_memory_stats['total_memories']}")
    console.print(f"  命中率: {initial_memory_stats['hit_rate']:.4f} → {final_memory_stats['hit_rate']:.4f}")
    
    return {
        'mode': 'progressive',
        'miou': miou,
        'dice': dice,
        'accuracy': accuracy,
        'memory_weight': avg_memory_weight,
        'detailed_results': results,
        'predictions': predictions,
        'memory_stats': {
            'initial': initial_memory_stats,
            'final': final_memory_stats
        },
        'progress_log': progress_log
    }


def save_predictions(predictions: List[Dict], categories: List, save_dir: str):
    """保存预测结果可视化"""
    console.print(f"\n[bold cyan]💾 保存预测结果...")
    
    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    visualizer = IdMapVisualizer(categories)
    img_saver = ImgSaver(pred_dir, visualizer)
    
    for i, pred_data in enumerate(predictions[:20]):  # 只保存前20个
        filename = pred_data['filename']
        pred = pred_data['pred']
        ann = pred_data['ann']
        memory_weight = pred_data['memory_weight']
        
        # 保存预测和标注
        img_saver.save_pred(pred[None, :], f"{filename}_pred.png")
        img_saver.save_ann(ann, f"{filename}_gt.png")
        
        console.print(f"  {filename}: 记忆权重 {memory_weight:.4f}")
    
    console.print(f"✅ 预测结果保存到: {pred_dir}")


def save_results(results: Dict, args: Namespace):
    """保存测试结果"""
    import json
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存数值结果
    results_file = os.path.join(args.save_dir, f"test_results_{results['mode']}.json")
    
    # 准备可序列化的结果
    serializable_results = {
        'mode': results['mode'],
        'miou': float(results['miou']),
        'dice': float(results['dice']),
        'accuracy': float(results['accuracy']),
        'memory_weight': float(results['memory_weight']),
        'memory_stats': results['memory_stats']
    }
    
    if 'progress_log' in results:
        serializable_results['progress_log'] = results['progress_log']
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 生成文本报告
    report_file = os.path.join(args.save_dir, f"test_report_{results['mode']}.txt")
    with open(report_file, 'w') as f:
        f.write(f"Memory-Enhanced GeoSegformer Test Report ({results['mode'].upper()} Mode)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Test Data: {args.img_dir}\n")
        f.write(f"  GPS Data: {args.gps_csv}\n")
        f.write(f"  Test Mode: {results['mode']}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")
        
        f.write("Performance Results:\n")
        f.write(f"  mIoU: {results['miou']:.5f}\n")
        f.write(f"  mDice: {results['dice']:.5f}\n")
        f.write(f"  Overall Accuracy: {results['accuracy']:.5f}\n")
        f.write(f"  Average Memory Weight: {results['memory_weight']:.4f}\n\n")
        
        f.write("Memory Statistics:\n")
        initial_stats = results['memory_stats']['initial']
        final_stats = results['memory_stats']['final']
        f.write(f"  Initial Locations: {initial_stats['total_locations']}\n")
        f.write(f"  Final Locations: {final_stats['total_locations']}\n")
        f.write(f"  Initial Memories: {initial_stats['total_memories']}\n")
        f.write(f"  Final Memories: {final_stats['total_memories']}\n")
        f.write(f"  Initial Hit Rate: {initial_stats['hit_rate']:.4f}\n")
        f.write(f"  Final Hit Rate: {final_stats['hit_rate']:.4f}\n")
    
    console.print(f"✅ 结果保存到:")
    console.print(f"  数值结果: {results_file}")
    console.print(f"  文本报告: {report_file}")


def main(args: Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"[bold green]🚀 记忆增强版 GeoSegformer 测试")
    console.print(f"  设备: {device}")
    console.print(f"  测试模式: {args.test_mode}")
    
    # 加载类别
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    console.print(f"  类别数: {num_categories}")
    
    # 设置GPS正规化（必须和训练时保持一致）
    console.print(f"\n📍 设置GPS正规化...")
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv, 
        args.val_gps_csv,
        method=args.gps_norm_method
    )
    
    # 数据变换
    image_size = 1080, 1920
    transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,  # 关键：使用和训练时相同的GPS正规化
        transform.Resize(image_size),
        transform.Normalize(),
    ]
    
    # 创建测试数据集
    console.print(f"\n📂 加载测试数据...")
    dataset = MemoryEnhancedGeoSegDataset(
        transforms=transforms,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        gps_csv=args.gps_csv,
        max_len=args.max_len,
    )
    
    dataloader = dataset.get_loader(
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=args.num_workers
    )
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, num_categories, args, device)
    
    # 根据测试模式进行测试
    if args.test_mode == "inference":
        results = inference_mode_test(model, dataloader, device, args)
    elif args.test_mode == "progressive":
        results = progressive_mode_test(model, dataloader, device, args)
    elif args.test_mode == "fast":
        results = fast_mode_test(model, dataloader, device, args)
    elif args.test_mode == "benchmark":
        results = benchmark_mode_test(model, dataloader, device, args)
    else:  # memory_aware
        results = memory_aware_test(model, dataloader, device, args, match_info)
        results['overlap_stats'] = overlap_stats
    
    # 保存预测结果
    if args.save_predictions and results['predictions']:
        save_predictions(results['predictions'], categories, args.save_dir)
    
    # 保存测试结果
    save_results(results, args)
    
    # 显示总结
    console.print(f"\n[bold green]🎉 测试完成！")
    console.print(f"📊 {args.test_mode.upper()} 模式结果:")
    console.print(f"  mIoU: {results['miou']:.5f}")
    console.print(f"  记忆权重: {results['memory_weight']:.4f}")
    if args.test_mode == "progressive":
        initial = results['memory_stats']['initial']
        final = results['memory_stats']['final']
        console.print(f"  记忆增长: {initial['total_locations']} → {final['total_locations']} 位置")
    console.print(f"💾 结果保存在: {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)