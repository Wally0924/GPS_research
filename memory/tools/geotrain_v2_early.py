import os
import json
import csv
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset
from engine.logger import Logger
from engine.metric import Metrics
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.geo_v2 import MemoryEnhancedGeoSegformer, create_memory_enhanced_geo_segformer


def set_seed(seed=42):
    """設定所有隨機種子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience=10, min_delta=0, monitor='mIoU', mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, current_score):
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            return True  # Improvement detected
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
            return False  # No improvement


def save_best_metrics_to_csv(categories, result, epoch, score, logdir):
    """Save the best metrics to CSV file."""
    csv_file = os.path.join(logdir, "best_metrics_per_category.csv")
    
    # Prepare data for CSV
    data = []
    for i, cat in enumerate(categories):
        row = {
            'Category_ID': cat.id,
            'Category_Name': cat.name,
            'Category_Abbr': cat.abbr,
            'Accuracy': float(result["Acc"][i]),
            'IoU': float(result["IoU"][i]),
            'Dice': float(result["Dice"][i]),
            'Fscore': float(result["Fscore"][i]),
            'Precision': float(result["Precision"][i]),
            'Recall': float(result["Recall"][i]),
            'Epoch': epoch,
            'Best_Score': float(score)
        }
        data.append(row)
    
    # Add average row
    avg_row = {
        'Category_ID': 'Average',
        'Category_Name': 'Average',
        'Category_Abbr': 'Avg',
        'Accuracy': float(result["Acc"].mean()),
        'IoU': float(result["IoU"].mean()),
        'Dice': float(result["Dice"].mean()),
        'Fscore': float(result["Fscore"].mean()),
        'Precision': float(result["Precision"].mean()),
        'Recall': float(result["Recall"].mean()),
        'Epoch': epoch,
        'Best_Score': float(score)
    }
    data.append(avg_row)
    
    # Write to CSV
    fieldnames = ['Category_ID', 'Category_Name', 'Category_Abbr', 'Accuracy', 
                  'IoU', 'Dice', 'Fscore', 'Precision', 'Recall', 'Epoch', 'Best_Score']
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ Best metrics saved to: {csv_file}")


def save_args_to_file(args: Namespace, logdir: str):
    """Save training arguments to a text file."""
    args_dict = vars(args)
    args_file = os.path.join(logdir, "training_args.txt")
    
    with open(args_file, 'w') as f:
        f.write("Training Arguments\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nTraining started at: {__import__('datetime').datetime.now()}\n")


def save_checkpoint_geo(model, optimizer, warmup_scheduler, poly_scheduler, epoch, 
                       best_score, filepath, is_best=False, keep_only_best=False, 
                       memory_stats=None, is_warmup=False):
    """Save GeoSegformer checkpoint with memory stats."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_score': best_score,
        'memory_stats': memory_stats or {}
    }
    
    # 根據是否在warmup階段選擇調度器
    if is_warmup:
        checkpoint['warmup_scheduler'] = warmup_scheduler.state_dict()
    else:
        checkpoint['poly_scheduler'] = poly_scheduler.state_dict()
    
    if keep_only_best:
        # 只在是最佳模型時才保存
        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_filepath)
            print(f"💾 Saved best model: {best_filepath}")
    else:
        # 正常保存邏輯
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_filepath)


class ContrastiveLoss(torch.nn.Module):
    """
    對比學習損失，用於對齊影像和 GPS 特徵
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_embeds: torch.Tensor, location_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: Image embeddings, shape (batch_size, embed_dim)
            location_embeds: Location embeddings, shape (batch_size, embed_dim)
        Returns:
            Contrastive loss value
        """
        if image_embeds is None or location_embeds is None:
            return torch.tensor(0.0, device=image_embeds.device if image_embeds is not None else 'cpu')
        
        batch_size = image_embeds.shape[0]
        
        if image_embeds.shape != location_embeds.shape:
            return torch.tensor(0.0, device=image_embeds.device)
        
        if torch.isnan(image_embeds).any() or torch.isnan(location_embeds).any():
            return torch.tensor(0.0, device=image_embeds.device)
        
        # 正規化嵌入
        image_embeds = F.normalize(image_embeds, dim=-1)
        location_embeds = F.normalize(location_embeds, dim=-1)
        
        # 計算相似度矩陣
        similarity = torch.matmul(image_embeds, location_embeds.T) / self.temperature
        
        # 創建標籤（對角線為正樣本）
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # 雙向對比損失
        loss_i2l = F.cross_entropy(similarity, labels)
        loss_l2i = F.cross_entropy(similarity.T, labels)
        
        final_loss = (loss_i2l + loss_l2i) / 2
        
        if torch.isnan(final_loss):
            return torch.tensor(0.0, device=image_embeds.device)
        
        return final_loss


class MemoryAwareContrastiveLoss(torch.nn.Module):
    """
    記憶感知的對比學習損失
    考慮相似位置的樣本不應該被強制分離
    """
    def __init__(self, temperature: float = 0.07, spatial_threshold: float = 0.0001):
        super().__init__()
        self.temperature = temperature
        self.spatial_threshold = spatial_threshold
        
    def forward(self, image_embeds: torch.Tensor, location_embeds: torch.Tensor, gps_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: Image embeddings, shape (batch_size, embed_dim)
            location_embeds: Location embeddings, shape (batch_size, embed_dim)
            gps_coords: GPS coordinates, shape (batch_size, 2)
        Returns:
            Memory-aware contrastive loss value
        """
        if image_embeds is None or location_embeds is None:
            return torch.tensor(0.0, device=image_embeds.device if image_embeds is not None else 'cpu')
        
        batch_size = image_embeds.shape[0]
        
        # 計算GPS距離矩陣
        gps_distances = torch.cdist(gps_coords, gps_coords)
        
        # 正規化嵌入
        image_embeds = F.normalize(image_embeds, dim=-1)
        location_embeds = F.normalize(location_embeds, dim=-1)
        
        # 計算相似度矩陣
        similarity = torch.matmul(image_embeds, location_embeds.T) / self.temperature
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            # 找到距離較遠的負樣本（避免相近位置被強制分離）
            far_mask = gps_distances[i] > self.spatial_threshold
            neg_indices = torch.where(far_mask)[0]
            
            if len(neg_indices) > 0:
                # 正樣本（自己）
                pos_sim = similarity[i, i]
                
                # 負樣本（距離較遠的位置）
                neg_sims = similarity[i, neg_indices]
                
                # 對比損失
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                labels = torch.zeros(1, dtype=torch.long, device=image_embeds.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)


class MemoryEnhancedGeoSegDataset(ImgAnnDataset):
    """
    記憶增強版數據集
    """
    def __init__(
        self,
        transforms: list,
        img_dir: str,
        ann_dir: str,
        gps_csv: str,
        max_len: int = None,
    ):
        super().__init__(transforms, img_dir, ann_dir, max_len)
        
        # 載入 GPS 數據
        self.gps_data = pd.read_csv(gps_csv)
        
        # 創建檔名到 GPS 的映射
        self.filename_to_gps = {}
        for _, row in self.gps_data.iterrows():
            filename = os.path.splitext(row['filename'])[0]
            self.filename_to_gps[filename] = [row['lat'], row['long']]
        
        print(f"✅ Loaded GPS data for {len(self.filename_to_gps)} images")
        
        # 分析GPS數據分佈
        self.analyze_gps_distribution()
    
    def analyze_gps_distribution(self):
        """分析GPS數據分佈"""
        lats = [coords[0] for coords in self.filename_to_gps.values()]
        lons = [coords[1] for coords in self.filename_to_gps.values()]
        
        print(f"📊 GPS數據分析:")
        print(f"  緯度範圍: [{min(lats):.6f}, {max(lats):.6f}] (範圍: {max(lats)-min(lats):.6f})")
        print(f"  經度範圍: [{min(lons):.6f}, {max(lons):.6f}] (範圍: {max(lons)-min(lons):.6f})")
        
        # 計算重複位置
        unique_positions = set()
        duplicate_count = 0
        for coords in self.filename_to_gps.values():
            coord_str = f"{coords[0]:.7f},{coords[1]:.7f}"
            if coord_str in unique_positions:
                duplicate_count += 1
            else:
                unique_positions.add(coord_str)
        
        duplicate_rate = duplicate_count / len(self.filename_to_gps) * 100
        print(f"  唯一位置數: {len(unique_positions)}")
        print(f"  重複位置率: {duplicate_rate:.2f}%")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 獲取原始數據
        data = super().__getitem__(idx)
        
        # 從路徑中提取檔名
        img_path = self.img_ann_paths[idx][0]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 添加 GPS 數據和檔名
        if filename in self.filename_to_gps:
            gps_coords = self.filename_to_gps[filename]
            data['gps'] = torch.tensor(gps_coords, dtype=torch.float32)
        else:
            print(f"⚠️ Warning: No GPS data found for {filename}")
            data['gps'] = torch.zeros(2, dtype=torch.float32)
        
        # 添加檔名用於追蹤
        data['filename'] = filename
        
        return data


def setup_gps_normalization(train_gps_csv: str, val_gps_csv: str, method: str = "minmax"):
    """
    設置GPS正規化
    """
    # 合併訓練和驗證集計算全局統計
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    all_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    if method == "minmax":
        lat_min = all_gps['lat'].min()
        lat_max = all_gps['lat'].max()
        lon_min = all_gps['long'].min()
        lon_max = all_gps['long'].max()
        
        # 添加小量padding避免邊界問題
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        padding = 0.01  # 1%的padding
        
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        
        return transform.GPSNormalize(
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max)
        )
    else:
        raise ValueError(f"Method {method} not implemented yet")


# 🆕 集成功能
def ensemble_models(model_paths, save_path=None):
    """
    將多個模型的權重進行平均
    """
    print(f"🔄 Ensembling {len(model_paths)} models...")
    
    # 載入第一個模型作為基準
    first_checkpoint = torch.load(model_paths[0], map_location='cpu')
    ensembled_state_dict = OrderedDict()
    
    # 初始化：複製第一個模型的結構
    for key in first_checkpoint['model'].keys():
        ensembled_state_dict[key] = first_checkpoint['model'][key].clone().float()
    
    # 累加其他模型的權重
    for path in model_paths[1:]:
        checkpoint = torch.load(path, map_location='cpu')
        for key in ensembled_state_dict.keys():
            ensembled_state_dict[key] += checkpoint['model'][key].float()
    
    # 計算平均
    for key in ensembled_state_dict.keys():
        ensembled_state_dict[key] /= len(model_paths)
    
    # 創建新的 checkpoint
    ensembled_checkpoint = {
        'epoch': 'ensemble',
        'model': ensembled_state_dict,
        'ensemble_info': {
            'num_models': len(model_paths),
            'source_models': model_paths,
            'creation_time': str(__import__('datetime').datetime.now())
        }
    }
    
    if save_path:
        torch.save(ensembled_checkpoint, save_path)
        print(f"✅ Ensembled model saved to: {save_path}")
    
    return ensembled_state_dict


def create_ensemble_from_multi_seed(base_logdir, seeds, save_ensemble=True):
    """
    從多種子實驗中創建集成模型
    """
    model_paths = []
    
    # 收集所有最佳模型路徑
    for seed in seeds:
        seed_logdir = f"{base_logdir}_seed_{seed}"
        best_model_path = os.path.join(seed_logdir, "checkpoint_best.pth")
        
        if os.path.exists(best_model_path):
            model_paths.append(best_model_path)
            print(f"✅ Found model for seed {seed}")
        else:
            print(f"❌ Missing model for seed {seed}: {best_model_path}")
    
    if len(model_paths) < 2:
        print(f"⚠️  Only found {len(model_paths)} models, need at least 2 for ensemble")
        return None
    
    # 創建集成模型
    if save_ensemble:
        ensemble_path = f"{base_logdir}_ensemble_model.pth"
        ensembled_weights = ensemble_models(model_paths, ensemble_path)
        return ensemble_path
    else:
        ensembled_weights = ensemble_models(model_paths)
        return ensembled_weights


def save_multi_seed_summary(all_results, args):
    """保存多種子實驗的統計摘要"""
    scores = [r['best_score'] for r in all_results]
    seeds = [r['seed'] for r in all_results]
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    summary_file = f"{args.logdir}_multi_seed_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Multi-Seed GeoSegformer Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Monitor Metric: {args.monitor}\n")
        f.write(f"Seeds Used: {seeds}\n")
        f.write(f"Number of Runs: {len(all_results)}\n")
        f.write(f"Model Size: {args.model_size}\n")
        f.write(f"Feature Dim: {args.feature_dim}\n")
        f.write(f"Memory Size: {args.memory_size}\n")
        f.write(f"Spatial Radius: {args.spatial_radius}\n\n")
        f.write("Results:\n")
        f.write("-" * 30 + "\n")
        for result in all_results:
            f.write(f"Seed {result['seed']:4d}: {result['best_score']:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean ± Std: {mean_score:.4f} ± {std_score:.4f}\n")
        f.write(f"Best:       {max_score:.4f}\n")
        f.write(f"Worst:      {min_score:.4f}\n")
        f.write(f"Range:      {max_score - min_score:.4f}\n")
    
    csv_file = f"{args.logdir}_multi_seed_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'best_score', 'logdir'])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n📊 Multi-Seed Results Summary:")
    print(f"Seeds: {seeds}")
    print(f"Mean {args.monitor}: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Best: {max_score:.4f}, Worst: {min_score:.4f}")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"📊 Detailed results saved to: {csv_file}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Enhanced Memory-Enhanced GeoSegformer Training")
    
    # 數據路徑
    parser.add_argument("train_img_dir", type=str)
    parser.add_argument("train_ann_dir", type=str)
    parser.add_argument("val_img_dir", type=str)  
    parser.add_argument("val_ann_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("train_gps_csv", type=str)
    parser.add_argument("val_gps_csv", type=str)
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("logdir", type=str)
    
    # 模型參數
    parser.add_argument("--model-size", type=str, default="b0", choices=["b0", "b1", "b2"])
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--fusion-method", type=str, default="attention", 
                       choices=["add", "concat", "attention"])
    
    # 記憶庫參數
    parser.add_argument("--memory-size", type=int, default=20, help="Memory size per location")
    parser.add_argument("--spatial-radius", type=float, default=0.05, help="Spatial radius for memory")
    parser.add_argument("--gps-norm-method", type=str, default="minmax", 
                       choices=["minmax", "zscore"], 
                       help="GPS normalization method")
    
    # 損失權重
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.05, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--spatial-threshold", type=float, default=0.15, help="GPS distance threshold")
    
    # 訓練參數
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-max-len", type=int, default=None)
    parser.add_argument("--val-max-len", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--resume", type=int, default=0)
    
    # 記憶庫相關
    parser.add_argument("--memory-warmup-epochs", type=int, default=3, help="Epochs to warm up memory bank")
    parser.add_argument("--save-memory-stats", action="store_true", help="Save memory bank statistics")
    
    # 🆕 早停相關參數
    parser.add_argument("--early-stop", action="store_true", default=False, 
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, 
                       help="Number of epochs to wait for improvement")
    parser.add_argument("--min-delta", type=float, default=0.001, 
                       help="Minimum change to qualify as improvement")
    parser.add_argument("--monitor", type=str, default="mIoU", 
                       choices=["mIoU", "loss"], help="Metric to monitor for early stopping")
    
    # 🆕 檔案保存選項
    parser.add_argument("--keep-only-best", action="store_true", default=False,
                       help="Only save the best model checkpoint to save disk space")
    parser.add_argument("--save-last-checkpoint", action="store_true", default=False,
                       help="Also save the last checkpoint for resuming training")

    # 🆕 隨機種子相關參數
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 2024, 777, 999],
                       help="List of seeds for multi-seed experiments")
    
    # 🆕 集成相關參數
    parser.add_argument("--create-ensemble", action="store_true", default=True,help="Create ensemble model from multi-seed results")
    parser.add_argument("--eval-ensemble", action="store_true", default=True,help="Evaluate ensemble model performance")
    
    # 學習率參數
    parser.add_argument("--lr-backbone", type=float, default=6e-5, help="Learning rate for backbone")
    parser.add_argument("--lr-head", type=float, default=6e-4, help="Learning rate for decode head")
    parser.add_argument("--lr-gps", type=float, default=3e-4, help="Learning rate for GPS encoder")
    parser.add_argument("--lr-memory", type=float, default=6e-4, help="Learning rate for memory components")
    parser.add_argument("--lr-fusion", type=float, default=6e-4, help="Learning rate for fusion components")
    
    return parser.parse_args()


def main_training_logic(args: Namespace):
    """主要的訓練邏輯"""
    
    image_size = 720, 1280
    crop_size = 320, 320
    stride = 240, 240
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入類別定義
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    
    # 🔍 分析GPS數據
    print("🔍 訓練前GPS數據分析:")
    print("=" * 50)
    from engine.geo_v2 import debug_memory_system
    debug_memory_system(args.train_gps_csv, args.spatial_radius)
    print("=" * 50)
    print()
    
    print(f"🚀 記憶增強版 GeoSegformer 訓練配置:")
    print(f"  GPS正規化方法: {getattr(args, 'gps_norm_method', 'minmax')}")
    print(f"  模型大小: {args.model_size}")
    print(f"  特徵維度: {args.feature_dim}")
    print(f"  記憶庫大小: {args.memory_size}")
    print(f"  空間半徑: {args.spatial_radius}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  分割權重: {args.seg_weight}")
    print(f"  對比學習權重: {args.contrastive_weight}")
    
    # 設置GPS正規化
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv, 
        args.val_gps_csv,
        method=getattr(args, 'gps_norm_method', 'minmax')
    )
    
    # 創建記憶增強版模型
    memory_save_path = os.path.join(args.logdir, "memory_stats.json") if args.save_memory_stats else None
    
    model = create_memory_enhanced_geo_segformer(
        num_classes=num_categories,
        model_size=args.model_size,
        feature_dim=args.feature_dim,
        fusion_method=args.fusion_method,
        memory_size=args.memory_size,
        spatial_radius=args.spatial_radius,
        memory_save_path=memory_save_path
    ).to(device)
    
    print(f"✅ 創建記憶增強版模型，參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 數據變換
    train_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,
        transform.RandomResizeCrop(image_size, (0.5, 2), crop_size),
        transform.ColorJitter(0.3, 0.3, 0.3),
        transform.Normalize(),
    ]
    
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,
        transform.Resize(image_size),
        transform.Normalize(),
    ]
    
    # 創建數據集
    train_dataset = MemoryEnhancedGeoSegDataset(
        transforms=train_transforms,
        img_dir=args.train_img_dir,
        ann_dir=args.train_ann_dir,
        gps_csv=args.train_gps_csv,
        max_len=args.train_max_len,
    )
    
    val_dataset = MemoryEnhancedGeoSegDataset(
        transforms=val_transforms,
        img_dir=args.val_img_dir,
        ann_dir=args.val_ann_dir,
        gps_csv=args.val_gps_csv,
        max_len=args.val_max_len,
    )
    
    # 數據載入器
    train_dataloader = train_dataset.get_loader(
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    
    val_dataloader = val_dataset.get_loader(
        batch_size=1,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
    
    # 損失函數
    seg_criterion = torch.nn.CrossEntropyLoss().to(device)
    contrastive_criterion = MemoryAwareContrastiveLoss(
        temperature=args.temperature,
        spatial_threshold=args.spatial_threshold
    ).to(device)
    
    # 評估指標
    metrics = Metrics(num_categories, nan_to_num=0)
    
    # 優化器
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": args.lr_backbone},
        {"params": model.location_encoder.parameters(), "lr": args.lr_gps},
        {"params": model.memory_fusion.parameters(), "lr": args.lr_memory},
        {"params": model.memory_attention.parameters(), "lr": args.lr_memory},
        {"params": model.cross_modal_fusion.parameters(), "lr": args.lr_fusion},
        {"params": model.segmentation_head.parameters(), "lr": args.lr_head},
        {"params": model.contrastive_proj.parameters(), "lr": args.lr_fusion},
    ])
    
    # 學習率調度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 1e-4, 1, len(train_dataloader) * args.memory_warmup_epochs
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, args.max_epochs, 1
    )
    
    # 🆕 初始化早停
    early_stopping = None
    if args.early_stop:
        mode = 'max' if args.monitor == 'mIoU' else 'min'
        early_stopping = EarlyStopping(
            patience=args.patience, 
            min_delta=args.min_delta, 
            monitor=args.monitor,
            mode=mode
        )
    
    # 🆕 初始化最佳分數追蹤
    best_score = float('-inf') if args.monitor == 'mIoU' else float('inf')
    
    # 檢查點恢復
    if args.resume:
        checkpoint = torch.load(
            os.path.join(args.logdir, f"checkpoint_{args.resume}.pth")
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score = checkpoint.get('best_score', best_score)
        start_epoch = args.resume + 1
        print(f"✅ 從epoch {args.resume}恢復訓練")
    else:
        start_epoch = 1
    
    # 創建日誌目錄
    if not args.resume and os.path.exists(args.logdir):
        raise FileExistsError(
            f"{args.logdir} already exists. Please specify a different logdir or resume a checkpoint."
        )
    
    os.makedirs(args.logdir, exist_ok=True)
    
    # 🆕 保存訓練參數
    save_args_to_file(args, args.logdir)
    
    logger = Logger(args.logdir)
    img_saver = ImgSaver(args.logdir, IdMapVisualizer(categories))
    
    # 🆕 記錄訓練配置
    logger.info("Training", f"Starting training with early stopping: {args.early_stop}")
    if args.early_stop:
        logger.info("Training", f"Monitoring: {args.monitor}, Patience: {args.patience}")
    if args.keep_only_best:
        logger.info("Training", "💾 Keep only best model enabled - saving disk space")
    else:
        logger.info("Training", f"📁 Regular checkpoint saving every {args.checkpoint_interval} epochs")
    
    # 訓練循環
    with Progress() as prog:
        whole_task = prog.add_task("Memory-Enhanced Training", total=args.max_epochs)
        
        for e in range(start_epoch, args.max_epochs + 1):
            train_task = prog.add_task(f"Train - {e}", total=len(train_dataloader))
            
            # 訓練階段
            model.train()
            train_seg_loss = 0
            train_contrastive_loss = 0
            train_total_loss = 0
            train_memory_weight = 0
            
            is_warmup = e <= args.memory_warmup_epochs
            
            for batch_idx, data in enumerate(train_dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                optimizer.zero_grad()
                
                # 前向傳播
                outputs = model(img, gps, return_embeddings=True, update_memory=True)
                
                # 分割損失
                seg_loss = seg_criterion(outputs['segmentation_logits'], ann)
                
                # 記憶感知對比學習損失
                contrastive_loss = contrastive_criterion(
                    outputs['image_embeddings'], 
                    outputs['location_embeddings'],
                    gps
                )
                
                # 總損失
                contrastive_weight = args.contrastive_weight * (0.1 if is_warmup else 1.0)
                total_loss = (args.seg_weight * seg_loss + 
                             contrastive_weight * contrastive_loss)
                
                # 反向傳播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if is_warmup:
                    warmup_scheduler.step()
                
                # 記錄損失
                train_seg_loss += seg_loss.item()
                train_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
                train_total_loss += total_loss.item()
                train_memory_weight += outputs.get('memory_weight', 0)
                
                # 定期輸出記憶庫統計
                if batch_idx % 50 == 0:
                    memory_stats = model.get_memory_stats()
                    logger.info("Memory", 
                               f"Locations: {memory_stats['total_locations']}, "
                               f"Memories: {memory_stats['total_memories']}, "
                               f"Hit Rate: {memory_stats['hit_rate']:.3f}")
                
                prog.update(train_task, advance=1)
            
            # 計算平均值
            train_seg_loss /= len(train_dataloader)
            train_contrastive_loss /= len(train_dataloader)
            train_total_loss /= len(train_dataloader)
            train_memory_weight /= len(train_dataloader)
            
            # 記錄訓練結果
            logger.info("TrainLoop", f"Total Loss: {train_total_loss:.5f}")
            logger.info("TrainLoop", f"Seg Loss: {train_seg_loss:.5f}")
            logger.info("TrainLoop", f"Contrastive Loss: {train_contrastive_loss:.5f}")
            logger.info("TrainLoop", f"Memory Weight: {train_memory_weight:.4f}")
            
            logger.tb_log("TrainLoop/TotalLoss", train_total_loss, e)
            logger.tb_log("TrainLoop/SegLoss", train_seg_loss, e)
            logger.tb_log("TrainLoop/ContrastiveLoss", train_contrastive_loss, e)
            logger.tb_log("TrainLoop/MemoryWeight", train_memory_weight, e)
            
            # 保存訓練樣本
            if e % args.save_interval == 0:
                img_saver.save_img(img, f"train_{e}_img.png")
                img_saver.save_ann(ann, f"train_{e}_ann.png")
                img_saver.save_pred(outputs['segmentation_logits'], f"train_{e}_pred.png")
            
            prog.remove_task(train_task)
            
            # 驗證階段
            if e % args.val_interval == 0:
                with torch.no_grad():
                    val_task = prog.add_task(f"Val - {e}", total=len(val_dataloader))
                    model.eval()
                    
                    val_seg_loss = 0
                    val_contrastive_loss = 0
                    val_total_loss = 0
                    val_memory_weight = 0
                    
                    for data in val_dataloader:
                        img = data["img"].to(device)
                        ann = data["ann"].to(device)[:, 0, :, :]
                        gps = data["gps"].to(device)
                        
                        # 推理（不更新記憶庫）
                        outputs = model(img, gps, return_embeddings=True, update_memory=False)
                        pred = outputs['segmentation_logits']
                        
                        # 計算損失
                        seg_loss = seg_criterion(pred, ann)
                        contrastive_loss = contrastive_criterion(
                            outputs['image_embeddings'], 
                            outputs['location_embeddings'],
                            gps
                        )
                        total_loss = (args.seg_weight * seg_loss + 
                                     args.contrastive_weight * contrastive_loss)
                        
                        val_seg_loss += seg_loss.item()
                        val_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
                        val_total_loss += total_loss.item()
                        val_memory_weight += outputs.get('memory_weight', 0)
                        
                        # 計算評估指標
                        metrics.compute_and_accum(pred.argmax(1), ann)
                        
                        prog.update(val_task, advance=1)
                    
                    # 平均損失
                    val_seg_loss /= len(val_dataloader)
                    val_contrastive_loss /= len(val_dataloader)
                    val_total_loss /= len(val_dataloader)
                    val_memory_weight /= len(val_dataloader)
                    
                    # 保存驗證樣本
                    img_saver.save_img(img, f"val_{e}_img.png")
                    img_saver.save_ann(ann, f"val_{e}_ann.png")
                    img_saver.save_pred(pred, f"val_{e}_pred.png")
                    
                    # 獲取評估結果
                    result = metrics.get_and_reset()
                    current_miou = result["IoU"].mean()
                    
                    # 🆕 創建結果表格
                    table = Table()
                    table.add_column("Category")
                    table.add_column("Acc")
                    table.add_column("IoU")
                    table.add_column("Dice")
                    table.add_column("Fscore")
                    table.add_column("Precision")
                    table.add_column("Recall")
                    
                    for cat, acc, iou, dice, fs, pre, rec in zip(
                        categories,
                        result["Acc"],
                        result["IoU"],
                        result["Dice"],
                        result["Fscore"],
                        result["Precision"],
                        result["Recall"],
                    ):
                        table.add_row(
                            cat.name,
                            "{:.5f}".format(acc),
                            "{:.5f}".format(iou),
                            "{:.5f}".format(dice),
                            "{:.5f}".format(fs),
                            "{:.5f}".format(pre),
                            "{:.5f}".format(rec),
                        )
                    
                    table.add_row(
                        "Avg.",
                        "{:.5f}".format(result["Acc"].mean()),
                        "{:.5f}".format(result["IoU"].mean()),
                        "{:.5f}".format(result["Dice"].mean()),
                        "{:.5f}".format(result["Fscore"].mean()),
                        "{:.5f}".format(result["Precision"].mean()),
                        "{:.5f}".format(result["Recall"].mean()),
                    )
                    
                    prog.remove_task(val_task)
                    print(table)
                    
                    # 記錄驗證結果
                    logger.info("ValLoop", f"Total Loss: {val_total_loss:.5f}")
                    logger.info("ValLoop", f"Seg Loss: {val_seg_loss:.5f}")
                    logger.info("ValLoop", f"Contrastive Loss: {val_contrastive_loss:.5f}")
                    logger.info("ValLoop", f"Memory Weight: {val_memory_weight:.4f}")
                    logger.info("ValLoop", f"mIoU: {result['IoU'].mean():.5f}")
                    
                    logger.tb_log("ValLoop/TotalLoss", val_total_loss, e)
                    logger.tb_log("ValLoop/SegLoss", val_seg_loss, e)
                    logger.tb_log("ValLoop/ContrastiveLoss", val_contrastive_loss, e)
                    logger.tb_log("ValLoop/MemoryWeight", val_memory_weight, e)
                    logger.tb_log("ValLoop/mIoU", result["IoU"].mean(), e)
                    
                    # 記憶庫統計
                    memory_stats = model.get_memory_stats()
                    logger.info("Memory", f"Final Stats - Locations: {memory_stats['total_locations']}, "
                                         f"Memories: {memory_stats['total_memories']}, "
                                         f"Hit Rate: {memory_stats['hit_rate']:.4f}")
                    
                    logger.tb_log("Memory/TotalLocations", memory_stats['total_locations'], e)
                    logger.tb_log("Memory/TotalMemories", memory_stats['total_memories'], e)
                    logger.tb_log("Memory/HitRate", memory_stats['hit_rate'], e)
                    
                    # 🆕 確定當前分數用於早停
                    current_score = current_miou if args.monitor == 'mIoU' else val_total_loss
                    
                    # 🆕 檢查是否是最佳模型
                    is_best = False
                    if args.monitor == 'mIoU':
                        is_best = current_score > best_score
                    else:
                        is_best = current_score < best_score
                    
                    if is_best:
                        best_score = current_score
                        logger.info("Training", f"New best {args.monitor}: {best_score:.5f}")
                        save_best_metrics_to_csv(categories, result, e, best_score, args.logdir)
                    
                    # 🆕 保存檢查點
                    checkpoint_path = os.path.join(args.logdir, f"checkpoint_{e}.pth")
                    save_checkpoint_geo(model, optimizer, warmup_scheduler, poly_scheduler, 
                                      e, best_score, checkpoint_path, is_best, args.keep_only_best,
                                      memory_stats, is_warmup)
                    
                    # 🆕 早停檢查
                    if early_stopping is not None:
                        improved = early_stopping(current_score)
                        if not improved:
                            logger.info("Training", f"No improvement for {early_stopping.wait}/{args.patience} epochs")
                        
                        if early_stopping.early_stop:
                            logger.info("Training", f"Early stopping triggered at epoch {e}")
                            logger.info("Training", f"Best {args.monitor}: {best_score:.5f}")
                            break
            
            # 學習率調度
            if not is_warmup:
                poly_scheduler.step()
            
            # 保存檢查點
            if e % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(args.logdir, f"checkpoint_{e}.pth")
                if not args.keep_only_best:
                    memory_stats = model.get_memory_stats()
                    save_checkpoint_geo(model, optimizer, warmup_scheduler, poly_scheduler, 
                                      e, best_score, checkpoint_path, memory_stats=memory_stats,
                                      is_warmup=is_warmup)
            
            # 🆕 保存最後檢查點
            if args.save_last_checkpoint or not args.keep_only_best:
                last_checkpoint_path = os.path.join(args.logdir, "checkpoint_last.pth")
                memory_stats = model.get_memory_stats()
                save_checkpoint_geo(model, optimizer, warmup_scheduler, poly_scheduler, 
                                  e, best_score, last_checkpoint_path, memory_stats=memory_stats,
                                  is_warmup=is_warmup)
            
            prog.update(whole_task, advance=1)
        
        prog.remove_task(whole_task)
    
    # 訓練完成後保存最終記憶庫統計
    if args.save_memory_stats:
        model.save_memory_bank()
        
    # 最終統計
    final_memory_stats = model.get_memory_stats()
    logger.info("Training", "Training completed!")
    logger.info("Training", f"Final best {args.monitor}: {best_score:.5f}")
    
    print(f"\n🎉 記憶增強版 GeoSegformer 訓練完成！")
    print(f"📊 最終記憶庫統計:")
    print(f"  總位置數: {final_memory_stats['total_locations']}")
    print(f"  總記憶數: {final_memory_stats['total_memories']}")
    print(f"  命中率: {final_memory_stats['hit_rate']:.4f}")
    print(f"  平均每位置記憶數: {final_memory_stats['avg_memories_per_location']:.2f}")
    
    return best_score


def run_single_seed_experiment(args: Namespace, seed: int):
    """執行單一種子的實驗"""
    set_seed(seed)
    
    original_logdir = args.logdir
    seed_logdir = f"{original_logdir}_seed_{seed}"
    args.logdir = seed_logdir
    
    print(f"\n🎲 Running GeoSegformer experiment with seed {seed}")
    print(f"📁 Results will be saved to: {seed_logdir}")
    
    best_score = main_training_logic(args)
    
    args.logdir = original_logdir
    
    return {
        'seed': seed,
        'best_score': best_score,
        'logdir': seed_logdir
    }


def run_multi_seed_experiment(args: Namespace):
    """執行多種子實驗"""
    seeds = args.seeds
    all_results = []
    
    print(f"🎯 Starting multi-seed GeoSegformer experiment with seeds: {seeds}")
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"GeoSegformer Experiment {i+1}/{len(seeds)} - Seed: {seed}")
        print(f"{'='*50}")
        
        try:
            result = run_single_seed_experiment(args, seed)
            all_results.append(result)
            print(f"✅ Completed seed {seed}, best {args.monitor}: {result['best_score']:.4f}")
        except Exception as e:
            print(f"❌ Failed seed {seed}: {e}")
            continue
    
    # 計算統計結果
    if all_results:
        save_multi_seed_summary(all_results, args)
        
        # 🆕 創建集成模型
        if args.create_ensemble and len(all_results) >= 2:
            print(f"\n🔗 Creating ensemble model...")
            ensemble_path = create_ensemble_from_multi_seed(
                args.logdir, 
                [r['seed'] for r in all_results], 
                save_ensemble=True
            )
            
            if ensemble_path:
                print(f"🎉 Ensemble model created: {ensemble_path}")
                
                # 更新摘要文件，加入集成結果
                summary_file = f"{args.logdir}_multi_seed_summary.txt"
                with open(summary_file, 'a') as f:
                    f.write(f"\nEnsemble Model Results:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Ensemble model path: {ensemble_path}\n")
            else:
                print("❌ Failed to create ensemble model")
        else:
            if not args.create_ensemble:
                print("ℹ️  Ensemble creation disabled")
            else:
                print("⚠️  Need at least 2 successful runs for ensemble")
    
    return all_results


def main(args: Namespace):
    """主函數：決定執行單種子還是多種子實驗"""
    if args.multi_seed:
        # 多種子實驗
        run_multi_seed_experiment(args)
    else:
        # 單種子實驗
        result = run_single_seed_experiment(args, args.seed)
        print(f"✅ GeoSegformer training completed with seed {args.seed}")
        print(f"Final best {args.monitor}: {result['best_score']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)