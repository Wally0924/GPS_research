import os
import pandas as pd
from argparse import ArgumentParser, Namespace
from typing import Dict, Any

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
#from engine.geo_v2 import MemoryEnhancedGeoSegformer, create_memory_enhanced_geo_segformer
#from engine.geo_v2 import AdversarialMemoryLoss
from engine.geo_v2_v3 import AdversarialMemoryLoss,MemoryEnhancedGeoSegformer, create_memory_enhanced_geo_segformer

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


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Memory-Enhanced GeoSegformer Training")
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
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--fusion-method", type=str, default="attention", 
                       choices=["add", "concat", "attention"])
    
    # 記憶庫參數
    parser.add_argument("--memory-size", type=int, default=20, help="Memory size per location")
    parser.add_argument("--spatial-radius", type=float, default=0.00005, help="Spatial radius for memory")
    parser.add_argument("--gps-norm-method", type=str, default="minmax", 
                       choices=["minmax", "zscore"], 
                       help="GPS normalization method")
    
    # 損失權重
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.05, help="Reduced for memory-aware training")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--spatial-threshold", type=float, default=0.0001, help="GPS distance threshold for contrastive learning")
    
    # 訓練參數
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)  # 可能需要較小的batch size
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-max-len", type=int, default=None)
    parser.add_argument("--val-max-len", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--resume", type=int, default=0)
    
    # 記憶庫相關
    parser.add_argument("--memory-warmup-epochs", type=int, default=2, help="Epochs to warm up memory bank")
    parser.add_argument("--save-memory-stats", action="store_true", help="Save memory bank statistics")

    parser.add_argument("--adversarial-weight", type=float, default=0.1, 
                       help="Weight for adversarial memory loss")
    parser.add_argument("--dependency-threshold", type=float, default=0.8,
                       help="Dependency threshold for adversarial loss")
    parser.add_argument("--independence-weight", type=float, default=0.1,
                       help="Independence weight in adversarial loss")
    parser.add_argument("--diversity-weight", type=float, default=0.05,
                       help="Diversity weight in adversarial loss")

    
    
    return parser.parse_args()


def main(args: Namespace):
    
    image_size = 1080, 1920  # H x W
    crop_size = 512, 512
    stride = 384, 384
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入類別定義
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    
    #⭐ 添加這裡：在訓練開始前分析GPS數據
    print("🔍 訓練前GPS數據分析:")
    print("=" * 50)
    
    # 導入調試函數
    from engine.geo_v2 import debug_memory_system
    
    # 分析訓練集GPS數據
    debug_memory_system(args.train_gps_csv, args.spatial_radius)
    
    print("=" * 50)
    print()
    # ⭐ 添加結束
    print(f"🚀 記憶增強版 GeoSegformer 訓練配置:")
    print(f"  GPS正規化方法: {getattr(args, 'gps_norm_method', 'minmax')}")
    print(f"  模型大小: {args.model_size}")
    print(f"  特徵維度: {args.feature_dim}")
    print(f"  記憶庫大小: {args.memory_size}")
    print(f"  空間半徑: {args.spatial_radius}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  分割權重: {args.seg_weight}")
    print(f"  對比學習權重: {args.contrastive_weight}")
    
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
        gps_normalizer,  # ⭐ 加入GPS正規化 ⭐
        transform.RandomResizeCrop(image_size, (0.5, 2), crop_size),
        transform.ColorJitter(0.3, 0.3, 0.3),
        transform.Normalize(),
    ]
    
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,  # ⭐ 加入GPS正規化 ⭐
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
        batch_size=1,  # 驗證時使用batch_size=1
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
    
    # 損失函數
    seg_criterion = torch.nn.CrossEntropyLoss().to(device)
    contrastive_criterion = MemoryAwareContrastiveLoss(
        temperature=args.temperature,
        spatial_threshold=args.spatial_threshold
    ).to(device)

    adversarial_memory_loss = AdversarialMemoryLoss(
        dependency_threshold=args.dependency_threshold,
        independence_weight=args.independence_weight,
        diversity_weight=args.diversity_weight
    ).to(device)

    # 評估指標
    metrics = Metrics(num_categories, nan_to_num=0)
    
    # 優化器 - 為記憶增強組件設置適當的學習率
    optimizer = torch.optim.AdamW([
        # Segformer backbone (較低學習率)
        {"params": model.image_encoder.parameters(), "lr": 6e-5},
        # GPS encoder (中等學習率)
        {"params": model.location_encoder.parameters(), "lr": 3e-4},
        # Memory components (較高學習率)
        {"params": model.memory_fusion.parameters(), "lr": 6e-4},
        {"params": model.memory_attention.parameters(), "lr": 6e-4},
        # Other components
        {"params": model.cross_modal_fusion.parameters(), "lr": 6e-4},
        {"params": model.segmentation_head.parameters(), "lr": 6e-4},
        {"params": model.contrastive_proj.parameters(), "lr": 6e-4},
    ])
    
    # 學習率調度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 1e-4, 1, len(train_dataloader) * args.memory_warmup_epochs
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, args.max_epochs, 1
    )
    
    # 檢查點恢復
    if args.resume:
        checkpoint = torch.load(
            os.path.join(args.logdir, f"checkpoint_{args.resume}.pth")
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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
    logger = Logger(args.logdir)
    img_saver = ImgSaver(args.logdir, IdMapVisualizer(categories))
    
    # 訓練循環
    with Progress() as prog:
        whole_task = prog.add_task("Memory-Enhanced Training", total=args.max_epochs)
        
        # 训练循环修复版本
        for e in range(start_epoch, args.max_epochs + 1):
            train_task = prog.add_task(f"Train - {e}", total=len(train_dataloader))
            
            # 训练阶段
            model.train()
            train_seg_loss = 0
            train_contrastive_loss = 0
            train_total_loss = 0
            train_memory_weight = 0
            train_adversarial_loss = 0
            
            is_warmup = e <= args.memory_warmup_epochs
            
            for batch_idx, data in enumerate(train_dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                optimizer.zero_grad()
                
                # ✅ 只进行一次前向传播
                outputs = model(
                    img, gps, 
                    return_embeddings=True, 
                    update_memory=True,
                    return_intermediate_features=True,
                    return_adversarial_features=True

                )
                if batch_idx % 10 == 0:
                    print(f"\n🔍 Debug Batch {batch_idx}:")
                    print(f"  Available keys: {list(outputs.keys())}")
                    print(f"  Memory Weight: {outputs.get('memory_weight', 'MISSING')}")
                    
                    # 檢查是否有對抗性特徵
                    required_keys = ['original_image_embeddings', 'original_memory_features', 
                                    'enhanced_image_embeddings', 'valid_memory_mask']
                    for key in required_keys:
                        if key in outputs:
                            print(f"  ✅ {key}: {outputs[key].shape}")
                        else:
                            print(f"  ❌ MISSING: {key}")
                    
                    # 檢查特徵有效性
                    if 'valid_memory_mask' in outputs:
                        valid_count = outputs['valid_memory_mask'].sum().item()
                        print(f"  Valid Memory: {valid_count}/{len(outputs['valid_memory_mask'])}")
                        
                        if valid_count > 0:
                            # 檢查特徵相似度
                            valid_orig = outputs['original_image_embeddings'][outputs['valid_memory_mask']]
                            valid_mem = outputs['original_memory_features'][outputs['valid_memory_mask']]
                            similarity = F.cosine_similarity(valid_orig, valid_mem, dim=-1).mean()
                            print(f"  Avg Similarity: {similarity:.4f} (threshold: {args.dependency_threshold})")
                                # ✅ 按正确顺序计算损失
                seg_loss = seg_criterion(outputs['segmentation_logits'], ann)
                
                contrastive_loss = contrastive_criterion(
                    outputs['image_embeddings'], 
                    outputs['location_embeddings'],
                    gps
                )
                if 'original_image_embeddings' in outputs:
                    adversarial_losses = model.compute_adversarial_loss(
                        outputs['original_image_embeddings'],
                        outputs['original_memory_features'],
                        outputs['enhanced_image_embeddings'],
                        torch.full((img.shape[0],), outputs['memory_weight'], device=device),
                        outputs['valid_memory_mask']
                    )
                    
                    # ✅ 添加調試信息 - 檢查對抗性損失
                    if batch_idx % 10 == 0:
                        print(f"  🎯 Adversarial Loss Components:")
                        for key, value in adversarial_losses.items():
                            print(f"    {key}: {value.item():.6f}")
                        print()
                else:
                    print(f"❌ ERROR: 缺少對抗性特徵，使用零損失")
                    adversarial_losses = {
                        'adversarial_memory_loss': torch.tensor(0.0, device=device),
                        'dependency_loss': torch.tensor(0.0, device=device),
                        'independence_loss': torch.tensor(0.0, device=device),
                        'diversity_loss': torch.tensor(0.0, device=device),
                        'quality_loss': torch.tensor(0.0, device=device),
                        'balance_loss': torch.tensor(0.0, device=device)
                    }
                
                # ✅ 只计算一次对抗性损失
                #adversarial_losses = model.compute_adversarial_loss(
                 #   outputs['original_image_embeddings'],
                  #  outputs['original_memory_features'],
                  #  outputs['enhanced_image_embeddings'],
                  #  torch.full((img.shape[0],), outputs['memory_weight'], device=device),
                  #  outputs['valid_memory_mask']
                #)

                
                # ✅ 总损失计算
                contrastive_weight = args.contrastive_weight * (0.1 if is_warmup else 1.0)
                adversarial_weight = args.adversarial_weight
                
                total_loss = (
                    args.seg_weight * seg_loss + 
                    contrastive_weight * contrastive_loss +
                    args.adversarial_weight * adversarial_losses['adversarial_memory_loss']
                )
                                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if is_warmup:
                    warmup_scheduler.step()
                
                # ✅ 简化损失记录
                train_seg_loss += seg_loss.item()
                train_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
                train_total_loss += total_loss.item()
                train_memory_weight += outputs.get('memory_weight', 0)
                train_adversarial_loss += adversarial_losses['adversarial_memory_loss'].item()
                
                # 定期输出
                if batch_idx % 50 == 0:
                    memory_stats = model.get_memory_stats()
                    logger.info("Memory", 
                            f"Locations: {memory_stats['total_locations']}, "
                            f"Memories: {memory_stats['total_memories']}, "
                            f"Hit Rate: {memory_stats['hit_rate']:.3f}")
                    logger.info("Adversarial", 
                            f"Dependency: {adversarial_losses['dependency_loss']:.4f}, "
                            f"Independence: {adversarial_losses['independence_loss']:.4f}, "
                            f"Diversity: {adversarial_losses['diversity_loss']:.4f}")
                
                prog.update(train_task, advance=1)
            
            # ✅ 计算平均值
            train_seg_loss /= len(train_dataloader)
            train_contrastive_loss /= len(train_dataloader)
            train_total_loss /= len(train_dataloader)
            train_memory_weight /= len(train_dataloader)
            train_adversarial_loss /= len(train_dataloader)
            
            # ✅ 记录训练结果
            logger.info("TrainLoop", f"Total Loss: {train_total_loss:.5f}")
            logger.info("TrainLoop", f"Seg Loss: {train_seg_loss:.5f}")
            logger.info("TrainLoop", f"Contrastive Loss: {train_contrastive_loss:.5f}")
            logger.info("TrainLoop", f"Memory Weight: {train_memory_weight:.4f}")
            logger.info("TrainLoop", f"Adversarial Loss: {train_adversarial_loss:.5f}")
            
            logger.tb_log("TrainLoop/TotalLoss", train_total_loss, e)
            logger.tb_log("TrainLoop/SegLoss", train_seg_loss, e)
            logger.tb_log("TrainLoop/ContrastiveLoss", train_contrastive_loss, e)
            logger.tb_log("TrainLoop/MemoryWeight", train_memory_weight, e)
            logger.tb_log("TrainLoop/AdversarialLoss", train_adversarial_loss, e)
            
            # 验证阶段修复
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
                        
                        # ✅ 修复：取消注释并正确进行推理
                        outputs = model(img, gps, return_embeddings=True, update_memory=False)
                        pred = outputs['segmentation_logits']
                        
                        # ✅ 简化验证损失计算（不需要对抗性损失）
                        seg_loss = seg_criterion(pred, ann)
                        contrastive_loss = contrastive_criterion(
                            outputs['image_embeddings'], 
                            outputs['location_embeddings'],
                            gps
                        )
                        total_loss = args.seg_weight * seg_loss + args.contrastive_weight * contrastive_loss
                        
                        val_seg_loss += seg_loss.item()
                        val_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
                        val_total_loss += total_loss.item()
                        val_memory_weight += outputs.get('memory_weight', 0)
                        
                        # 计算评估指标
                        metrics.compute_and_accum(pred.argmax(1), ann)
                        
                        prog.update(val_task, advance=1)
                    
                    prog.remove_task(val_task)
                    
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
                    
                    # 創建結果表格
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
                    
                    # 添加平均值
                    table.add_row(
                        "Avg.",
                        "{:.5f}".format(result["Acc"].mean()),
                        "{:.5f}".format(result["IoU"].mean()),
                        "{:.5f}".format(result["Dice"].mean()),
                        "{:.5f}".format(result["Fscore"].mean()),
                        "{:.5f}".format(result["Precision"].mean()),
                        "{:.5f}".format(result["Recall"].mean()),
                    )
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
            
            # 學習率調度
            if not is_warmup:
                poly_scheduler.step()
            
            # 保存檢查點
            if e % args.checkpoint_interval == 0:
                checkpoint_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": e,
                    "args": args,
                    "memory_stats": model.get_memory_stats()
                }
                
                if is_warmup:
                    checkpoint_data["warmup_scheduler"] = warmup_scheduler.state_dict()
                else:
                    checkpoint_data["poly_scheduler"] = poly_scheduler.state_dict()
                
                torch.save(
                    checkpoint_data,
                    os.path.join(args.logdir, f"checkpoint_{e}.pth"),
                )
                
                # 保存記憶庫統計
                if args.save_memory_stats:
                    model.save_memory_bank()
            
            prog.update(whole_task, advance=1)
        
        prog.remove_task(whole_task)
    
    # 訓練完成後保存最終記憶庫統計
    if args.save_memory_stats:
        model.save_memory_bank()
        
    # 最終統計
    final_memory_stats = model.get_memory_stats()
    print(f"\n🎉 記憶增強版 GeoSegformer 訓練完成！")
    print(f"📊 最終記憶庫統計:")
    print(f"  總位置數: {final_memory_stats['total_locations']}")
    print(f"  總記憶數: {final_memory_stats['total_memories']}")
    print(f"  命中率: {final_memory_stats['hit_rate']:.4f}")
    print(f"  平均每位置記憶數: {final_memory_stats['avg_memories_per_location']:.2f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)