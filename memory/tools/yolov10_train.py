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
from engine.yolov10 import YOLOv10MemoryEnhancedGeoSegformer, create_yolov10_memory_enhanced_geo_segformer


class ContrastiveLoss(torch.nn.Module):
    """对比学习损失，用于对齐图像和 GPS 特征"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_embeds: torch.Tensor, location_embeds: torch.Tensor) -> torch.Tensor:
        if image_embeds is None or location_embeds is None:
            return torch.tensor(0.0, device=image_embeds.device if image_embeds is not None else 'cpu')
        
        batch_size = image_embeds.shape[0]
        
        if image_embeds.shape != location_embeds.shape:
            return torch.tensor(0.0, device=image_embeds.device)
        
        if torch.isnan(image_embeds).any() or torch.isnan(location_embeds).any():
            return torch.tensor(0.0, device=image_embeds.device)
        
        # 正规化嵌入
        image_embeds = F.normalize(image_embeds, dim=-1)
        location_embeds = F.normalize(location_embeds, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(image_embeds, location_embeds.T) / self.temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # 双向对比损失
        loss_i2l = F.cross_entropy(similarity, labels)
        loss_l2i = F.cross_entropy(similarity.T, labels)
        
        final_loss = (loss_i2l + loss_l2i) / 2
        
        if torch.isnan(final_loss):
            return torch.tensor(0.0, device=image_embeds.device)
        
        return final_loss


class MemoryAwareContrastiveLoss(torch.nn.Module):
    """记忆感知的对比学习损失"""
    def __init__(self, temperature: float = 0.07, spatial_threshold: float = 0.0001):
        super().__init__()
        self.temperature = temperature
        self.spatial_threshold = spatial_threshold
        
    def forward(self, image_embeds: torch.Tensor, location_embeds: torch.Tensor, gps_coords: torch.Tensor) -> torch.Tensor:
        if image_embeds is None or location_embeds is None:
            return torch.tensor(0.0, device=image_embeds.device if image_embeds is not None else 'cpu')
        
        batch_size = image_embeds.shape[0]
        
        # 计算GPS距离矩阵
        gps_distances = torch.cdist(gps_coords, gps_coords)
        
        # 正规化嵌入
        image_embeds = F.normalize(image_embeds, dim=-1)
        location_embeds = F.normalize(location_embeds, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(image_embeds, location_embeds.T) / self.temperature
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            # 找到距离较远的负样本
            far_mask = gps_distances[i] > self.spatial_threshold
            neg_indices = torch.where(far_mask)[0]
            
            if len(neg_indices) > 0:
                # 正样本（自己）
                pos_sim = similarity[i, i]
                
                # 负样本（距离较远的位置）
                neg_sims = similarity[i, neg_indices]
                
                # 对比损失
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                labels = torch.zeros(1, dtype=torch.long, device=image_embeds.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)


class MemoryEnhancedGeoSegDataset(ImgAnnDataset):
    """记忆增强版数据集"""
    def __init__(
        self,
        transforms: list,
        img_dir: str,
        ann_dir: str,
        gps_csv: str,
        max_len: int = None,
    ):
        super().__init__(transforms, img_dir, ann_dir, max_len)
        
        # 载入 GPS 数据
        self.gps_data = pd.read_csv(gps_csv)
        
        # 创建文件名到 GPS 的映射
        self.filename_to_gps = {}
        for _, row in self.gps_data.iterrows():
            filename = os.path.splitext(row['filename'])[0]
            self.filename_to_gps[filename] = [row['lat'], row['long']]
        
        print(f"✅ Loaded GPS data for {len(self.filename_to_gps)} images")
        
        # 分析GPS数据分布
        self.analyze_gps_distribution()
    
    def analyze_gps_distribution(self):
        """分析GPS数据分布"""
        lats = [coords[0] for coords in self.filename_to_gps.values()]
        lons = [coords[1] for coords in self.filename_to_gps.values()]
        
        print(f"📊 GPS数据分析:")
        print(f"  纬度范围: [{min(lats):.6f}, {max(lats):.6f}] (范围: {max(lats)-min(lats):.6f})")
        print(f"  经度范围: [{min(lons):.6f}, {max(lons):.6f}] (范围: {max(lons)-min(lons):.6f})")
        
        # 计算重复位置
        unique_positions = set()
        duplicate_count = 0
        for coords in self.filename_to_gps.values():
            coord_str = f"{coords[0]:.7f},{coords[1]:.7f}"
            if coord_str in unique_positions:
                duplicate_count += 1
            else:
                unique_positions.add(coord_str)
        
        duplicate_rate = duplicate_count / len(self.filename_to_gps) * 100
        print(f"  唯一位置数: {len(unique_positions)}")
        print(f"  重复位置率: {duplicate_rate:.2f}%")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 获取原始数据
        data = super().__getitem__(idx)
        
        # 从路径中提取文件名
        img_path = self.img_ann_paths[idx][0]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 添加 GPS 数据和文件名
        if filename in self.filename_to_gps:
            gps_coords = self.filename_to_gps[filename]
            data['gps'] = torch.tensor(gps_coords, dtype=torch.float32)
        else:
            print(f"⚠️ Warning: No GPS data found for {filename}")
            data['gps'] = torch.zeros(2, dtype=torch.float32)
        
        # 添加文件名用于跟踪
        data['filename'] = filename
        
        return data
    

def setup_gps_normalization(train_gps_csv: str, val_gps_csv: str, method: str = "minmax"):
    """设置GPS正规化"""
    
    # 合并训练和验证集计算全局统计
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    all_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    if method == "minmax":
        lat_min = all_gps['lat'].min()
        lat_max = all_gps['lat'].max()
        lon_min = all_gps['long'].min()
        lon_max = all_gps['long'].max()
        
        # 添加小量padding避免边界问题
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
    parser = ArgumentParser(description="YOLOv10 Memory-Enhanced GeoSegformer Training")
    
    # 数据路径
    parser.add_argument("train_img_dir", type=str)
    parser.add_argument("train_ann_dir", type=str)
    parser.add_argument("val_img_dir", type=str)  
    parser.add_argument("val_ann_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("train_gps_csv", type=str)
    parser.add_argument("val_gps_csv", type=str)
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("logdir", type=str)
    
    # 模型参数
    parser.add_argument("--yolo-model", type=str, default="yolov10n", 
                       choices=["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"],
                       help="YOLOv10 model variant")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--fusion-method", type=str, default="attention", 
                       choices=["add", "concat", "attention"])
    parser.add_argument("--freeze-backbone", action="store_true", 
                       help="Freeze YOLOv10 backbone weights")
    
    # 记忆库参数
    parser.add_argument("--memory-size", type=int, default=20, help="Memory size per location")
    parser.add_argument("--spatial-radius", type=float, default=0.00005, help="Spatial radius for memory")
    parser.add_argument("--gps-norm-method", type=str, default="minmax", 
                       choices=["minmax", "zscore"], help="GPS normalization method")
    
    # 损失权重
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--spatial-threshold", type=float, default=0.0001)
    
    # 训练参数
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (YOLOv10 may need smaller)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-max-len", type=int, default=None)
    parser.add_argument("--val-max-len", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--resume", type=int, default=0)
    
    # 记忆库相关
    parser.add_argument("--memory-warmup-epochs", type=int, default=2)
    parser.add_argument("--save-memory-stats", action="store_true")
    
    # YOLOv10特定参数
    parser.add_argument("--input-size", type=int, default=640, 
                       help="Input image size (YOLOv10 typically uses 640)")
    
    return parser.parse_args()


def main(args: Namespace):
    
    # YOLOv10通常使用640x640输入
    image_size = args.input_size, args.input_size  # H x W
    crop_size = 512, 512  # 可以保持512用于训练
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 载入类别定义
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    
    print(f"🚀 YOLOv10记忆增强版 GeoSegformer 训练配置:")
    print(f"  YOLO模型: {args.yolo_model}")
    print(f"  输入尺寸: {args.input_size}x{args.input_size}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  记忆库大小: {args.memory_size}")
    print(f"  空间半径: {args.spatial_radius}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  冻结backbone: {args.freeze_backbone}")
    
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv, 
        args.val_gps_csv,
        method=getattr(args, 'gps_norm_method', 'minmax')
    )
    
    # 创建修复版官方YOLOv10记忆增强模型
    memory_save_path = os.path.join(args.logdir, "yolov10_fixed_memory_stats.json") if args.save_memory_stats else None
    
    model = create_yolov10_memory_enhanced_geo_segformer(
        num_classes=num_categories,
        yolo_model=args.yolo_model,
        feature_dim=args.feature_dim,
        fusion_method=args.fusion_method,
        memory_size=args.memory_size,
        spatial_radius=args.spatial_radius,
        memory_save_path=memory_save_path,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    print(f"✅ 创建修复版官方YOLOv10记忆增强模型，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 数据变换 - 针对YOLOv10优化
    train_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,
        transform.RandomResizeCrop(image_size, (0.8, 2), crop_size),
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
    
    # 创建数据集
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
    
    # 数据加载器
    train_dataloader = train_dataset.get_loader(
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    
    val_dataloader = val_dataset.get_loader(
        batch_size=1,  # 验证时使用batch_size=1
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
    
    # 损失函数
    seg_criterion = torch.nn.CrossEntropyLoss().to(device)
    contrastive_criterion = MemoryAwareContrastiveLoss(
        temperature=args.temperature,
        spatial_threshold=args.spatial_threshold
    ).to(device)
    
    # 评估指标
    metrics = Metrics(num_categories, nan_to_num=0)
    
    # 优化器 - 支持冻结backbone选项
    if args.freeze_backbone:
        # 如果冻结backbone，只训练其他部分
        trainable_params = [
            {"params": model.location_encoder.parameters(), "lr": 3e-4},
            {"params": model.memory_fusion.parameters(), "lr": 6e-4},
            {"params": model.memory_attention.parameters(), "lr": 6e-4},
            {"params": model.cross_modal_fusion.parameters(), "lr": 6e-4},
            {"params": model.segmentation_head.parameters(), "lr": 6e-4},
            {"params": model.contrastive_proj.parameters(), "lr": 6e-4},
        ]
        print("🔒 官方YOLOv10 Backbone已冻结，只训练其他组件")
    else:
        # 训练所有参数，但给YOLOv10 backbone较低的学习率
        trainable_params = [
            {"params": model.image_encoder.parameters(), "lr": 1e-5},  # 官方权重用很低的学习率
            {"params": model.location_encoder.parameters(), "lr": 3e-4},
            {"params": model.memory_fusion.parameters(), "lr": 6e-4},
            {"params": model.memory_attention.parameters(), "lr": 6e-4},
            {"params": model.cross_modal_fusion.parameters(), "lr": 6e-4},
            {"params": model.segmentation_head.parameters(), "lr": 6e-4},
            {"params": model.contrastive_proj.parameters(), "lr": 6e-4},
        ]
        print("🔄 微调官方YOLOv10权重 + 训练其他组件")
    
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
    
    # 学习率调度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 1e-4, 1, len(train_dataloader) * args.memory_warmup_epochs
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, args.max_epochs, 1e-6
    )
    
    # 检查点恢复
    if args.resume:
        checkpoint = torch.load(
            os.path.join(args.logdir, f"checkpoint_{args.resume}.pth")
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = args.resume + 1
        print(f"✅ 从epoch {args.resume}恢复训练")
    else:
        start_epoch = 1
    
    # 创建日志目录
    if not args.resume and os.path.exists(args.logdir):
        raise FileExistsError(
            f"{args.logdir} already exists. Please specify a different logdir or resume a checkpoint."
        )
    
    os.makedirs(args.logdir, exist_ok=True)
    logger = Logger(args.logdir)
    img_saver = ImgSaver(args.logdir, IdMapVisualizer(categories))
    
    # 训练循环
    with Progress() as prog:
        whole_task = prog.add_task("YOLOv10 Memory-Enhanced Training", total=args.max_epochs)
        
        for e in range(start_epoch, args.max_epochs + 1):
            train_task = prog.add_task(f"Train - {e}", total=len(train_dataloader))
            
            # 训练阶段
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
                
                # 前向传播
                outputs = model(img, gps, return_embeddings=True, update_memory=True)
                
                # 分割损失
                seg_loss = seg_criterion(outputs['segmentation_logits'], ann)
                
                # 记忆感知对比学习损失
                contrastive_loss = contrastive_criterion(
                    outputs['image_embeddings'], 
                    outputs['location_embeddings'],
                    gps
                )
                
                # 总损失（记忆预热期间减少对比学习权重）
                contrastive_weight = args.contrastive_weight * (0.1 if is_warmup else 1.0)
                total_loss = (args.seg_weight * seg_loss + 
                             contrastive_weight * contrastive_loss)
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                if is_warmup:
                    warmup_scheduler.step()
                
                # 记录损失
                train_seg_loss += seg_loss.item()
                train_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item()
                train_total_loss += total_loss.item()
                train_memory_weight += outputs.get('memory_weight', 0)
                
                # 定期输出记忆库统计
                if batch_idx % 50 == 0:
                    memory_stats = model.get_memory_stats()
                    logger.info("Memory", 
                               f"Locations: {memory_stats['total_locations']}, "
                               f"Memories: {memory_stats['total_memories']}, "
                               f"Hit Rate: {memory_stats['hit_rate']:.3f}")
                
                prog.update(train_task, advance=1)
            
            # 计算平均值
            train_seg_loss /= len(train_dataloader)
            train_contrastive_loss /= len(train_dataloader)
            train_total_loss /= len(train_dataloader)
            train_memory_weight /= len(train_dataloader)
            
            # 记录训练结果
            logger.info("TrainLoop", f"Total Loss: {train_total_loss:.5f}")
            logger.info("TrainLoop", f"Seg Loss: {train_seg_loss:.5f}")
            logger.info("TrainLoop", f"Contrastive Loss: {train_contrastive_loss:.5f}")
            logger.info("TrainLoop", f"Memory Weight: {train_memory_weight:.4f}")
            
            logger.tb_log("TrainLoop/TotalLoss", train_total_loss, e)
            logger.tb_log("TrainLoop/SegLoss", train_seg_loss, e)
            logger.tb_log("TrainLoop/ContrastiveLoss", train_contrastive_loss, e)
            logger.tb_log("TrainLoop/MemoryWeight", train_memory_weight, e)
            
            # 保存训练样本
            if e % args.save_interval == 0:
                img_saver.save_img(img, f"train_{e}_img.png")
                img_saver.save_ann(ann, f"train_{e}_ann.png")
                img_saver.save_pred(outputs['segmentation_logits'], f"train_{e}_pred.png")
            
            prog.remove_task(train_task)
            
            # 验证阶段
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
                        
                        # 推理（不更新记忆库）
                        outputs = model(img, gps, return_embeddings=True, update_memory=False)
                        pred = outputs['segmentation_logits']
                        
                        # 计算损失
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
                        
                        # 计算评估指标
                        metrics.compute_and_accum(pred.argmax(1), ann)
                        
                        prog.update(val_task, advance=1)
                    
                    # 平均损失
                    val_seg_loss /= len(val_dataloader)
                    val_contrastive_loss /= len(val_dataloader)
                    val_total_loss /= len(val_dataloader)
                    val_memory_weight /= len(val_dataloader)
                    
                    # 保存验证样本
                    img_saver.save_img(img, f"val_{e}_img.png")
                    img_saver.save_ann(ann, f"val_{e}_ann.png")
                    img_saver.save_pred(pred, f"val_{e}_pred.png")
                    
                    # 获取评估结果
                    result = metrics.get_and_reset()
                    
                    # 创建结果表格
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
                    
                    prog.remove_task(val_task)
                    print(table)
                    
                    # 记录验证结果
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
                    
                    # 记忆库统计
                    memory_stats = model.get_memory_stats()
                    logger.info("Memory", f"Final Stats - Locations: {memory_stats['total_locations']}, "
                                         f"Memories: {memory_stats['total_memories']}, "
                                         f"Hit Rate: {memory_stats['hit_rate']:.4f}")
                    
                    logger.tb_log("Memory/TotalLocations", memory_stats['total_locations'], e)
                    logger.tb_log("Memory/TotalMemories", memory_stats['total_memories'], e)
                    logger.tb_log("Memory/HitRate", memory_stats['hit_rate'], e)
            
            # 学习率调度
            if not is_warmup:
                poly_scheduler.step()
            
            # 保存检查点
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
                
                # 保存记忆库统计
                if args.save_memory_stats:
                    model.save_memory_bank()
            
            prog.update(whole_task, advance=1)
        
        prog.remove_task(whole_task)
    
    # 训练完成后保存最终记忆库统计
    if args.save_memory_stats:
        model.save_memory_bank()
        
    # 最终统计
    final_memory_stats = model.get_memory_stats()
    print(f"\n🎉 修复版官方YOLOv10记忆增强GeoSegformer训练完成！")
    print(f"✅ 使用了官方YOLOv10预训练权重")
    print(f"📊 最终记忆库统计:")
    print(f"  总位置数: {final_memory_stats['total_locations']}")
    print(f"  总记忆数: {final_memory_stats['total_memories']}")
    print(f"  命中率: {final_memory_stats['hit_rate']:.4f}")
    print(f"  平均每位置记忆数: {final_memory_stats['avg_memories_per_location']:.2f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
                        