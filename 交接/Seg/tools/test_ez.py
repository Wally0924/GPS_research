import os
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any

import torch
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset
from engine.metric import Metrics
from engine.geo_v2 import create_memory_enhanced_geo_segformer
from engine.visualizer import IdMapVisualizer, ImgSaver


class MemoryEnhancedGeoSegDataset(ImgAnnDataset):
    """記憶增強版測試數據集"""
    def __init__(self, transforms: list, img_dir: str, ann_dir: str, gps_csv: str, max_len: int = None):
        super().__init__(transforms, img_dir, ann_dir, max_len)
        
        # 載入 GPS 數據
        self.gps_data = pd.read_csv(gps_csv)
        self.filename_to_gps = {}
        for _, row in self.gps_data.iterrows():
            filename = os.path.splitext(row['filename'])[0]
            self.filename_to_gps[filename] = [row['lat'], row['long']]
        
        print(f"✅ Loaded GPS data for {len(self.filename_to_gps)} images")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = super().__getitem__(idx)
        img_path = self.img_ann_paths[idx][0]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        if filename in self.filename_to_gps:
            gps_coords = self.filename_to_gps[filename]
            data['gps'] = torch.tensor(gps_coords, dtype=torch.float32)
        else:
            print(f"⚠️ Warning: No GPS data found for {filename}")
            data['gps'] = torch.zeros(2, dtype=torch.float32)
        
        data['filename'] = filename
        return data


def setup_gps_normalization(train_gps_csv: str, val_gps_csv: str, method: str = "minmax"):
    """設置GPS正規化"""
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    all_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    if method == "minmax":
        lat_min, lat_max = all_gps['lat'].min(), all_gps['lat'].max()
        lon_min, lon_max = all_gps['long'].min(), all_gps['long'].max()
        
        lat_range, lon_range = lat_max - lat_min, lon_max - lon_min
        padding = 0.01
        
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
    parser = ArgumentParser(description="Simple GeoSegformer Model Testing")
    
    # 必要參數
    parser.add_argument("img_dir", type=str)
    parser.add_argument("ann_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("test_gps_csv", type=str)
    parser.add_argument("train_gps_csv", type=str)
    parser.add_argument("val_gps_csv", type=str)
    
    # 模型參數（手動指定，確保與訓練時一致）
    parser.add_argument("--model-size", type=str, default="b0", choices=["b0", "b1", "b2"])
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--fusion-method", type=str, default="attention", choices=["add", "concat", "attention"])
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--spatial-radius", type=float, default=0.000001)
    
    # 可選參數
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--gps-norm-method", type=str, default="minmax")
    parser.add_argument("--show-memory-stats", action="store_true")
    
    return parser.parse_args()


def main(args: Namespace):
    print("🧪 Simple GeoSegformer Model Testing")
    print("=" * 50)
    
    image_size = 720,1280
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入類別
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    print(f"📋 Categories: {num_categories} classes")
    
    # 創建模型（使用指定的參數）
    print(f"🏗️  Creating model with parameters:")
    print(f"   Model size: {args.model_size}")
    print(f"   Feature dim: {args.feature_dim}")
    print(f"   Memory size: {args.memory_size}")
    print(f"   Spatial radius: {args.spatial_radius}")
    
    model = create_memory_enhanced_geo_segformer(
        num_classes=num_categories,
        model_size=args.model_size,
        feature_dim=args.feature_dim,
        fusion_method=args.fusion_method,
        memory_size=args.memory_size,
        spatial_radius=args.spatial_radius,
        memory_save_path=None
    ).to(device)
    
    # 載入檢查點權重
    print(f"📂 Loading weights from: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', 'unknown')
            best_score = checkpoint.get('best_score', 'unknown')
            print(f"✅ Loaded checkpoint from epoch {epoch}, best score: {best_score}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded state dict directly")
    except Exception as e:
        print(f"⚠️  Using weights_only=True due to: {e}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights only")
    
    model.eval()
    
    # 設置GPS正規化
    print("🗺️  Setting up GPS normalization...")
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv, args.val_gps_csv, method=args.gps_norm_method
    )
    
    # 設置保存目錄
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"📁 Save directory: {args.save_dir}")
    
    # 數據變換
    transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,
        transform.Resize(image_size),
        transform.Normalize(),
    ]
    
    # 創建數據集
    test_dataset = MemoryEnhancedGeoSegDataset(
        transforms=transforms,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        gps_csv=args.test_gps_csv,
        max_len=args.max_len,
    )
    
    dataloader = test_dataset.get_loader(
        batch_size=args.batch_size, 
        pin_memory=False, 
        num_workers=args.num_workers
    )
    
    # 評估工具
    if args.save_dir:
        visualizer = IdMapVisualizer(categories)
        img_saver = ImgSaver(args.save_dir, visualizer)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(num_categories, nan_to_num=0)
    
    print(f"🚀 Testing on {len(dataloader)} samples...")
    
    # 開始測試
    with Progress() as prog:
        with torch.no_grad():
            task = prog.add_task("Testing", total=len(dataloader))
            avg_loss = 0
            
            for batch_idx, data in enumerate(dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                # GeoSegformer 推理
                outputs = model(img, gps, return_embeddings=False, update_memory=False)
                pred = outputs['segmentation_logits']
                
                # 計算指標
                loss = criterion(pred, ann)
                avg_loss += loss.item()
                metrics.compute_and_accum(pred.argmax(1), ann)
                
                # 保存預測
                if args.save_dir:
                    for fn, p in zip(data["img_path"], pred):
                        filename = Path(fn).stem + ".png"
                        img_saver.save_pred(p[None, :], filename)
                
                # 記憶庫統計
                if args.show_memory_stats and batch_idx % 100 == 0:
                    memory_stats = model.get_memory_stats()
                    print(f"🧠 Memory: {memory_stats['total_locations']} locations, "
                          f"Hit rate: {memory_stats['hit_rate']:.3f}")
                
                prog.update(task, advance=1)
            
            result = metrics.get_and_reset()
            avg_loss /= len(dataloader)
            prog.remove_task(task)
    
    # 顯示結果
    print("\n📊 Test Results:")
    table = Table(title="🎯 GeoSegformer Performance")
    table.add_column("Category")
    table.add_column("Acc")
    table.add_column("IoU")
    table.add_column("Dice")
    
    for cat, acc, iou, dice in zip(categories, result["Acc"], result["IoU"], result["Dice"]):
        table.add_row(cat.name, f"{acc:.5f}", f"{iou:.5f}", f"{dice:.5f}")
    
    table.add_row("Average", f"{result['Acc'].mean():.5f}", 
                  f"{result['IoU'].mean():.5f}", f"{result['Dice'].mean():.5f}")
    
    print(table)
    print(f"\n📈 Summary:")
    print(f"   Average Loss: {avg_loss:.5f}")
    print(f"   Mean IoU: {result['IoU'].mean():.5f}")
    
    # 記憶庫統計
    if args.show_memory_stats:
        final_stats = model.get_memory_stats()
        print(f"\n🧠 Memory Bank:")
        print(f"   Locations: {final_stats['total_locations']}")
        print(f"   Hit Rate: {final_stats['hit_rate']:.4f}")
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)