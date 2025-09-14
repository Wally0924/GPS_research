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


class GPSOnlyGeoSegDataset(ImgAnnDataset):
    """
    GPS-only 數據集 (記憶庫禁用版本)
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
        
        # 載入GPS數據
        self.gps_data = pd.read_csv(gps_csv)
        
        # 創建檔名到GPS的映射
        self.filename_to_gps = {}
        for _, row in self.gps_data.iterrows():
            filename = os.path.splitext(row['filename'])[0]
            self.filename_to_gps[filename] = [row['lat'], row['long']]
        
        print(f"✅ Loaded GPS data for {len(self.filename_to_gps)} images")
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
        
        # 添加GPS數據
        if filename in self.filename_to_gps:
            gps_coords = self.filename_to_gps[filename]
            data['gps'] = torch.tensor(gps_coords, dtype=torch.float32)
        else:
            print(f"⚠️ Warning: No GPS data found for {filename}")
            data['gps'] = torch.zeros(2, dtype=torch.float32)
        
        data['filename'] = filename
        return data


def setup_gps_normalization(train_gps_csv: str, val_gps_csv: str, method: str = "minmax"):
    """設置GPS正規化 (與訓練時保持一致)"""
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    all_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    if method == "minmax":
        lat_min = all_gps['lat'].min()
        lat_max = all_gps['lat'].max()
        lon_min = all_gps['long'].min()
        lon_max = all_gps['long'].max()
        
        # 添加padding
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
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


def load_gps_model_from_checkpoint(checkpoint_path: str, num_categories: int, device: str):
    """載入GPS-only模型 (記憶庫禁用)"""
    print(f"📂 Loading GPS-only checkpoint from: {checkpoint_path}")
    
    try:
        # 載入檢查點
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️ Standard loading failed: {e}")
        print("🔄 Trying alternative loading method...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print("✅ Loaded weights only (without training args)")
    
    # 從檢查點中提取模型參數
    model_args = extract_model_args_from_checkpoint(checkpoint, checkpoint_path)
    
    if model_args:
        print(f"✅ Extracted model configuration:")
        print(f"   Model: {model_args['model_size']}, Feature dim: {model_args['feature_dim']}")
        print(f"   Memory: {model_args['memory_size']} (disabled), Fusion: {model_args['fusion_method']}")
        
        # 🆕 強制設定記憶庫為禁用狀態 - 關鍵修復
        model_args['memory_size'] = 0
        model_args['spatial_radius'] = 1e-6  # 設為極小值而非0，避免除零錯誤
        
        model = create_memory_enhanced_geo_segformer(
            num_classes=num_categories,
            model_size=model_args['model_size'],
            feature_dim=model_args['feature_dim'],
            fusion_method=model_args['fusion_method'],
            memory_size=model_args['memory_size'],  # 設為0禁用記憶庫
            spatial_radius=model_args['spatial_radius'],  # 極小值避免除零
            memory_save_path=None  # 不保存記憶庫
        ).to(device)
        
        # 載入權重，允許部分權重不匹配
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 過濾記憶庫相關的權重
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # 跳過記憶庫相關的權重
            if 'memory_bank' in key:
                skipped_keys.append(key)
                continue
            elif 'memory_fusion' in key:
                skipped_keys.append(key)
                continue
            elif 'memory_attention' in key:
                skipped_keys.append(key)
                continue
            else:
                filtered_state_dict[key] = value
        
        if skipped_keys:
            print(f"🚫 Skipped {len(skipped_keys)} memory-related keys:")
            for key in skipped_keys[:5]:  # 只顯示前5個
                print(f"   - {key}")
            if len(skipped_keys) > 5:
                print(f"   ... and {len(skipped_keys) - 5} more")
        
        # 載入過濾後的權重
        try:
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Model weights loaded successfully (memory components ignored)")
        except Exception as e:
            print(f"⚠️ Warning: Some weights could not be loaded: {e}")
            # 只載入匹配的權重
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in filtered_state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(matched_dict)} matching weights")
        
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        best_score = checkpoint.get('best_score', 'unknown')
        print(f"✅ GPS-only model loaded from epoch {epoch}, best score: {best_score}")
        
        return model, model_args
    
    else:
        print("⚠️ Could not extract model args, using defaults")
        model = load_model_with_default_params(checkpoint, num_categories, device)
        return model, None


def extract_model_args_from_checkpoint(checkpoint, checkpoint_path: str):
    """從檢查點中提取模型參數"""
    
    # 方法1: 檢查是否有顯式保存的args
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, 'model_size'):
            return {
                'model_size': getattr(args, 'model_size', 'b0'),
                'feature_dim': getattr(args, 'feature_dim', 512),
                'fusion_method': getattr(args, 'fusion_method', 'attention'),
                'memory_size': getattr(args, 'memory_size', 0),  # 設為0
                'spatial_radius': getattr(args, 'spatial_radius', 1e-6),  # 極小值避免除零
            }
    
    # 方法2: 檢查model_config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        return {
            'model_size': config.get('model_size', 'b0'),
            'feature_dim': config.get('feature_dim', 512),
            'fusion_method': config.get('fusion_method', 'attention'),
            'memory_size': 0,  # 強制設為0
            'spatial_radius': 1e-6,  # 極小值避免除零
        }
    
    # 方法3: 從模型權重推斷參數
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        extracted_args = infer_model_args_from_weights(model_state)
        if extracted_args:
            extracted_args['memory_size'] = 0  # 強制設為0
            extracted_args['spatial_radius'] = 1e-6  # 極小值避免除零
            return extracted_args
    
    # 方法4: 默認參數
    return {
        'model_size': 'b0',
        'feature_dim': 512,
        'fusion_method': 'attention',
        'memory_size': 0,
        'spatial_radius': 1e-6,  # 極小值避免除零
    }


def infer_model_args_from_weights(model_state_dict):
    """從模型權重推斷參數"""
    try:
        # 推斷feature_dim
        feature_dim = None
        for key, tensor in model_state_dict.items():
            if 'location_encoder.mlp.4.weight' in key:
                feature_dim = tensor.shape[0]
                break
            elif 'segmentation_head.0.weight' in key:
                feature_dim = tensor.shape[1]
                break
        
        # 推斷model_size
        model_size = "b0"
        for key, tensor in model_state_dict.items():
            if 'image_encoder.feature_fusion.0.weight' in key:
                input_channels = tensor.shape[1]
                if input_channels == 512:
                    model_size = "b0"
                elif input_channels == 1024:
                    model_size = "b1"
                break
        
        if feature_dim:
            return {
                'model_size': model_size,
                'feature_dim': feature_dim,
                'fusion_method': 'attention',
                'memory_size': 0,  # 強制設為0
                'spatial_radius': 1e-6,  # 極小值避免除零
            }
    
    except Exception as e:
        print(f"⚠️ Failed to infer from weights: {e}")
    
    return None


def load_model_with_default_params(checkpoint, num_categories: int, device: str):
    """使用默認參數載入模型"""
    print("🔧 Using default parameters for GPS-only model:")
    model_size = "b0"
    feature_dim = 512
    fusion_method = "attention"
    memory_size = 0  # 記憶庫禁用
    spatial_radius = 1e-6  # 極小值避免除零錯誤
    
    print(f"   Model: {model_size}, Feature dim: {feature_dim}")
    print(f"   Memory: {memory_size} (disabled), Fusion: {fusion_method}")
    print(f"   Spatial radius: {spatial_radius} (avoid division by zero)")
    
    model = create_memory_enhanced_geo_segformer(
        num_classes=num_categories,
        model_size=model_size,
        feature_dim=feature_dim,
        fusion_method=fusion_method,
        memory_size=memory_size,
        spatial_radius=spatial_radius,
        memory_save_path=None
    ).to(device)
    
    if 'model' in checkpoint:
        # 過濾記憶庫相關權重
        state_dict = checkpoint['model']
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if not any(mem_key in k for mem_key in ['memory_bank', 'memory_fusion', 'memory_attention'])}
        model.load_state_dict(filtered_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    best_score = checkpoint.get('best_score', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    print(f"✅ GPS-only model loaded from epoch {epoch}, best score: {best_score}")
    
    return model


def run_gps_inference(model, dataloader, device, args):
    """執行GPS-only推理 (無記憶庫)"""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(len(Category.load(args.category_csv, show=False)), nan_to_num=0)
    
    total_loss = 0
    
    with Progress() as prog:
        with torch.no_grad():
            task = prog.add_task("GPS-only Testing", total=len(dataloader))
            
            for batch_idx, data in enumerate(dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                # GPS-only推理 (記憶庫已禁用)
                outputs = model(img, gps, return_embeddings=False, update_memory=False)
                pred = outputs['segmentation_logits']
                
                # 計算損失
                loss = criterion(pred, ann)
                total_loss += loss.item()
                
                # 計算評估指標
                metrics.compute_and_accum(pred.argmax(1), ann)
                
                # 保存預測結果
                if args.save_dir:
                    for fn, p in zip(data["img_path"], pred):
                        filename = Path(fn).stem + "_gps_only.png"
                        img_saver = ImgSaver(args.save_dir, IdMapVisualizer(Category.load(args.category_csv, show=False)))
                        img_saver.save_pred(p[None, :], filename)
                
                prog.update(task, advance=1)
            
            # 獲取最終結果
            result = metrics.get_and_reset()
            avg_loss = total_loss / len(dataloader)
            
            prog.remove_task(task)
    
    return {
        'result': result,
        'avg_loss': avg_loss,
        'memory_enabled': False
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description="GPS-only Segmentation Test (Memory Bank Disabled)")
    
    # 基本參數
    parser.add_argument("img_dir", type=str, help="Test images directory")
    parser.add_argument("ann_dir", type=str, help="Test annotations directory")
    parser.add_argument("category_csv", type=str, help="Category CSV file")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint file")
    parser.add_argument("test_gps_csv", type=str, help="Test GPS CSV file")
    parser.add_argument("train_gps_csv", type=str, help="Training GPS CSV file (for normalization)")
    parser.add_argument("val_gps_csv", type=str, help="Validation GPS CSV file (for normalization)")
    
    # 測試參數
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save prediction results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum number of test samples")
    
    # GPS正規化
    parser.add_argument("--gps-norm-method", type=str, default="minmax", 
                       choices=["minmax", "zscore"], help="GPS normalization method")
    
    # 調試選項
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    
    return parser.parse_args()


def main(args: Namespace):
    print("🚫 GPS-only Segmentation Test (Memory Bank Disabled)")
    print("=" * 60)
    
    # 基本設置
    image_size = 720, 1280
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入類別
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    print(f"📋 Categories: {num_categories} classes")
    
    # 載入GPS-only模型
    model_info = load_gps_model_from_checkpoint(args.checkpoint, num_categories, device)
    if isinstance(model_info, tuple):
        model, model_args = model_info
    else:
        model = model_info
        model_args = None
    
    print(f"🚫 Memory bank status: DISABLED")
    
    # 設置GPS正規化
    print("🗺️ Setting up GPS normalization...")
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv,
        args.val_gps_csv,
        method=args.gps_norm_method
    )
    
    # 創建保存目錄
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"📁 Created save directory: {args.save_dir}")
    
    # 數據變換
    transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,
        transform.Resize(image_size),
        transform.Normalize(),
    ]
    
    # 創建測試數據集
    test_dataset = GPSOnlyGeoSegDataset(
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
    
    print(f"🚀 Starting GPS-only evaluation on {len(dataloader)} samples...")
    
    # 執行測試
    results = run_gps_inference(model, dataloader, device, args)
    
    # 顯示結果
    result = results['result']
    avg_loss = results['avg_loss']
    
    print("\n📊 GPS-only Test Results:")
    print("=" * 60)
    
    # 創建結果表格
    table = Table(title="🎯 GPS-only GeoSegformer Performance")
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
        "Average",
        "{:.5f}".format(result["Acc"].mean()),
        "{:.5f}".format(result["IoU"].mean()),
        "{:.5f}".format(result["Dice"].mean()),
        "{:.5f}".format(result["Fscore"].mean()),
        "{:.5f}".format(result["Precision"].mean()),
        "{:.5f}".format(result["Recall"].mean()),
    )
    
    print(table)
    
    print(f"\n📈 Overall Performance:")
    print(f"   Average Loss: {avg_loss:.5f}")
    print(f"   Mean IoU: {result['IoU'].mean():.5f}")
    print(f"   Mean Accuracy: {result['Acc'].mean():.5f}")
    print(f"   Model Type: GPS-only (Memory Bank Disabled)")
    
    # 保存結果
    if args.save_dir:
        results_file = os.path.join(args.save_dir, "gps_only_test_results.txt")
        with open(results_file, 'w') as f:
            f.write("GPS-only GeoSegformer Test Results (Memory Bank Disabled)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Memory Bank: DISABLED\n")
            f.write(f"Test samples: {len(dataloader)}\n")
            f.write(f"Average Loss: {avg_loss:.5f}\n")
            f.write(f"Mean IoU: {result['IoU'].mean():.5f}\n")
            f.write(f"Mean Accuracy: {result['Acc'].mean():.5f}\n")
            f.write(f"\nPer-category results:\n")
            for i, cat in enumerate(categories):
                f.write(f"{cat.name}: IoU={result['IoU'][i]:.5f}, Acc={result['Acc'][i]:.5f}\n")
        
        print(f"💾 Results saved to: {results_file}")
        
        # 保存詳細結果到CSV
        import csv
        csv_file = os.path.join(args.save_dir, "gps_only_detailed_results.csv")
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['Category', 'Category_ID', 'Accuracy', 'IoU', 'Dice', 'Fscore', 'Precision', 'Recall']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, cat in enumerate(categories):
                writer.writerow({
                    'Category': cat.name,
                    'Category_ID': cat.id,
                    'Accuracy': float(result["Acc"][i]),
                    'IoU': float(result["IoU"][i]),
                    'Dice': float(result["Dice"][i]),
                    'Fscore': float(result["Fscore"][i]),
                    'Precision': float(result["Precision"][i]),
                    'Recall': float(result["Recall"][i])
                })
            
            # 添加平均值
            writer.writerow({
                'Category': 'Average',
                'Category_ID': 'AVG',
                'Accuracy': float(result["Acc"].mean()),
                'IoU': float(result["IoU"].mean()),
                'Dice': float(result["Dice"].mean()),
                'Fscore': float(result["Fscore"].mean()),
                'Precision': float(result["Precision"].mean()),
                'Recall': float(result["Recall"].mean())
            })
        
        print(f"📊 Detailed results saved to: {csv_file}")
    
    print(f"\n✅ GPS-only testing completed successfully!")
    print(f"🚫 Memory bank was disabled during inference")


if __name__ == "__main__":
    args = parse_args()
    main(args)