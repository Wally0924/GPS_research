import os
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any
import numpy as np

import torch
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset
from engine.metric import Metrics
from engine.geo_v2_memory import create_memory_enhanced_geo_segformer
from engine.visualizer import IdMapVisualizer, ImgSaver


class MemoryEnhancedGeoSegDataset(ImgAnnDataset):
    """
    記憶增強版測試數據集
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


def manual_gps_normalization_exact_training_params(train_gps_csv: str, val_gps_csv: str, method: str = "minmax"):
    """
    🆕 手動設置與訓練時完全相同的GPS正規化參數
    確保與 geotrain_v2_early_v1.py 中的 setup_gps_normalization 邏輯完全一致
    """
    print(f"🗺️  手動設置GPS正規化參數 (method: {method})")
    print("   ⚠️  確保與訓練時參數完全一致!")
    
    # 🔑 關鍵：使用與訓練時完全相同的數據載入和處理邏輯
    train_gps = pd.read_csv(train_gps_csv)
    val_gps = pd.read_csv(val_gps_csv)
    all_gps = pd.concat([train_gps, val_gps], ignore_index=True)
    
    if method == "minmax":
        # 🔑 關鍵：與訓練時完全相同的計算方式
        lat_min = all_gps['lat'].min()
        lat_max = all_gps['lat'].max()
        lon_min = all_gps['long'].min()
        lon_max = all_gps['long'].max()
        
        # 🔑 關鍵：使用完全相同的padding計算
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        padding = 0.01  # 必須與訓練時完全一致
        
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        
        print(f"📋 計算的GPS正規化參數:")
        print(f"   原始範圍 - 緯度: [{all_gps['lat'].min():.8f}, {all_gps['lat'].max():.8f}]")
        print(f"   原始範圍 - 經度: [{all_gps['long'].min():.8f}, {all_gps['long'].max():.8f}]")
        print(f"   Padding: {padding}")
        print(f"   最終範圍 - 緯度: [{lat_min:.8f}, {lat_max:.8f}]")
        print(f"   最終範圍 - 經度: [{lon_min:.8f}, {lon_max:.8f}]")
        
        return transform.GPSNormalize(
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max)
        )
    else:
        raise ValueError(f"Method {method} not implemented in manual mode")


def analyze_gps_quantization_matching(
    train_gps_csv: str, 
    test_gps_csv: str, 
    spatial_radius: float,
    gps_normalizer,
    model_memory_bank=None
):
    """
    🆕 分析GPS量化匹配情況（包含正規化）
    """
    
    def gps_to_key_with_normalization(lat, lon, normalizer, radius):
        """模擬完整的GPS處理流程：正規化 + 量化"""
        # 1. 正規化
        original = torch.tensor([lat, lon], dtype=torch.float32)
        data = {'gps': original}
        normalized_data = normalizer(data)
        normalized_gps = normalized_data['gps']
        
        # 2. 量化
        lat_grid = round(normalized_gps[0].item() / radius) * radius
        lon_grid = round(normalized_gps[1].item() / radius) * radius
        return f"{lat_grid:.7f},{lon_grid:.7f}", normalized_gps.numpy()
    
    # 載入GPS數據
    train_gps = pd.read_csv(train_gps_csv)
    test_gps = pd.read_csv(test_gps_csv)
    
    print(f"📊 GPS量化匹配分析 (包含正規化)")
    print(f"   spatial_radius: {spatial_radius:.7f}")
    print("=" * 60)
    
    # 1. 建立訓練集的量化鍵集合
    train_keys = set()
    train_key_counts = {}
    train_normalized_coords = []
    
    for _, row in train_gps.iterrows():
        key, normalized = gps_to_key_with_normalization(
            row['lat'], row['long'], gps_normalizer, spatial_radius
        )
        train_keys.add(key)
        train_key_counts[key] = train_key_counts.get(key, 0) + 1
        train_normalized_coords.append(normalized)
    
    print(f"📍 訓練集統計:")
    print(f"   總GPS記錄: {len(train_gps)}")
    print(f"   唯一量化位置: {len(train_keys)}")
    print(f"   平均每位置樣本數: {len(train_gps) / len(train_keys):.2f}")
    
    # 2. 分析測試集的量化匹配
    test_matches = 0
    test_keys = set()
    test_key_counts = {}
    test_normalized_coords = []
    match_details = []
    
    for idx, row in test_gps.iterrows():
        key, normalized = gps_to_key_with_normalization(
            row['lat'], row['long'], gps_normalizer, spatial_radius
        )
        test_keys.add(key)
        test_key_counts[key] = test_key_counts.get(key, 0) + 1
        test_normalized_coords.append(normalized)
        
        is_match = key in train_keys
        if is_match:
            test_matches += 1
            
        match_details.append({
            'idx': idx,
            'filename': row['filename'],
            'original_lat': row['lat'],
            'original_long': row['long'],
            'normalized_lat': normalized[0],
            'normalized_long': normalized[1],
            'quantized_key': key,
            'matches_train': is_match,
            'train_count': train_key_counts.get(key, 0)
        })
    
    print(f"\n🎯 測試集匹配統計:")
    print(f"   總GPS記錄: {len(test_gps)}")
    print(f"   唯一量化位置: {len(test_keys)}")
    print(f"   匹配訓練集的記錄: {test_matches}")
    print(f"   匹配率: {test_matches / len(test_gps) * 100:.2f}%")
    
    # 3. 正規化範圍分析
    train_normalized_coords = np.array(train_normalized_coords)
    test_normalized_coords = np.array(test_normalized_coords)
    
    print(f"\n🗺️  正規化後的GPS範圍:")
    print(f"   訓練集 - 緯度: [{train_normalized_coords[:, 0].min():.6f}, {train_normalized_coords[:, 0].max():.6f}]")
    print(f"   訓練集 - 經度: [{train_normalized_coords[:, 1].min():.6f}, {train_normalized_coords[:, 1].max():.6f}]")
    print(f"   測試集 - 緯度: [{test_normalized_coords[:, 0].min():.6f}, {test_normalized_coords[:, 0].max():.6f}]")
    print(f"   測試集 - 經度: [{test_normalized_coords[:, 1].min():.6f}, {test_normalized_coords[:, 1].max():.6f}]")
    
    # 4. 重疊分析
    overlapping_keys = train_keys.intersection(test_keys)
    train_only_keys = train_keys - test_keys
    test_only_keys = test_keys - train_keys
    
    print(f"\n🔄 位置重疊分析:")
    print(f"   重疊量化位置: {len(overlapping_keys)}")
    print(f"   僅訓練集位置: {len(train_only_keys)}")
    print(f"   僅測試集位置: {len(test_only_keys)}")
    print(f"   位置重疊率: {len(overlapping_keys) / len(train_keys.union(test_keys)) * 100:.2f}%")
    
    # 5. 如果提供了模型記憶庫，檢查實際記憶庫內容
    if model_memory_bank is not None:
        memory_stats = model_memory_bank.get_memory_stats()
        memory_keys = set(model_memory_bank.memory_bank.keys())
        
        print(f"\n🧠 實際記憶庫統計:")
        print(f"   記憶庫位置數: {memory_stats['total_locations']}")
        print(f"   記憶庫總記憶數: {memory_stats['total_memories']}")
        
        # 檢查記憶庫鍵與訓練集的匹配
        memory_train_overlap = memory_keys.intersection(train_keys)
        memory_test_overlap = memory_keys.intersection(test_keys)
        
        print(f"   記憶庫與訓練集重疊: {len(memory_train_overlap)}/{len(memory_keys)} ({len(memory_train_overlap)/max(len(memory_keys), 1)*100:.1f}%)")
        print(f"   記憶庫與測試集重疊: {len(memory_test_overlap)}/{len(memory_keys)} ({len(memory_test_overlap)/max(len(memory_keys), 1)*100:.1f}%)")
        
        # 分析測試集能從記憶庫獲得多少有效記憶
        effective_test_matches = 0
        for key in test_keys:
            if key in memory_keys:
                effective_test_matches += test_key_counts[key]
        
        print(f"   測試記錄的有效記憶覆蓋: {effective_test_matches}/{len(test_gps)} ({effective_test_matches/len(test_gps)*100:.1f}%)")
    
    return {
        'match_rate': test_matches / len(test_gps),
        'overlap_rate': len(overlapping_keys) / len(train_keys.union(test_keys)),
        'train_keys': train_keys,
        'test_keys': test_keys,
        'overlapping_keys': overlapping_keys,
        'match_details': pd.DataFrame(match_details),
        'train_key_counts': train_key_counts,
        'test_key_counts': test_key_counts
    }


def test_single_gps_quantization_with_normalization(model, test_gps_tensor, gps_normalizer, device):
    """🆕 測試單個GPS的完整處理流程：正規化 + 量化 + 記憶檢索"""
    
    print(f"\n🔬 單GPS完整處理流程測試:")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        for i, original_gps in enumerate(test_gps_tensor[:5]):  # 測試前5個
            
            print(f"GPS {i+1}: 原始座標 [{original_gps[0]:.6f}, {original_gps[1]:.6f}]")
            
            # 1. 手動正規化
            data = {'gps': original_gps.clone()}
            normalized_data = gps_normalizer(data)
            normalized_gps = normalized_data['gps']
            
            print(f"   步驟1 - 正規化: [{normalized_gps[0]:.6f}, {normalized_gps[1]:.6f}]")
            
            # 2. 手動量化
            spatial_radius = model.memory_bank.spatial_radius
            expected_key = model.memory_bank.gps_to_key(normalized_gps)
            
            print(f"   步驟2 - 量化鍵: {expected_key}")
            
            # 3. 檢查記憶庫
            has_memory = expected_key in model.memory_bank.memory_bank
            memory_count = len(model.memory_bank.memory_bank.get(expected_key, {}).get('features', []))
            
            print(f"   步驟3 - 記憶庫檢查:")
            print(f"           有此鍵: {has_memory}")
            print(f"           記憶數量: {memory_count}")
            
            # 4. 完整的模型推理測試
            gps_batch = normalized_gps.unsqueeze(0).to(device)
            memory_features = model.memory_bank.retrieve_memory(gps_batch)
            memory_norm = torch.norm(memory_features).item()
            
            print(f"   步驟4 - 記憶檢索:")
            print(f"           檢索特徵範數: {memory_norm:.6f}")
            print(f"           有效記憶: {'✅' if memory_norm > 1e-6 else '❌'}")
            
            # 5. 模型完整前向傳播測試
            dummy_image = torch.randn(1, 3, 224, 224).to(device)  # 假的圖像
            try:
                outputs = model(dummy_image, gps_batch, return_embeddings=False, update_memory=False)
                memory_weight = outputs.get('memory_weight', 0)
                print(f"   步驟5 - 完整推理:")
                print(f"           記憶權重: {memory_weight:.4f}")
                print(f"           推理成功: ✅")
            except Exception as e:
                print(f"   步驟5 - 完整推理:")
                print(f"           推理失敗: ❌ {str(e)}")
            
            print()


def load_model_from_checkpoint(checkpoint_path: str, num_categories: int, device: str):
    """🆕 增強版模型載入，支持從檢查點推斷模型參數"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️  Standard loading failed: {e}")
        print("🔄 Trying alternative loading method...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            print("✅ Loaded weights only (without training args)")
            return load_model_with_default_params(checkpoint, num_categories, device)
        except Exception as e2:
            print(f"❌ Alternative loading also failed: {e2}")
            raise e2
    
    # 🆕 嘗試從檢查點中提取模型參數
    model_args = extract_model_args_from_checkpoint(checkpoint, checkpoint_path)
    
    if model_args:
        print(f"✅ Extracted model configuration:")
        print(f"   Model: {model_args['model_size']}, Feature dim: {model_args['feature_dim']}")
        print(f"   Memory: {model_args['memory_size']}, Spatial radius: {model_args['spatial_radius']}")
        
        model = create_memory_enhanced_geo_segformer(
            num_classes=num_categories,
            model_size=model_args['model_size'],
            feature_dim=model_args['feature_dim'],
            fusion_method=model_args['fusion_method'],
            memory_size=model_args['memory_size'],
            spatial_radius=model_args['spatial_radius'],
            memory_save_path=None
        ).to(device)
        
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        best_score = checkpoint.get('best_score', 'unknown')
        print(f"✅ Model loaded from epoch {epoch}, best score: {best_score}")
        
        return model, model_args
    
    else:
        print("⚠️  Could not extract model args, using defaults")
        model = load_model_with_default_params(checkpoint, num_categories, device)
        return model, None


def extract_model_args_from_checkpoint(checkpoint, checkpoint_path: str):
    """🆕 從檢查點中提取模型參數"""
    
    # 方法1: 檢查是否有顯式保存的args
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, 'model_size'):
            return {
                'model_size': getattr(args, 'model_size', 'b0'),
                'feature_dim': getattr(args, 'feature_dim', 512),
                'fusion_method': getattr(args, 'fusion_method', 'attention'),
                'memory_size': getattr(args, 'memory_size', 20),
                'spatial_radius': getattr(args, 'spatial_radius', 0.00005),
            }
    
    # 方法2: 檢查model_config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        return {
            'model_size': config.get('model_size', 'b0'),
            'feature_dim': config.get('feature_dim', 512),
            'fusion_method': config.get('fusion_method', 'attention'),
            'memory_size': config.get('memory_size', 20),
            'spatial_radius': config.get('spatial_radius', 0.00005),
        }
    
    # 方法3: 從模型權重推斷參數
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        extracted_args = infer_model_args_from_weights(model_state)
        if extracted_args:
            return extracted_args
    
    # 方法4: 從檔案路徑推斷參數
    path_args = infer_model_args_from_path(checkpoint_path)
    if path_args:
        return path_args
    
    return None


def infer_model_args_from_weights(model_state_dict):
    """🆕 從模型權重推斷參數"""
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
        model_size = "b0"  # 默認
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
                'memory_size': 20,
                'spatial_radius': 0.00005,
            }
    
    except Exception as e:
        print(f"⚠️  Failed to infer from weights: {e}")
    
    return None


def infer_model_args_from_path(checkpoint_path: str):
    """🆕 從檔案路徑推斷參數"""
    try:
        path_str = str(checkpoint_path).lower()
        
        # 推斷model_size
        if 'b1' in path_str:
            model_size = 'b1'
        elif 'b2' in path_str:
            model_size = 'b2'
        else:
            model_size = 'b0'
        
        # 推斷feature_dim
        feature_dim = 512  # 默認
        if 'dim256' in path_str or 'feature256' in path_str:
            feature_dim = 256
        elif 'dim1024' in path_str or 'feature1024' in path_str:
            feature_dim = 1024
        
        return {
            'model_size': model_size,
            'feature_dim': feature_dim,
            'fusion_method': 'attention',
            'memory_size': 20,
            'spatial_radius': 0.00005,
        }
    
    except Exception as e:
        print(f"⚠️  Failed to infer from path: {e}")
    
    return None


def load_model_with_default_params(checkpoint, num_categories: int, device: str):
    """使用默認參數載入模型"""
    print("🔧 Using default parameters for model creation:")
    model_size = "b0"
    feature_dim = 512
    fusion_method = "attention"
    memory_size = 20
    spatial_radius = 0.00005
    
    print(f"   Model: {model_size}, Feature dim: {feature_dim}")
    print(f"   Memory: {memory_size}, Spatial radius: {spatial_radius}")
    
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
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    best_score = checkpoint.get('best_score', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    print(f"✅ Model loaded from epoch {epoch}, best score: {best_score}")
    
    return model


def find_memory_bank_file(checkpoint_path: str, logdir: str = None):
    """🆕 智能尋找記憶庫文件"""
    possible_paths = []
    
    # 1. 從檢查點路徑推斷記憶庫路徑
    checkpoint_dir = os.path.dirname(checkpoint_path)
    possible_paths.extend([
        os.path.join(checkpoint_dir, "memory_stats.pth"),
        os.path.join(checkpoint_dir, "memory_stats.json"),
        os.path.join(checkpoint_dir, "multilayer_memory_stats.pth"),
        os.path.join(checkpoint_dir, "multilayer_memory_stats.json"),
    ])
    
    # 2. 如果提供了logdir
    if logdir:
        possible_paths.extend([
            os.path.join(logdir, "memory_stats.pth"),
            os.path.join(logdir, "memory_stats.json"),
            os.path.join(logdir, "multilayer_memory_stats.pth"),
            os.path.join(logdir, "multilayer_memory_stats.json"),
        ])
    
    # 3. 檢查是否存在
    for path in possible_paths:
        if os.path.exists(path):
            print(f"🔍 Found memory bank file: {path}")
            return path
    
    print("❌ No memory bank file found in expected locations:")
    for path in possible_paths:
        print(f"   - {path}")
    
    return None


def run_inference_with_memory(model, dataloader, device, args, memory_enabled=True):
    """🆕 執行帶記憶或不帶記憶的推理"""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(len(Category.load(args.category_csv, show=False)), nan_to_num=0)
    
    total_loss = 0
    total_memory_weight = 0
    memory_hit_count = 0
    
    # 🆕 GPS量化統計
    quantization_hits = 0
    effective_memory_count = 0
    total_samples_processed = 0
    
    # 🆕 記憶庫狀態管理
    original_memory_enabled = memory_enabled
    if not memory_enabled and hasattr(model, 'memory_bank'):
        print("🔄 Temporarily disabling memory bank for comparison...")
        original_memory_bank = model.memory_bank.memory_bank
        from collections import defaultdict
        model.memory_bank.memory_bank = defaultdict(lambda: {'features': [], 'count': 0, 'last_updated': 0})
    
    progress_desc = "Testing (with memory)" if memory_enabled else "Testing (without memory)"
    
    with Progress() as prog:
        with torch.no_grad():
            task = prog.add_task(progress_desc, total=len(dataloader))
            
            for batch_idx, data in enumerate(dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                # 推理（不更新記憶庫）
                outputs = model(img, gps, return_embeddings=False, update_memory=False)
                pred = outputs['segmentation_logits']
                
                # 計算損失
                loss = criterion(pred, ann)
                total_loss += loss.item()
                
                # 記錄記憶統計
                memory_weight = outputs.get('memory_weight', 0)
                total_memory_weight += memory_weight
                if memory_weight > 0.1:  # 有效記憶閾值
                    memory_hit_count += 1
                
                # 🆕 GPS量化統計（僅在memory_enabled時統計）
                if memory_enabled and hasattr(model, 'memory_bank'):
                    for i in range(gps.shape[0]):
                        gps_coord = gps[i]
                        quantized_key = model.memory_bank.gps_to_key(gps_coord)
                        
                        # 檢查是否有記憶
                        has_memory = quantized_key in model.memory_bank.memory_bank
                        if has_memory:
                            quantization_hits += 1
                            memory_count = len(model.memory_bank.memory_bank[quantized_key]['features'])
                            if memory_count > 0:
                                effective_memory_count += 1
                        
                        total_samples_processed += 1
                
                # 計算評估指標
                metrics.compute_and_accum(pred.argmax(1), ann)
                
                # 保存預測結果
                if args.save_dir:
                    save_suffix = "_with_mem" if memory_enabled else "_no_mem"
                    for fn, p in zip(data["img_path"], pred):
                        filename = Path(fn).stem + save_suffix + ".png"
                        img_saver = ImgSaver(args.save_dir, IdMapVisualizer(Category.load(args.category_csv, show=False)))
                        img_saver.save_pred(p[None, :], filename)
                
                # 顯示記憶統計
                if args.show_memory_stats and batch_idx % 100 == 0 and memory_enabled:
                    if hasattr(model, 'get_memory_stats'):
                        memory_stats = model.get_memory_stats()
                        hit_rate = quantization_hits / max(total_samples_processed, 1)
                        effective_rate = effective_memory_count / max(total_samples_processed, 1)
                        print(f"🧠 Batch {batch_idx}: Locations: {memory_stats['total_locations']}, "
                              f"Memory hit rate: {memory_stats['hit_rate']:.3f}, "
                              f"GPS quantization hit: {hit_rate:.3f}, "
                              f"Effective memory: {effective_rate:.3f}, "
                              f"Current weight: {memory_weight:.3f}")
                
                prog.update(task, advance=1)
            
            # 獲取最終結果
            result = metrics.get_and_reset()
            avg_loss = total_loss / len(dataloader)
            avg_memory_weight = total_memory_weight / len(dataloader)
            memory_usage_rate = memory_hit_count / len(dataloader)
            
            # 🆕 GPS量化統計
            gps_quantization_hit_rate = quantization_hits / max(total_samples_processed, 1) if memory_enabled else 0
            effective_memory_rate = effective_memory_count / max(total_samples_processed, 1) if memory_enabled else 0
            
            prog.remove_task(task)
    
    # 🆕 恢復記憶庫（如果之前禁用了）
    if not original_memory_enabled and hasattr(model, 'memory_bank'):
        print("🔄 Restoring original memory bank...")
        model.memory_bank.memory_bank = original_memory_bank
    
    return {
        'result': result,
        'avg_loss': avg_loss,
        'avg_memory_weight': avg_memory_weight,
        'memory_usage_rate': memory_usage_rate,
        'memory_enabled': memory_enabled,
        'gps_quantization_hit_rate': gps_quantization_hit_rate,
        'effective_memory_rate': effective_memory_rate,
        'total_samples': total_samples_processed
    }


def print_comparison_results(with_mem_results, without_mem_results):
    """🆕 打印記憶對比結果"""
    print("\n" + "="*80)
    print("🆚 Memory Bank Impact Analysis")
    print("="*80)
    
    # 基本指標對比
    with_miou = with_mem_results['result']['IoU'].mean()
    without_miou = without_mem_results['result']['IoU'].mean()
    miou_improvement = with_miou - without_miou
    
    with_acc = with_mem_results['result']['Acc'].mean()
    without_acc = without_mem_results['result']['Acc'].mean()
    acc_improvement = with_acc - without_acc
    
    with_loss = with_mem_results['avg_loss']
    without_loss = without_mem_results['avg_loss']
    loss_improvement = without_loss - with_loss  # 損失越小越好
    
    print(f"📊 Overall Performance Comparison:")
    print(f"   Mean IoU:      {without_miou:.5f} → {with_miou:.5f} ({miou_improvement:+.5f})")
    print(f"   Mean Accuracy: {without_acc:.5f} → {with_acc:.5f} ({acc_improvement:+.5f})")
    print(f"   Average Loss:  {without_loss:.5f} → {with_loss:.5f} ({loss_improvement:+.5f})")
    
    # 記憶使用統計
    print(f"\n🧠 Memory Usage Statistics:")
    print(f"   Average Memory Weight: {with_mem_results['avg_memory_weight']:.4f}")
    print(f"   Memory Usage Rate:     {with_mem_results['memory_usage_rate']:.2%}")
    print(f"   GPS Quantization Hit Rate: {with_mem_results['gps_quantization_hit_rate']:.2%}")
    print(f"   Effective Memory Rate:     {with_mem_results['effective_memory_rate']:.2%}")
    
    # 改進分析
    print(f"\n📈 Improvement Analysis:")
    if miou_improvement > 0.001:
        print(f"   ✅ Memory bank provides {miou_improvement:.4f} mIoU improvement")
    elif miou_improvement > -0.001:
        print(f"   ➡️  Memory bank has minimal impact ({miou_improvement:+.4f} mIoU)")
    else:
        print(f"   ❌ Memory bank may be hurting performance ({miou_improvement:+.4f} mIoU)")
    
    # GPS量化診斷
    print(f"\n🎯 GPS Quantization Diagnosis:")
    hit_rate = with_mem_results['gps_quantization_hit_rate']
    if hit_rate < 0.1:
        print(f"   ❌ Very low GPS quantization hit rate ({hit_rate:.1%})")
        print(f"      → GPS normalization may be inconsistent with training")
    elif hit_rate < 0.3:
        print(f"   ⚠️  Low GPS quantization hit rate ({hit_rate:.1%})")
        print(f"      → Some GPS coordinates don't match training quantization")
    else:
        print(f"   ✅ Good GPS quantization hit rate ({hit_rate:.1%})")
        print(f"      → GPS normalization appears consistent with training")


def save_comparison_results_csv(with_mem_results, without_mem_results, categories, save_dir):
    """保存記憶對比結果到CSV"""
    
    print("🆚 Exporting memory comparison results to CSV...")
    
    comparison_data = []
    
    # 各類別對比
    for i, cat in enumerate(categories):
        with_iou = with_mem_results['result']['IoU'][i]
        without_iou = without_mem_results['result']['IoU'][i]
        with_acc = with_mem_results['result']['Acc'][i]
        without_acc = without_mem_results['result']['Acc'][i]
        
        row = {
            'Category': cat.name,
            'Category_ID': cat.id,
            'IoU_Without_Memory': float(without_iou),
            'IoU_With_Memory': float(with_iou),
            'IoU_Improvement': float(with_iou - without_iou),
            'IoU_Improvement_Percent': float((with_iou - without_iou) / max(without_iou, 1e-8) * 100),
            'Acc_Without_Memory': float(without_acc),
            'Acc_With_Memory': float(with_acc),
            'Acc_Improvement': float(with_acc - without_acc),
            'Acc_Improvement_Percent': float((with_acc - without_acc) / max(without_acc, 1e-8) * 100)
        }
        comparison_data.append(row)
    
    # 添加平均對比
    with_avg_iou = with_mem_results['result']['IoU'].mean()
    without_avg_iou = without_mem_results['result']['IoU'].mean()
    with_avg_acc = with_mem_results['result']['Acc'].mean()
    without_avg_acc = without_mem_results['result']['Acc'].mean()
    
    avg_row = {
        'Category': 'Average',
        'Category_ID': 'AVG',
        'IoU_Without_Memory': float(without_avg_iou),
        'IoU_With_Memory': float(with_avg_iou),
        'IoU_Improvement': float(with_avg_iou - without_avg_iou),
        'IoU_Improvement_Percent': float((with_avg_iou - without_avg_iou) / without_avg_iou * 100),
        'Acc_Without_Memory': float(without_avg_acc),
        'Acc_With_Memory': float(with_avg_acc),
        'Acc_Improvement': float(with_avg_acc - without_avg_acc),
        'Acc_Improvement_Percent': float((with_avg_acc - without_avg_acc) / without_avg_acc * 100)
    }
    comparison_data.append(avg_row)
    
    # 保存對比結果
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = os.path.join(save_dir, "memory_comparison_results.csv")
    comparison_df.to_csv(comparison_csv, index=False, float_format='%.5f')
    print(f"   ✅ Memory comparison: {comparison_csv}")
    
    return comparison_csv


def parse_args() -> Namespace:
    parser = ArgumentParser(description="🧠 Enhanced GeoSegformer Model Testing with Memory Bank")
    
    # 基本參數
    parser.add_argument("img_dir", type=str, help="Test images directory")
    parser.add_argument("ann_dir", type=str, help="Test annotations directory")
    parser.add_argument("category_csv", type=str, help="Category CSV file")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint file")
    parser.add_argument("test_gps_csv", type=str, help="Test GPS CSV file")
    parser.add_argument("train_gps_csv", type=str, help="Training GPS CSV file (for normalization)")
    parser.add_argument("val_gps_csv", type=str, help="Validation GPS CSV file (for normalization)")
    
    # 🆕 記憶庫相關參數
    parser.add_argument("--memory-bank-path", type=str, default=None,
                       help="Path to saved memory bank (auto-search if not provided)")
    parser.add_argument("--force-no-memory", action="store_true",
                       help="Force testing without memory bank (for comparison)")
    parser.add_argument("--logdir", type=str, default=None,
                       help="Training log directory (for auto-finding memory bank)")
    
    # 測試參數
    parser.add_argument("--save-dir", type=str, default=None, 
                       help="Directory to save prediction results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum number of test samples")
    
    # GPS正規化
    parser.add_argument("--gps-norm-method", type=str, default="minmax", 
                       choices=["minmax", "zscore"], help="GPS normalization method")
    
    # 調試選項
    parser.add_argument("--show-memory-stats", action="store_true",
                       help="Show memory bank statistics during testing")
    parser.add_argument("--compare-with-without-memory", action="store_true",
                       help="Run comparison between with/without memory")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress and statistics")
    parser.add_argument("--verify-gps-quantization", action="store_true", default=True,
                       help="Verify GPS quantization matching before testing")
    
    return parser.parse_args()


def main(args: Namespace):
    print("🧠 Enhanced GeoSegformer Model Testing with Memory Bank")
    print("=" * 70)
    
    # 基本設置
    image_size = 720, 1280
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入類別
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    print(f"📋 Categories: {num_categories} classes")
    
    # 🆕 載入模型（支持參數推斷）
    model_info = load_model_from_checkpoint(args.checkpoint, num_categories, device)
    if isinstance(model_info, tuple):
        model, model_args = model_info
    else:
        model = model_info
        model_args = None
    
    # 🆕 尋找和載入記憶庫
    memory_loaded = False
    if not args.force_no_memory:
        memory_bank_path = args.memory_bank_path
        
        if not memory_bank_path:
            # 自動尋找記憶庫文件
            memory_bank_path = find_memory_bank_file(args.checkpoint, args.logdir)
        
        if memory_bank_path:
            print(f"🔄 Loading memory bank from: {memory_bank_path}")
            memory_loaded = model.load_memory_bank(memory_bank_path)
            
            if memory_loaded:
                initial_stats = model.get_memory_stats()
                print(f"🧠 Memory bank loaded successfully!")
                print(f"   📍 {initial_stats['total_locations']} GPS locations")
                print(f"   🧠 {initial_stats['total_memories']} stored memories")
        else:
            print("⚠️  No memory bank found, testing without historical memory")
    
    if args.force_no_memory:
        print("🚫 Force testing without memory bank")
    
    # 🆕 設置GPS正規化（手動方式，確保與訓練時一致）
    print("\n🗺️  Setting up GPS normalization (manual mode)...")
    gps_normalizer = manual_gps_normalization_exact_training_params(
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
    
    # 🆕 GPS量化匹配檢查
    if args.verify_gps_quantization and memory_loaded:
        print("\n🔍 GPS量化匹配檢查:")
        print("=" * 50)
        
        # 分析GPS量化匹配
        quantization_results = analyze_gps_quantization_matching(
            args.train_gps_csv,
            args.test_gps_csv,
            model.memory_bank.spatial_radius,
            gps_normalizer,
            model.memory_bank
        )
        
        # 測試單個GPS的量化
        if len(test_dataset) > 0:
            sample_gps = torch.stack([test_dataset[i]['gps'] for i in range(min(5, len(test_dataset)))])
            test_single_gps_quantization_with_normalization(model, sample_gps, gps_normalizer, device)
        
        # 根據匹配率給出建議
        match_rate = quantization_results['match_rate']
        if match_rate < 0.1:
            print(f"❌ GPS匹配率過低 ({match_rate*100:.1f}%)，記憶庫可能無法有效工作！")
            print(f"建議檢查:")
            print(f"  1. spatial_radius是否與訓練時一致")
            print(f"  2. GPS正規化參數是否正確")
            print(f"  3. 訓練和測試數據的GPS分佈差異")
        elif match_rate > 0.5:
            print(f"✅ GPS匹配率良好 ({match_rate*100:.1f}%)，記憶庫應該能有效工作")
        else:
            print(f"💡 GPS匹配率中等 ({match_rate*100:.1f}%)，記憶庫會部分有效")
    
    print(f"\n🚀 Starting evaluation on {len(dataloader)} samples...")
    
    # 🆕 執行測試
    if args.compare_with_without_memory and memory_loaded:
        print("\n🆚 Running comparison: with vs without memory bank")
        
        # 測試帶記憶庫
        print("\n1️⃣ Testing WITH memory bank...")
        with_mem_results = run_inference_with_memory(model, dataloader, device, args, memory_enabled=True)
        
        # 測試不帶記憶庫
        print("\n2️⃣ Testing WITHOUT memory bank...")
        without_mem_results = run_inference_with_memory(model, dataloader, device, args, memory_enabled=False)
        
        # 打印對比結果
        print_comparison_results(with_mem_results, without_mem_results)
        
        # 使用帶記憶的結果作為主要結果
        main_results = with_mem_results
        
        # 🆕 保存對比結果到CSV
        if args.save_dir:
            save_comparison_results_csv(with_mem_results, without_mem_results, categories, args.save_dir)
        
    else:
        # 標準測試流程
        memory_enabled = memory_loaded and not args.force_no_memory
        main_results = run_inference_with_memory(model, dataloader, device, args, memory_enabled=memory_enabled)
    
    # 顯示主要結果
    result = main_results['result']
    avg_loss = main_results['avg_loss']
    
    print("\n📊 Final Test Results:")
    print("=" * 80)
    
    # 創建結果表格
    table = Table(title="🎯 GeoSegformer Test Performance")
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
    
    if memory_loaded:
        print(f"   Average Memory Weight: {main_results['avg_memory_weight']:.4f}")
        print(f"   Memory Usage Rate: {main_results['memory_usage_rate']:.2%}")
        print(f"   GPS Quantization Hit Rate: {main_results['gps_quantization_hit_rate']:.2%}")
        print(f"   Effective Memory Rate: {main_results['effective_memory_rate']:.2%}")
    
    # 最終記憶庫統計
    if args.show_memory_stats and memory_loaded:
        final_memory_stats = model.get_memory_stats()
        print(f"\n🧠 Final Memory Bank Statistics:")
        print(f"   Total Locations: {final_memory_stats['total_locations']}")
        print(f"   Total Memories: {final_memory_stats['total_memories']}")
        print(f"   Hit Rate: {final_memory_stats['hit_rate']:.4f}")
        print(f"   Avg Memories per Location: {final_memory_stats['avg_memories_per_location']:.2f}")
    
    # 保存結果
    if args.save_dir:
        results_file = os.path.join(args.save_dir, "test_results.txt")
        with open(results_file, 'w') as f:
            f.write("Enhanced GeoSegformer Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Memory Bank: {memory_loaded}\n")
            f.write(f"Test samples: {len(dataloader)}\n")
            f.write(f"Average Loss: {avg_loss:.5f}\n")
            f.write(f"Mean IoU: {result['IoU'].mean():.5f}\n")
            f.write(f"Mean Accuracy: {result['Acc'].mean():.5f}\n")
            
            if memory_loaded:
                f.write(f"Average Memory Weight: {main_results['avg_memory_weight']:.4f}\n")
                f.write(f"Memory Usage Rate: {main_results['memory_usage_rate']:.2%}\n")
                f.write(f"GPS Quantization Hit Rate: {main_results['gps_quantization_hit_rate']:.2%}\n")
                f.write(f"Effective Memory Rate: {main_results['effective_memory_rate']:.2%}\n")
            
            f.write(f"\nPer-category results:\n")
            for i, cat in enumerate(categories):
                f.write(f"{cat.name}: IoU={result['IoU'][i]:.5f}, Acc={result['Acc'][i]:.5f}\n")
            
            if memory_loaded and args.show_memory_stats:
                final_memory_stats = model.get_memory_stats()
                f.write(f"\nMemory Bank Statistics:\n")
                f.write(f"Total Locations: {final_memory_stats['total_locations']}\n")
                f.write(f"Total Memories: {final_memory_stats['total_memories']}\n")
                f.write(f"Hit Rate: {final_memory_stats['hit_rate']:.4f}\n")
        
        print(f"💾 Results saved to: {results_file}")
    
    print(f"\n✅ Testing completed successfully!")
    
    if memory_loaded:
        print(f"🧠 Memory-enhanced inference completed")
        print(f"   Memory usage rate: {main_results['memory_usage_rate']:.1%}")
        print(f"   GPS quantization hit rate: {main_results['gps_quantization_hit_rate']:.1%}")
        if main_results['gps_quantization_hit_rate'] < 0.1:
            print(f"   ⚠️  Low GPS quantization hit rate suggests GPS normalization issues")
        elif main_results['gps_quantization_hit_rate'] > 0.5:
            print(f"   ✅ Good GPS quantization hit rate indicates proper memory bank utilization")
    else:
        print(f"🔄 Standard inference completed without memory bank")


if __name__ == "__main__":
    args = parse_args()
    main(args)