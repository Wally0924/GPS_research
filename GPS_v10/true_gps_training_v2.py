#!/usr/bin/env python3
"""
真正的 GPS 訓練整合 - 完整改進版
在訓練過程中使用 GPS 信息，通過 hook 機制注入 GPS 特徵
包含所有改進功能：GPS正規化、相似位置學習、YOLOv10適配、數據增強等
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.nn.modules import Conv, C2f, SPPF
import yaml


class ImprovedGPSEncoder(nn.Module):
    """改進的 GPS 坐標編碼器 - 包含正規化和位置編碼"""
    
    def __init__(self, input_dim=2, embed_dim=64, sigma=0.0001, gps_stats=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma = sigma
        
        # 动态设置 GPS 数据范围
        if gps_stats:
            self.register_buffer('lat_min', torch.tensor(gps_stats['lat_min']))
            self.register_buffer('lat_max', torch.tensor(gps_stats['lat_max']))
            self.register_buffer('lon_min', torch.tensor(gps_stats['lon_min']))
            self.register_buffer('lon_max', torch.tensor(gps_stats['lon_max']))
        else:
            # 默认值
            self.register_buffer('lat_min', torch.tensor(-90.0))
            self.register_buffer('lat_max', torch.tensor(90.0))
            self.register_buffer('lon_min', torch.tensor(-180.0))
            self.register_buffer('lon_max', torch.tensor(180.0))
        
        self.stats_initialized = gps_stats is not None
        
        # Random Fourier Features for GPS encoding
        self.rff_dim = embed_dim
        self.omega = nn.Parameter(torch.randn((self.rff_dim // 2, 2)) / sigma, requires_grad=False)
        self.b = nn.Parameter(2 * math.pi * torch.rand(self.rff_dim // 2), requires_grad=False)
        
        # 🔧 改進: 添加正規化和dropout的MLP
        self.encoder = nn.Sequential(
            nn.Linear(self.rff_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 🔧 新增: 相似位置學習 - 位置編碼表
        self.register_buffer('position_table', self._get_sinusoid_encoding_table(1000, embed_dim))
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """生成正弦位置編碼表 - 用於相似位置學習"""
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)
    
    def forward(self, gps_coords):
        """
        gps_coords: (batch_size, 2) [lat, lon]
        return: (batch_size, embed_dim)
        """
        # 🔧 改進: GPS正規化 - Min-Max標準化
        lat_norm = 2 * (gps_coords[:, 0] - self.lat_min) / (self.lat_max - self.lat_min) - 1
        lon_norm = 2 * (gps_coords[:, 1] - self.lon_min) / (self.lon_max - self.lon_min) - 1
        normalized_gps = torch.stack([lat_norm, lon_norm], dim=1)
        
        # 🔧 新增: 訓練時數據增強
        if self.training:
            noise = torch.randn_like(normalized_gps) * 0.01
            normalized_gps = normalized_gps + noise
        
        # Random Fourier Features encoding
        proj = normalized_gps @ self.omega.T + self.b
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        # 通過 MLP 編碼
        encoded = self.encoder(rff)
        
        # 🔧 新增: 相似位置學習 - 添加位置編碼
        batch_size = gps_coords.size(0)
        pos_indices = torch.clamp(
            ((normalized_gps[:, 0] + normalized_gps[:, 1] + 2) * 250).long(), 
            0, 999
        )
        pos_encoding = self.position_table[pos_indices]
        
        return encoded + pos_encoding
    def update_gps_stats(self, gps_stats):
        """动态更新 GPS 统计信息"""
        self.lat_min.data = torch.tensor(gps_stats['lat_min'])
        self.lat_max.data = torch.tensor(gps_stats['lat_max'])
        self.lon_min.data = torch.tensor(gps_stats['lon_min'])
        self.lon_max.data = torch.tensor(gps_stats['lon_max'])
        self.stats_initialized = True
        LOGGER.info(f"Updated GPS encoder stats: lat=[{gps_stats['lat_min']:.6f}, {gps_stats['lat_max']:.6f}], "
                f"lon=[{gps_stats['lon_min']:.6f}, {gps_stats['lon_max']:.6f}]")


class AdaptiveGPSFeatureFusion(nn.Module):
    """自適應 GPS 特徵融合模塊 - 支持多種融合策略"""
    
    def __init__(self, feature_channels, gps_embed_dim=64, fusion_type='attention', gps_stats=None):
        super().__init__()
        self.feature_channels = feature_channels
        self.gps_embed_dim = gps_embed_dim
        self.fusion_type = fusion_type
        
        # GPS 编码器
        self.gps_encoder = ImprovedGPSEncoder(embed_dim=gps_embed_dim, gps_stats=gps_stats)
        
        # 🔧 新增: 多種融合策略
        if fusion_type == 'attention':
            # 注意力融合（推薦）
            self.attention = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels // 4, feature_channels, 1),
                nn.Sigmoid()
            )
            self.gps_projection = nn.Conv2d(gps_embed_dim, feature_channels, 1)
            
        elif fusion_type == 'adaptive':
            # 自適應融合
            self.adaptive_weight = nn.Sequential(
                nn.Conv2d(feature_channels + gps_embed_dim, feature_channels, 1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, feature_channels, 1),
                nn.Sigmoid()
            )
        
        elif fusion_type == 'multiply':
            # 乘法融合
            self.gps_gate = nn.Sequential(
                nn.Conv2d(gps_embed_dim, feature_channels, 1),
                nn.Sigmoid()
            )
            
        else:  # concat (原始方法)
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_channels + gps_embed_dim, feature_channels, 1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, features, gps_coords):
        """
        features: (B, C, H, W)
        gps_coords: (B, 2)
        """
        B, C, H, W = features.shape
        
        # 編碼 GPS 坐標
        gps_features = self.gps_encoder(gps_coords)  # (B, gps_embed_dim)
        
        # 將 GPS 特徵擴展到空間維度
        gps_spatial = gps_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # 🔧 改進: 根據融合類型處理
        if self.fusion_type == 'attention':
            gps_proj = self.gps_projection(gps_spatial)
            attention_weights = self.attention(features)
            enhanced_features = features + gps_proj * attention_weights
            
        elif self.fusion_type == 'adaptive':
            concatenated = torch.cat([features, gps_spatial], dim=1)
            adaptive_weights = self.adaptive_weight(concatenated)
            enhanced_features = features * adaptive_weights
            
        elif self.fusion_type == 'multiply':
            gps_gate = self.gps_gate(gps_spatial)
            enhanced_features = features * gps_gate
            
        else:  # concat
            concatenated = torch.cat([features, gps_spatial], dim=1)
            enhanced_features = self.fusion_conv(concatenated)
        
        return enhanced_features
class GPSDataManager:
    """GPS 數據管理器 - 改進版"""
    
    def __init__(self, gps_csv_path):
        self.gps_mapping = self.load_gps_data(gps_csv_path)
        self.gps_stats = None
        self._compute_gps_statistics()
        
    def load_gps_data(self, gps_csv_path):
        """載入 GPS 數據"""
        try:
            gps_df = pd.read_csv(gps_csv_path)
            gps_mapping = {}
            for _, row in gps_df.iterrows():
                gps_mapping[row['filename']] = torch.tensor([row['lat'], row['long']], dtype=torch.float32)
            LOGGER.info(f"Loaded GPS data: {len(gps_mapping)} records")
            return gps_mapping
        except Exception as e:
            LOGGER.error(f"Error loading GPS data: {e}")
            return {}
    
    def _compute_gps_statistics(self):
        """🔧 新增: 計算GPS統計信息"""
        if not self.gps_mapping:
            return
        
        gps_values = list(self.gps_mapping.values())
        gps_tensor = torch.stack(gps_values)
        
        self.gps_stats = {
            'lat_min': gps_tensor[:, 0].min().item(),
            'lat_max': gps_tensor[:, 0].max().item(),
            'lat_mean': gps_tensor[:, 0].mean().item(),
            'lon_min': gps_tensor[:, 1].min().item(),
            'lon_max': gps_tensor[:, 1].max().item(), 
            'lon_mean': gps_tensor[:, 1].mean().item(),
        }
        
        LOGGER.info(f"GPS statistics: lat=({self.gps_stats['lat_min']:.6f}, {self.gps_stats['lat_max']:.6f}), "
                   f"lon=({self.gps_stats['lon_min']:.6f}, {self.gps_stats['lon_max']:.6f})")
    
    def get_gps_stats(self):
        """获取 GPS 统计信息"""
        return self.gps_stats

    def has_valid_stats(self):
        """检查是否有有效的统计信息"""
        return self.gps_stats is not None and len(self.gps_mapping) > 0

    def get_gps_batch(self, image_paths):
        """獲取批次的 GPS 數據 - 改進版"""
        batch_gps = []
        for path in image_paths:
            filename = os.path.basename(path)
            gps_coords = self.gps_mapping.get(filename)
            
            if gps_coords is None:
                # 🔧 改進: 使用統計信息作為回退
                if self.gps_stats:
                    gps_coords = torch.tensor([
                        self.gps_stats['lat_mean'], 
                        self.gps_stats['lon_mean']
                    ], dtype=torch.float32)
                else:
                    gps_coords = torch.tensor([0.0, 0.0], dtype=torch.float32)
                LOGGER.debug(f"Using fallback GPS for {filename}")
            
            batch_gps.append(gps_coords)
        
        return torch.stack(batch_gps) if batch_gps else torch.zeros((len(image_paths), 2), dtype=torch.float32)


def get_yolov10_fusion_layers(model_name):
    """🔧 新增: YOLOv10適配 - 獲取適合的融合層"""
    yolov10_layers = {
        'yolov10n': [9, 12, 15],   # 針對 nano 版本
        'yolov10s': [12, 15, 18],  # 針對 small 版本
        'yolov10m': [15, 18, 21],  # 針對 medium 版本
        'yolov10l': [18, 21, 24],  # 針對 large 版本
        'yolov10x': [21, 24, 27],  # 針對 xlarge 版本
    }
    
    model_lower = model_name.lower()
    for model_key in yolov10_layers:
        if model_key in model_lower:
            return yolov10_layers[model_key]
    
    # 默認返回（通用）
    return [15, 18, 21]


def load_config_from_yaml(yaml_path):
    """🔧 新增: 從 YAML 文件讀取配置"""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        LOGGER.error(f"Failed to load config from {yaml_path}: {e}")
        return {}
class GPSEnhancedTrainer:
    """GPS 增強的訓練器 - 完整改進版"""
    
    def __init__(self, model_path, gps_train_manager: GPSDataManager):
        self.model = YOLO(model_path)
        self.gps_manager = gps_train_manager
        self.val_gps_manager = None  # ✅ 修改: 新增 val 階段專用 GPS 管理器
        
        self.fusion_layers = get_yolov10_fusion_layers(model_path)
        self.fusion_type = 'attention'
        self.gps_embed_dim = 64
        
        self.fusion_modules = {}
        self.hooks = []
        self.current_gps_batch = None
        self.gps_fusion_enabled = False

    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if not self.gps_fusion_enabled:
                    return output
                if self.current_gps_batch is not None:
                    gps_coords = self.current_gps_batch.to(output.device)
                    if layer_idx not in self.fusion_modules:
                        fusion_module = AdaptiveGPSFeatureFusion(
                            output.shape[1],
                            self.gps_embed_dim,
                            self.fusion_type,
                            self.gps_manager.get_gps_stats()
                        ).to(output.device)
                        self.fusion_modules[layer_idx] = fusion_module
                    return self.fusion_modules[layer_idx](output, gps_coords)
                return output
            return hook_fn

        for layer_idx in self.fusion_layers:
            if layer_idx < len(self.model.model.model):
                hook = self.model.model.model[layer_idx].register_forward_hook(create_hook(layer_idx))
                self.hooks.append(hook)

    def set_gps_batch(self, image_paths, is_val=False):
        """✅ 修改: 支援 val 階段使用不同 GPS CSV"""
        gps_manager = self.val_gps_manager if (is_val and self.val_gps_manager) else self.gps_manager
        self.current_gps_batch = gps_manager.get_gps_batch(image_paths)

    def _extract_image_paths(self, trainer):
        if isinstance(trainer.batch, dict):
            return trainer.batch.get('im_file') or trainer.batch.get('path') or trainer.batch.get('paths')
        elif hasattr(trainer.batch, 'im_file'):
            return trainer.batch.im_file
        return None

    def train(self, **kwargs):
        self._register_hooks()
        original_callbacks = self.model.callbacks.copy()

        def on_train_batch_start(trainer):
            image_paths = self._extract_image_paths(trainer)
            if image_paths:
                self.set_gps_batch(image_paths, is_val=False)
                self.gps_fusion_enabled = True

        self.model.add_callback('on_train_batch_start', on_train_batch_start)
        results = self.model.train(**kwargs)
        self.model.callbacks = original_callbacks
        return results

    def validate(self, **kwargs):
        """✅ 修改: 驗證時自動啟用 val_gps_manager"""
        self._register_hooks()
        original_callbacks = self.model.callbacks.copy()

        def on_val_batch_start(trainer):
            image_paths = self._extract_image_paths(trainer)
            if image_paths:
                self.set_gps_batch(image_paths, is_val=True)
                self.gps_fusion_enabled = True

        self.model.add_callback('on_val_batch_start', on_val_batch_start)
        results = self.model.val(**kwargs)
        self.model.callbacks = original_callbacks
        return results

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    parser = argparse.ArgumentParser(description='GPS-Enhanced YOLO Training v1.1 (train/val CSV supported)')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--gps-train-csv', type=str, required=True, help='Train GPS CSV file')  # ✅ 修改
    parser.add_argument('--gps-val-csv', type=str, required=True, help='Validation GPS CSV file')  # ✅ 修改
    parser.add_argument('--model', type=str, default='yolov10n.pt', help='Model file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device')
    args = parser.parse_args()

    # 初始化 train/val GPS 管理器
    gps_train_manager = GPSDataManager(args.gps_train_csv)
    gps_val_manager = GPSDataManager(args.gps_val_csv)

    trainer = GPSEnhancedTrainer(args.model, gps_train_manager)
    trainer.val_gps_manager = gps_val_manager  # ✅ 修改: 綁定 val CSV

    LOGGER.info("====== 開始訓練 ======")
    trainer.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size, device=args.device)

    LOGGER.info("====== 驗證階段 ======")
    trainer.validate(data=args.data, imgsz=args.imgsz, batch=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()

