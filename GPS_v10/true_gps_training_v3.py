#!/usr/bin/env python3
"""
真正的 GPS 訓練整合 - v3.1 修正版
✅ 基於 v3 新增 YOLO v8~v10 版本適配，修復 trainer.batch 錯誤
✅ 保持多尺度 RFF、train/val CSV、hook 機制、位置編碼、數據增強
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.nn.modules import Conv, C2f, SPPF
import yaml


class ImprovedGPSEncoder(nn.Module):
    """改進的 GPS 坐標編碼器 - 加入多尺度 RFF、正規化和位置編碼"""

    def __init__(self, input_dim=2, embed_dim=64, sigmas=None, gps_stats=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigmas = sigmas if sigmas else [0.01, 0.001, 0.0001]

        # 動態設定 GPS 數據範圍
        if gps_stats:
            self.register_buffer('lat_min', torch.tensor(gps_stats['lat_min']))
            self.register_buffer('lat_max', torch.tensor(gps_stats['lat_max']))
            self.register_buffer('lon_min', torch.tensor(gps_stats['lon_min']))
            self.register_buffer('lon_max', torch.tensor(gps_stats['lon_max']))
        else:
            self.register_buffer('lat_min', torch.tensor(-90.0))
            self.register_buffer('lat_max', torch.tensor(90.0))
            self.register_buffer('lon_min', torch.tensor(-180.0))
            self.register_buffer('lon_max', torch.tensor(180.0))
        self.stats_initialized = gps_stats is not None

        # ✅ 多尺度 RFF 初始化
        self.rff_dim_per_scale = embed_dim // len(self.sigmas)
        self.omega_list = nn.ParameterList([
            nn.Parameter(torch.randn((self.rff_dim_per_scale // 2, 2)) / s, requires_grad=False)
            for s in self.sigmas
        ])
        self.b_list = nn.ParameterList([
            nn.Parameter(2 * math.pi * torch.rand(self.rff_dim_per_scale // 2), requires_grad=False)
            for _ in self.sigmas
        ])

        # MLP 編碼
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 位置編碼表
        self.register_buffer('position_table', self._get_sinusoid_encoding_table(1000, embed_dim))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table)

    def forward(self, gps_coords):
        # GPS 正規化
        lat_norm = 2 * (gps_coords[:, 0] - self.lat_min) / (self.lat_max - self.lat_min) - 1
        lon_norm = 2 * (gps_coords[:, 1] - self.lon_min) / (self.lon_max - self.lon_min) - 1
        normalized_gps = torch.stack([lat_norm, lon_norm], dim=1)

        # 訓練時數據增強
        if self.training:
            normalized_gps = normalized_gps + torch.randn_like(normalized_gps) * 0.01

        # ✅ 多尺度 RFF
        rff_parts = []
        for omega, b in zip(self.omega_list, self.b_list):
            proj = normalized_gps @ omega.T + b
            rff_parts.append(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))
        rff = torch.cat(rff_parts, dim=-1)

        # MLP 編碼
        encoded = self.encoder(rff)

        # 位置編碼
        pos_indices = torch.clamp(
            ((normalized_gps[:, 0] + normalized_gps[:, 1] + 2) * 250).long(), 0, 999
        )
        pos_encoding = self.position_table[pos_indices]
        return encoded + pos_encoding


class AdaptiveGPSFeatureFusion(nn.Module):
    """自適應 GPS 特徵融合模塊 - 支持多種融合策略"""

    def __init__(self, feature_channels, gps_embed_dim=64, fusion_type='attention', gps_stats=None):
        super().__init__()
        self.feature_channels = feature_channels
        self.gps_embed_dim = gps_embed_dim
        self.fusion_type = fusion_type
        self.gps_encoder = ImprovedGPSEncoder(embed_dim=gps_embed_dim, gps_stats=gps_stats)

        if fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels // 4, feature_channels, 1),
                nn.Sigmoid()
            )
            self.gps_projection = nn.Conv2d(gps_embed_dim, feature_channels, 1)
        elif fusion_type == 'adaptive':
            self.adaptive_weight = nn.Sequential(
                nn.Conv2d(feature_channels + gps_embed_dim, feature_channels, 1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, feature_channels, 1),
                nn.Sigmoid()
            )
        elif fusion_type == 'multiply':
            self.gps_gate = nn.Sequential(
                nn.Conv2d(gps_embed_dim, feature_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_channels + gps_embed_dim, feature_channels, 1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, features, gps_coords):
        B, C, H, W = features.shape
        gps_features = self.gps_encoder(gps_coords)
        gps_spatial = gps_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

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
        else:
            concatenated = torch.cat([features, gps_spatial], dim=1)
            enhanced_features = self.fusion_conv(concatenated)
        return enhanced_features


class GPSDataManager:
    """GPS 數據管理器"""

    def __init__(self, gps_csv_path):
        self.gps_mapping = self.load_gps_data(gps_csv_path)
        self.gps_stats = None
        self._compute_gps_statistics()

    def load_gps_data(self, gps_csv_path):
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
        return self.gps_stats

    def has_valid_stats(self):
        return self.gps_stats is not None and len(self.gps_mapping) > 0

    def get_gps_batch(self, image_paths):
        batch_gps = []
        for path in image_paths:
            filename = os.path.basename(path)
            gps_coords = self.gps_mapping.get(filename)
            if gps_coords is None:
                if self.gps_stats:
                    gps_coords = torch.tensor([self.gps_stats['lat_mean'], self.gps_stats['lon_mean']], dtype=torch.float32)
                else:
                    gps_coords = torch.tensor([0.0, 0.0], dtype=torch.float32)
                LOGGER.debug(f"Using fallback GPS for {filename}")
            batch_gps.append(gps_coords)
        return torch.stack(batch_gps) if batch_gps else torch.zeros((len(image_paths), 2), dtype=torch.float32)


def get_yolov10_fusion_layers(model_name):
    yolov10_layers = {
        'yolov10n': [9, 12, 15],
        'yolov10s': [12, 15, 18],
        'yolov10m': [15, 18, 21],
        'yolov10l': [18, 21, 24],
        'yolov10x': [21, 24, 27],
    }
    model_lower = model_name.lower()
    for model_key in yolov10_layers:
        if model_key in model_lower:
            return yolov10_layers[model_key]
    return [15, 18, 21]


class GPSEnhancedTrainer:
    """GPS 增強的訓練器 - v3.1 修正版"""

    def __init__(self, model_path, gps_train_manager: GPSDataManager):
        self.model = YOLO(model_path)
        self.gps_manager = gps_train_manager
        self.val_gps_manager = None
        self.fusion_layers = get_yolov10_fusion_layers(model_path)
        self.fusion_type = 'attention'
        self.gps_embed_dim = 96

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
                            output.shape[1], self.gps_embed_dim, self.fusion_type, self.gps_manager.get_gps_stats()
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
        gps_manager = self.val_gps_manager if (is_val and self.val_gps_manager) else self.gps_manager
        self.current_gps_batch = gps_manager.get_gps_batch(image_paths)

    def _extract_image_paths(self, trainer):
        """✅ 修復 YOLO v8~v10 新舊版本兼容"""
        try:
            # 新版 (v8.2+ / v10)
            if hasattr(trainer, 'batch') and trainer.batch is not None:
                batch = trainer.batch
                if isinstance(batch, dict):
                    return batch.get('im_file') or batch.get('path') or batch.get('paths')
                elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                    # 嘗試從列表提取
                    if isinstance(batch[0], dict):
                        return batch[0].get('im_file') or batch[0].get('path')
            return None
        except Exception as e:
            LOGGER.warning(f"Failed to extract image paths: {e}")
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
    parser = argparse.ArgumentParser(description='GPS-Enhanced YOLO Training v3.1 (多尺度 RFF + 新版 YOLO 適配)')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--gps-train-csv', type=str, required=True, help='Train GPS CSV file')
    parser.add_argument('--gps-val-csv', type=str, required=True, help='Validation GPS CSV file')
    parser.add_argument('--model', type=str, default='yolov10n.pt', help='Model file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device')
    args = parser.parse_args()

    gps_train_manager = GPSDataManager(args.gps_train_csv)
    gps_val_manager = GPSDataManager(args.gps_val_csv)

    trainer = GPSEnhancedTrainer(args.model, gps_train_manager)
    trainer.val_gps_manager = gps_val_manager

    LOGGER.info("====== 開始訓練 ======")
    trainer.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size, device=args.device)

    LOGGER.info("====== 驗證階段 ======")
    trainer.validate(data=args.data, imgsz=args.imgsz, batch=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
