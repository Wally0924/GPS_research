#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import csv

# ========== ImprovedGPSEncoder ==========
class ImprovedGPSEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=96, sigmas=None, gps_stats=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigmas = sigmas if sigmas else [0.01, 0.001, 0.0001]
        assert embed_dim % len(self.sigmas) == 0, "embed_dim 必須是 len(sigmas) 的整數倍！"

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

        self.rff_dim_per_scale = embed_dim // len(self.sigmas)
        self.omega_list = nn.ParameterList([
            nn.Parameter(torch.randn((self.rff_dim_per_scale // 2, 2)) / s, requires_grad=False)
            for s in self.sigmas
        ])
        self.b_list = nn.ParameterList([
            nn.Parameter(2 * math.pi * torch.rand(self.rff_dim_per_scale // 2), requires_grad=False)
            for _ in self.sigmas
        ])

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.register_buffer('position_table', self._get_sinusoid_encoding_table(1000, embed_dim))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table)

    def forward(self, gps_coords):
        lat_norm = 2 * (gps_coords[:, 0] - self.lat_min) / (self.lat_max - self.lat_min) - 1
        lon_norm = 2 * (gps_coords[:, 1] - self.lon_min) / (self.lon_max - self.lon_min) - 1
        normalized_gps = torch.stack([lat_norm, lon_norm], dim=1)

        if self.training:
            normalized_gps = normalized_gps + torch.randn_like(normalized_gps) * 0.01

        rff_parts = []
        for omega, b in zip(self.omega_list, self.b_list):
            proj = normalized_gps @ omega.T + b
            rff_parts.append(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))
        rff = torch.cat(rff_parts, dim=-1)
        encoded = self.encoder(rff)

        pos_indices = torch.clamp(((normalized_gps[:, 0] + normalized_gps[:, 1] + 2) * 250).long(), 0, 999)
        pos_encoding = self.position_table[pos_indices]
        return encoded + pos_encoding


# ========== AdaptiveGPSFeatureFusion ==========
class AdaptiveGPSFeatureFusion(nn.Module):
    def __init__(self, feature_channels, gps_embed_dim=96, fusion_type='attention', gps_stats=None):
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
        else:
            concatenated = torch.cat([features, gps_spatial], dim=1)
            enhanced_features = self.fusion_conv(concatenated)


        if enhanced_features.shape[2:] != (H, W):
            enhanced_features = F.interpolate(enhanced_features, size=(H, W), mode="bilinear", align_corners=False)
        return enhanced_features


# ========== GPSDataManager ==========
class GPSDataManager:
    def __init__(self, gps_csv_path):
        self.gps_mapping = self.load_gps_data(gps_csv_path)
        self.gps_stats = None
        self._compute_gps_statistics()

    def load_gps_data(self, gps_csv_path):
        try:
            gps_df = pd.read_csv(gps_csv_path)
            gps_mapping = {row['filename']: torch.tensor([row['lat'], row['long']], dtype=torch.float32)
                           for _, row in gps_df.iterrows()}
            LOGGER.info(f"Loaded GPS data: {len(gps_mapping)} records")
            return gps_mapping
        except Exception as e:
            LOGGER.error(f"Error loading GPS data: {e}")
            return {}

    def _compute_gps_statistics(self):
        if not self.gps_mapping:
            return
        gps_tensor = torch.stack(list(self.gps_mapping.values()))
        self.gps_stats = {
            'lat_min': gps_tensor[:, 0].min().item(),
            'lat_max': gps_tensor[:, 0].max().item(),
            'lat_mean': gps_tensor[:, 0].mean().item(),
            'lon_min': gps_tensor[:, 1].min().item(),
            'lon_max': gps_tensor[:, 1].max().item(),
            'lon_mean': gps_tensor[:, 1].mean().item(),
        }

    def get_gps_stats(self):
        return self.gps_stats

    def get_gps_batch(self, image_paths):
        batch_gps = []
        for path in image_paths:
            filename = os.path.basename(path)
            gps_coords = self.gps_mapping.get(filename, None)
            if gps_coords is None:
                gps_coords = torch.tensor([self.gps_stats['lat_mean'], self.gps_stats['lon_mean']], dtype=torch.float32)
            batch_gps.append(gps_coords)
        return torch.stack(batch_gps)


# ========== Fusion Hooks Trainer ==========
def get_yolov10_fusion_layers(model_name):
    layers = {
        'yolov10n': [9, 12, 15],
        'yolov10s': [12, 15, 18],
        'yolov10m': [15, 18, 21],
        'yolov10l': [18, 21, 24],
        'yolov10x': [21, 24, 27],
    }
    for k in layers:
        if k in model_name.lower():
            return layers[k]
    return [15, 18, 21]


class GPSEnhancedTrainer:
    def __init__(self, model_path, gps_manager):
        self.model = YOLO(model_path)
        self.gps_manager = gps_manager
        self.val_gps_manager = None
        self.fusion_layers = get_yolov10_fusion_layers(model_path)
        self.gps_embed_dim = 96
        self.fusion_modules, self.hooks = {}, []
        self.current_gps_batch, self.gps_fusion_enabled = None, False

    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if not self.gps_fusion_enabled or self.current_gps_batch is None:
                    return output
                gps_coords = self.current_gps_batch.to(output.device)


                if gps_coords.shape[0] != output.shape[0]:
                    gps_coords = gps_coords[:output.shape[0], :]

                if layer_idx not in self.fusion_modules:
                    self.fusion_modules[layer_idx] = AdaptiveGPSFeatureFusion(
                        output.shape[1], self.gps_embed_dim, gps_stats=self.gps_manager.get_gps_stats()
                    ).to(output.device)
                    LOGGER.info(f"[HOOK] GPS Fusion Module registered at layer {layer_idx}")

                enhanced = self.fusion_modules[layer_idx](output, gps_coords)
                LOGGER.info(f"[HOOK] Layer {layer_idx}: before {tuple(output.shape)} -> after {tuple(enhanced.shape)}")
                return enhanced
            return hook_fn
        for idx in self.fusion_layers:
            self.hooks.append(self.model.model.model[idx].register_forward_hook(create_hook(idx)))

    def _set_batch(self, trainer, is_val=False):

        if not is_val:
            batch = trainer.batch_data if hasattr(trainer, 'batch_data') else getattr(trainer, 'batch', None)
            if isinstance(batch, dict) and 'im_file' in batch:
                self.set_gps_batch(batch['im_file'])
                self.gps_fusion_enabled = True
                LOGGER.info("[TRAIN] GPS Hook Enabled")

        else:
            if hasattr(trainer, 'batch') and isinstance(trainer.batch, (list, tuple)) and len(trainer.batch) >= 3:
                image_paths = trainer.batch[2]
                self.set_gps_batch(image_paths)
                self.gps_fusion_enabled = True
                LOGGER.info("[VAL] GPS Hook Enabled")

    def set_gps_batch(self, image_paths):
        self.current_gps_batch = self.val_gps_manager.get_gps_batch(image_paths)

    def train(self, **kwargs):
        self._register_hooks()
        self.model.add_callback('on_train_batch_start', lambda t: self._set_batch(t, is_val=False))
        return self.model.train(**kwargs)

    def validate(self, **kwargs):
        self._register_hooks()
        self.model.add_callback('on_val_batch_start', lambda t: self._set_batch(t, is_val=True))
        return self.model.val(**kwargs)

    def predict(self, source, imgsz=640):
        self._register_hooks()
        self.gps_fusion_enabled = True


        image_paths = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.jpg', '.png'))]
        if len(image_paths) > 1:
            LOGGER.warning(f"[PREDICT] 多圖批次模式，建議逐張處理，目前圖片數 {len(image_paths)}")
        self.current_gps_batch = self.val_gps_manager.get_gps_batch([image_paths[0]])

        LOGGER.info("[PREDICT] GPS Hook Enabled")
        return self.model.predict(source=source, save=True, save_txt=True, imgsz=imgsz)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description='GPS-Enhanced YOLO v4.3.1')
    parser.add_argument('--data', type=str, help='Dataset YAML file')
    parser.add_argument('--gps-train-csv', type=str, help='Train GPS CSV')
    parser.add_argument('--gps-val-csv', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--val', action='store_true', help='Only run validation')
    parser.add_argument('--source', type=str, help='Source for predict')
    args = parser.parse_args()

    gps_train_manager = GPSDataManager(args.gps_train_csv) if args.gps_train_csv else None
    gps_val_manager = GPSDataManager(args.gps_val_csv)

    trainer = GPSEnhancedTrainer(args.model, gps_train_manager or gps_val_manager)
    trainer.val_gps_manager = gps_val_manager

    if args.predict:
        trainer.predict(source=args.source, imgsz=args.imgsz)
    elif args.val:
        results = trainer.validate(data=args.data, batch=args.batch_size, imgsz=args.imgsz)

        print("\n========== Validation Results ==========")
        print(f"mAP@50 (是否抓到框): {results.box.map50:.4f}")
        print(f"mAP@50-95 (精細定位能力): {results.box.map:.4f}")
        print("========================================\n")


        print("Per-class AP50-95 Results:")
        for cls_id, cls_name in results.names.items():
            print(f"{cls_name}: {results.box.maps[cls_id]:.4f}")


        csv_filename = os.path.join(results.save_dir, "val_results.csv")

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "mAP50-95"])
            for cls_id, cls_name in results.names.items():
                writer.writerow([cls_name, results.box.maps[cls_id]])

        print(f"\nValidation results saved to: {os.path.abspath(csv_filename)}")
        print("========================================\n")

    else:
        trainer.train(data=args.data, epochs=args.epochs, batch=args.batch_size, imgsz=args.imgsz)
        trainer.validate(data=args.data, batch=args.batch_size, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
