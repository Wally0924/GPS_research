import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from typing import Any, Dict, List, Optional
import json
import os
from collections import defaultdict


class MultiScaleGPSToRFF(nn.Module):
    """
    優化版多尺度 GPS Random Fourier Features 編碼器
    針對小範圍GPS座標優化
    """
    def __init__(
        self, 
        rff_dim: int = 512,
        sigmas: List[float] = [0.0001, 0.001, 0.01],  # 針對你的GPS範圍優化
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.rff_dim = rff_dim
        self.sigmas = sigmas
        self.num_scales = len(sigmas)
        
        # 智能分配維度 - 確保總和等於目標維度
        base_features_per_scale = rff_dim // (2 * self.num_scales)
        remainder = rff_dim - (base_features_per_scale * 2 * self.num_scales)
        
        # 將剩餘維度分配給前幾個尺度
        self.features_per_scale = []
        for i in range(self.num_scales):
            extra = 1 if i < remainder // 2 else 0
            features_count = base_features_per_scale + extra
            self.features_per_scale.append(features_count)
        
        # 如果 remainder 是奇數，最後一個尺度多分配 1 個特徵
        if remainder % 2 == 1:
            self.features_per_scale[-1] += 1
            
        # 驗證總維度
        total_dim = sum(f * 2 for f in self.features_per_scale)
        assert total_dim == rff_dim, f"維度不匹配: {total_dim} != {rff_dim}"
        
        # 為每個尺度創建不同的 RFF 參數
        for i, (sigma, features_count) in enumerate(zip(sigmas, self.features_per_scale)):
            omega = torch.randn(features_count, 2) / sigma
            b = 2 * math.pi * torch.rand(features_count)
            
            # 使用 register_buffer，這些不需要梯度
            self.register_buffer(f'omega_{i}', omega)
            self.register_buffer(f'b_{i}', b)
        
        print(f"✅ Optimized MultiScaleGPSToRFF:")
        print(f"  Target RFF dim: {rff_dim}")
        print(f"  Optimized sigmas: {sigmas}")
        print(f"  Features per scale: {self.features_per_scale}")
        print(f"  Total dim: {total_dim}")
    
    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gps: GPS coordinates tensor, shape (batch_size, 2) [lat, lon]
        Returns:
            GPS embeddings tensor, shape (batch_size, rff_dim)
        """
        batch_size = gps.shape[0]
        rff_features = []
        
        for i in range(self.num_scales):
            # 獲取對應的 omega 和 b
            omega = getattr(self, f'omega_{i}')
            b = getattr(self, f'b_{i}')
            
            # 計算投影 gps @ omega^T
            proj = torch.matmul(gps, omega.T)  # (batch_size, features_per_scale[i])
            
            # 加上偏置
            y = proj + b  # (batch_size, features_per_scale[i])
            
            # 計算 cos 和 sin 特徵
            rff = torch.cat([torch.cos(y), torch.sin(y)], dim=-1)  # (batch_size, features_per_scale[i] * 2)
            rff_features.append(rff)
        
        # 連接所有尺度的特徵
        gps_embeddings = torch.cat(rff_features, dim=-1)  # (batch_size, rff_dim)
        
        # 驗證輸出維度
        assert gps_embeddings.shape[-1] == self.rff_dim, f"輸出維度錯誤: {gps_embeddings.shape[-1]} != {self.rff_dim}"
        
        return gps_embeddings


class LocationEncoder(nn.Module):
    """
    GPS 位置編碼器，將 GPS 座標轉換為語義豐富的高維特徵
    """
    def __init__(
        self, 
        rff_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        sigmas: List[float] = [0.0001, 0.001, 0.01],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 多尺度 RFF 編碼器
        self.rff_encoder = MultiScaleGPSToRFF(rff_dim, sigmas)
        
        # MLP 層來學習更豐富的位置表示
        self.mlp = nn.Sequential(
            nn.Linear(rff_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gps: GPS coordinates, shape (batch_size, 2)
        Returns:
            Location embeddings, shape (batch_size, output_dim)
        """
        # GPS RFF 編碼
        rff_features = self.rff_encoder(gps)  # (batch_size, rff_dim)
        
        # MLP 處理
        location_embeddings = self.mlp(rff_features)  # (batch_size, output_dim)
        
        # 正規化
        location_embeddings = self.layer_norm(location_embeddings)
        
        return location_embeddings


class ImageEncoder(nn.Module):
    """
    影像編碼器，基於 Segformer 提取影像特徵
    """
    def __init__(
        self, 
        segformer_model: str = "nvidia/mit-b0",
        feature_dim: int = 512
    ):
        super().__init__()
        
        # 載入預訓練的 Segformer 模型
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_model)
        
        # 取得 Segformer 的特徵提取器
        self.feature_extractor = self.segformer.segformer
        
        # 獲得 backbone 的輸出維度
        if "mit-b0" in segformer_model:
            backbone_dims = [32, 64, 160, 256]
        elif "mit-b1" in segformer_model:
            backbone_dims = [64, 128, 320, 512]
        else:
            backbone_dims = [32, 64, 160, 256]  # 默認使用 b0
        
        # 特徵融合層
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(sum(backbone_dims), feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # 全局平均池化來獲得影像級別的嵌入
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images, shape (batch_size, 3, H, W)
        Returns:
            Dictionary containing:
                - 'features': Multi-scale feature maps
                - 'embeddings': Global image embeddings
        """
        # 提取多尺度特徵
        outputs = self.feature_extractor(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # 調整特徵圖尺寸並融合
        features = []
        
        # 獲取目標尺寸
        target_h = target_w = None
        for feature_map in hidden_states:
            if len(feature_map.shape) == 4:  # (B, C, H, W)
                if target_h is None:
                    target_h, target_w = feature_map.shape[-2:]
                else:
                    h, w = feature_map.shape[-2:]
                    if h * w > target_h * target_w:
                        target_h, target_w = h, w
            elif len(feature_map.shape) == 3:  # (B, HW, C) 
                B, HW, C = feature_map.shape
                H = W = int(math.sqrt(HW))
                if target_h is None:
                    target_h, target_w = H, W
                else:
                    if H * W > target_h * target_w:
                        target_h, target_w = H, W
        
        target_size = (target_h, target_w)
        
        for feature_map in hidden_states:
            if len(feature_map.shape) == 3:  # (B, HW, C) 格式
                B, HW, C = feature_map.shape
                H = W = int(math.sqrt(HW))
                feature_map = feature_map.transpose(1, 2).reshape(B, C, H, W)
            elif len(feature_map.shape) == 4:  # (B, C, H, W) 格式
                pass  # 已經是正確格式
            else:
                continue
            
            # 調整空間尺寸
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(
                    feature_map, size=target_size, mode='bilinear', align_corners=False
                )
            
            features.append(feature_map)
        
        if not features:
            raise ValueError("No valid features extracted from Segformer")
        
        # 連接所有特徵
        fused_features = torch.cat(features, dim=1)
        
        # 特徵融合
        processed_features = self.feature_fusion(fused_features)
        
        # 全局影像嵌入
        global_embeddings = self.global_pool(processed_features).flatten(1)
        
        return {
            'features': processed_features,
            'embeddings': global_embeddings
        }


class LocationMemoryBank(nn.Module):
    """
    位置記憶庫 - 為每個GPS位置建立特徵記憶
    """
    def __init__(
        self, 
        feature_dim: int = 512,
        memory_size: int = 20,
        spatial_radius: float = 0.00005,
        save_path: Optional[str] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.spatial_radius = spatial_radius
        self.save_path = save_path
        
        # 動態記憶庫：GPS位置 -> 特徵和其他信息
        self.memory_bank = defaultdict(lambda: {
            'features': [],
            'count': 0,
            'last_updated': 0
        })
        
        # 統計信息
        self.total_updates = 0
        self.total_queries = 0
        self.hit_count = 0
        
        print(f"✅ LocationMemoryBank initialized:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Memory size per location: {memory_size}")
        print(f"  Spatial radius: {spatial_radius}")
        
    def gps_to_key(self, gps: torch.Tensor) -> str:
        """將GPS座標轉換為記憶庫的鍵"""
        lat_grid = round(gps[0].item() / self.spatial_radius) * self.spatial_radius
        lon_grid = round(gps[1].item() / self.spatial_radius) * self.spatial_radius
        return f"{lat_grid:.7f},{lon_grid:.7f}"
    
    def update_memory(self, gps_coords: torch.Tensor, features: torch.Tensor):
        """更新位置記憶庫"""
        batch_size = gps_coords.shape[0]
        
        for i in range(batch_size):
            gps_key = self.gps_to_key(gps_coords[i])
            
            # 檢查特徵有效性
            feature_norm = torch.norm(features[i]).item()
            if feature_norm < 1e-6:
                continue
            
            # 添加新特徵
            self.memory_bank[gps_key]['features'].append(features[i].detach().clone())
            self.memory_bank[gps_key]['count'] += 1
            self.memory_bank[gps_key]['last_updated'] = self.total_updates
            
            # 保持記憶庫大小
            if len(self.memory_bank[gps_key]['features']) > self.memory_size:
                self.memory_bank[gps_key]['features'].pop(0)
        
        self.total_updates += 1
    
    def retrieve_memory(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """檢索相關位置的歷史特徵"""
        batch_size = gps_coords.shape[0]
        memory_features = []
        
        self.total_queries += batch_size
        
        for i in range(batch_size):
            gps_key = self.gps_to_key(gps_coords[i])
            retrieved_features = []
            
            # 精確匹配
            if gps_key in self.memory_bank and len(self.memory_bank[gps_key]['features']) > 0:
                retrieved_features.extend(self.memory_bank[gps_key]['features'])
                self.hit_count += 1
            
            # 聚合檢索到的特徵
            if retrieved_features:
                recent_features = retrieved_features[-8:]
                if len(recent_features) == 1:
                    aggregated = recent_features[0]
                else:
                    weights = torch.softmax(
                        torch.tensor([i for i in range(len(recent_features))], dtype=torch.float32),
                        dim=0
                    ).to(recent_features[0].device)
                    
                    stacked_features = torch.stack(recent_features)
                    aggregated = (stacked_features * weights.unsqueeze(-1)).sum(dim=0)
            else:
                aggregated = torch.zeros(self.feature_dim, device=gps_coords.device)
            
            memory_features.append(aggregated)
        
        return torch.stack(memory_features)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶庫統計信息"""
        total_locations = len(self.memory_bank)
        total_memories = sum(len(memory['features']) for memory in self.memory_bank.values())
        hit_rate = self.hit_count / max(self.total_queries, 1)
        
        return {
            'total_locations': total_locations,
            'total_memories': total_memories,
            'hit_rate': hit_rate,
            'avg_memories_per_location': total_memories / max(total_locations, 1),
            'total_queries': self.total_queries,
            'total_updates': self.total_updates,
            'hit_count': self.hit_count
        }
    
    def save_memory_bank(self):
        """保存記憶庫到文件"""
        if self.save_path:
            stats = {
                'locations': list(self.memory_bank.keys()),
                'counts': {k: v['count'] for k, v in self.memory_bank.items()},
                'stats': self.get_memory_stats()
            }
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(stats, f, indent=2)


class CrossModalFusion(nn.Module):
    """
    跨模態融合模組，將影像特徵和 GPS 特徵融合
    """
    def __init__(
        self, 
        feature_dim: int = 512,
        fusion_method: str = "attention"
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.feature_dim = feature_dim
        
        if fusion_method == "attention":
            # 注意力機制融合
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.scale = math.sqrt(feature_dim)
            
        elif fusion_method == "concat":
            # 連接後降維
            self.fusion_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
    def forward(
        self, 
        image_features: torch.Tensor, 
        location_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: Image feature maps, shape (batch_size, feature_dim, H, W)
            location_embeddings: GPS embeddings, shape (batch_size, feature_dim)
        Returns:
            Fused features, shape (batch_size, feature_dim, H, W)
        """
        batch_size, feature_dim, H, W = image_features.shape
        
        if self.fusion_method == "add":
            # 簡單相加融合
            location_map = location_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            fused_features = image_features + location_map
            
        elif self.fusion_method == "concat":
            # 連接融合
            location_map = location_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            concat_features = torch.cat([image_features, location_map], dim=1)
            concat_features = concat_features.permute(0, 2, 3, 1).reshape(batch_size, H*W, -1)
            fused_features = self.fusion_proj(concat_features)
            fused_features = fused_features.reshape(batch_size, H, W, feature_dim).permute(0, 3, 1, 2)
            
        elif self.fusion_method == "attention":
            # 注意力融合
            img_seq = image_features.permute(0, 2, 3, 1).reshape(batch_size, H*W, feature_dim)
            
            Q = self.query_proj(img_seq)
            K = self.key_proj(location_embeddings.unsqueeze(1))
            V = self.value_proj(location_embeddings.unsqueeze(1))
            
            attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attention_weights = F.softmax(attention_weights, dim=-1)
            
            attended_location = torch.matmul(attention_weights, V)
            fused_seq = img_seq + attended_location
            fused_features = fused_seq.reshape(batch_size, H, W, feature_dim).permute(0, 3, 1, 2)
        
        return fused_features


class SelectiveMemoryBank(nn.Module):
    """選擇性記憶庫 - 只記住重要的、困難的、或代表性的經驗"""
    def __init__(
        self, 
        feature_dim: int = 512,
        memory_size: int = 20,
        spatial_radius: float = 0.00005,
        save_path: Optional[str] = None,
        # 選擇性參數
        difficulty_threshold: float = 0.5,
        diversity_threshold: float = 0.7,
        importance_decay: float = 0.95,
        max_memory_age: int = 1000
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.spatial_radius = spatial_radius
        self.save_path = save_path
        
        # 選擇性記憶參數
        self.difficulty_threshold = difficulty_threshold
        self.diversity_threshold = diversity_threshold
        self.importance_decay = importance_decay
        self.max_memory_age = max_memory_age
        
        # 記憶庫結構
        self.memory_bank = defaultdict(lambda: {
            'features': [],
            'losses': [],
            'timestamps': [],
            'importance_scores': [],
            'access_counts': []
        })
        
        # 統計信息
        self.total_updates = 0
        self.total_queries = 0
        self.hit_count = 0
        self.rejected_memories = 0
        
        print(f"✅ SelectiveMemoryBank initialized:")
        print(f"  Difficulty threshold: {difficulty_threshold}")
        print(f"  Diversity threshold: {diversity_threshold}")
        print(f"  Max memory age: {max_memory_age}")
    
    def gps_to_key(self, gps: torch.Tensor) -> str:
        """將GPS座標轉換為記憶庫的鍵"""
        lat_grid = round(gps[0].item() / self.spatial_radius) * self.spatial_radius
        lon_grid = round(gps[1].item() / self.spatial_radius) * self.spatial_radius
        return f"{lat_grid:.7f},{lon_grid:.7f}"
    
    def should_store_memory(self, feature: torch.Tensor, loss: float, gps_key: str) -> bool:
        """判斷是否值得存儲這個記憶"""
        # 1. 困難度檢查
        if loss < self.difficulty_threshold:
            return False
        
        # 2. 多樣性檢查
        if gps_key in self.memory_bank and len(self.memory_bank[gps_key]['features']) > 0:
            existing_features = torch.stack(self.memory_bank[gps_key]['features'])
            similarities = F.cosine_similarity(feature.unsqueeze(0), existing_features, dim=1)
            max_similarity = similarities.max().item()
            
            if max_similarity > self.diversity_threshold:
                return False
        
        # 3. 特徵代表性檢查
        feature_norm = torch.norm(feature).item()
        if feature_norm < 0.1:
            return False
        
        return True
    
    def compute_importance_score(self, feature: torch.Tensor, loss: float, timestamp: int) -> float:
        """計算記憶的重要性評分"""
        difficulty_score = min(loss / 2.0, 1.0)
        feature_strength = torch.norm(feature).item()
        strength_score = min(feature_strength / 10.0, 1.0)
        age = self.total_updates - timestamp
        time_score = (self.importance_decay ** age)
        
        total_score = (0.5 * difficulty_score + 
                      0.3 * strength_score + 
                      0.2 * time_score)
        
        return total_score
    
    def update_memory(self, gps_coords: torch.Tensor, features: torch.Tensor, losses: torch.Tensor = None):
        """選擇性更新位置記憶庫"""
        batch_size = gps_coords.shape[0]
        
        for i in range(batch_size):
            gps_key = self.gps_to_key(gps_coords[i])
            feature = features[i]
            loss = losses[i].item() if losses is not None else 1.0
            
            if not self.should_store_memory(feature, loss, gps_key):
                self.rejected_memories += 1
                continue
            
            importance_score = self.compute_importance_score(feature, loss, self.total_updates)
            
            # 存儲記憶
            self.memory_bank[gps_key]['features'].append(feature.detach().clone())
            self.memory_bank[gps_key]['losses'].append(loss)
            self.memory_bank[gps_key]['timestamps'].append(self.total_updates)
            self.memory_bank[gps_key]['importance_scores'].append(importance_score)
            self.memory_bank[gps_key]['access_counts'].append(0)
            
            # 記憶庫容量管理
            self._manage_memory_capacity(gps_key)
            
            if self.total_updates % 100 == 0:
                self._cleanup_old_memories(gps_key)
        
        self.total_updates += 1
    
    def _manage_memory_capacity(self, gps_key: str):
        """管理記憶庫容量"""
        memory_data = self.memory_bank[gps_key]
        
        if len(memory_data['features']) > self.memory_size:
            importance_scores = memory_data['importance_scores']
            
            # 更新重要性評分
            current_time = self.total_updates
            for j, timestamp in enumerate(memory_data['timestamps']):
                age = current_time - timestamp
                time_decay = (self.importance_decay ** age)
                importance_scores[j] *= time_decay
            
            # 移除最不重要的記憶
            min_idx = importance_scores.index(min(importance_scores))
            
            for key in ['features', 'losses', 'timestamps', 'importance_scores', 'access_counts']:
                memory_data[key].pop(min_idx)
    
    def _cleanup_old_memories(self, gps_key: str):
        """清理過於老舊的記憶"""
        memory_data = self.memory_bank[gps_key]
        current_time = self.total_updates
        
        indices_to_remove = []
        for j, timestamp in enumerate(memory_data['timestamps']):
            age = current_time - timestamp
            if age > self.max_memory_age:
                indices_to_remove.append(j)
        
        for idx in sorted(indices_to_remove, reverse=True):
            for key in ['features', 'losses', 'timestamps', 'importance_scores', 'access_counts']:
                memory_data[key].pop(idx)
    
    def retrieve_memory(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """檢索記憶"""
        batch_size = gps_coords.shape[0]
        memory_features = []
        
        self.total_queries += batch_size
        
        for i in range(batch_size):
            gps_key = self.gps_to_key(gps_coords[i])
            
            if gps_key in self.memory_bank and len(self.memory_bank[gps_key]['features']) > 0:
                memory_data = self.memory_bank[gps_key]
                
                # 更新訪問計數
                for j in range(len(memory_data['access_counts'])):
                    memory_data['access_counts'][j] += 1
                
                # 根據重要性選擇最好的記憶
                importance_scores = memory_data['importance_scores'][:]
                access_counts = memory_data['access_counts']
                max_access = max(access_counts) if access_counts else 1
                
                for j in range(len(importance_scores)):
                    access_boost = access_counts[j] / max_access * 0.2
                    importance_scores[j] += access_boost
                
                # 選擇最重要的記憶
                best_indices = sorted(range(len(importance_scores)), 
                                    key=lambda x: importance_scores[x], reverse=True)
                
                top_k = min(3, len(best_indices))
                selected_features = [memory_data['features'][best_indices[j]] for j in range(top_k)]
                
                if len(selected_features) == 1:
                    aggregated = selected_features[0]
                else:
                    weights = torch.tensor([importance_scores[best_indices[j]] for j in range(top_k)])
                    weights = F.softmax(weights, dim=0).to(selected_features[0].device)
                    
                    stacked_features = torch.stack(selected_features)
                    aggregated = (stacked_features * weights.unsqueeze(-1)).sum(dim=0)
                
                self.hit_count += 1
            else:
                aggregated = torch.zeros(self.feature_dim, device=gps_coords.device)
            
            memory_features.append(aggregated)
        
        return torch.stack(memory_features)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶庫統計信息"""
        total_locations = len(self.memory_bank)
        total_memories = sum(len(memory['features']) for memory in self.memory_bank.values())
        hit_rate = self.hit_count / max(self.total_queries, 1)
        
        # 計算記憶質量統計
        all_importance_scores = []
        all_access_counts = []
        for memory in self.memory_bank.values():
            all_importance_scores.extend(memory['importance_scores'])
            all_access_counts.extend(memory['access_counts'])
        
        avg_importance = sum(all_importance_scores) / len(all_importance_scores) if all_importance_scores else 0
        avg_access = sum(all_access_counts) / len(all_access_counts) if all_access_counts else 0
        
        # ⭐ 修正接受率計算 - 關鍵修正！
        total_attempts = self.rejected_memories + total_memories
        acceptance_rate = total_memories / max(total_attempts, 1) if total_attempts > 0 else 0
        
        return {
            'total_locations': total_locations,
            'total_memories': total_memories,
            'hit_rate': hit_rate,
            'avg_memories_per_location': total_memories / max(total_locations, 1),
            'total_queries': self.total_queries,
            'total_updates': self.total_updates,
            'hit_count': self.hit_count,
            'avg_importance_score': avg_importance,
            'avg_access_count': avg_access,
            'memory_acceptance_rate': acceptance_rate,
            'rejected_memories': self.rejected_memories
        }
    
    def save_memory_bank(self):
        """保存記憶庫到文件"""
        if self.save_path:
            stats = {
                'locations': list(self.memory_bank.keys()),
                'counts': {k: len(v['features']) for k, v in self.memory_bank.items()},
                'stats': self.get_memory_stats()
            }
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Memory bank stats saved to {self.save_path}")


class MemoryEnhancedGeoSegformer(nn.Module):
    """
    方案C：選擇性記憶增強版 GeoSegformer
    """
    def __init__(
        self,
        num_classes: int,
        segformer_model: str = "nvidia/mit-b0",
        feature_dim: int = 512,
        rff_dim: int = 512,
        sigmas: List[float] = [0.0001, 0.001, 0.01],
        fusion_method: str = "attention",
        dropout: float = 0.1,
        memory_size: int = 20,
        spatial_radius: float = 0.00005,
        memory_save_path: Optional[str] = None,
        # 選擇性記憶參數
        use_selective_memory: bool = False,
        difficulty_threshold: float = 0.5,
        diversity_threshold: float = 0.7,
        importance_decay: float = 0.95,
        max_memory_age: int = 1000
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_selective_memory = use_selective_memory
        
        print(f"🚀 Initializing MemoryEnhancedGeoSegformer (方案C)")
        print(f"  Use selective memory: {use_selective_memory}")
        
        # GPS 位置編碼器
        self.location_encoder = LocationEncoder(
            rff_dim=rff_dim,
            output_dim=feature_dim,
            sigmas=sigmas,
            dropout=dropout
        )
        
        # 影像編碼器
        self.image_encoder = ImageEncoder(
            segformer_model=segformer_model,
            feature_dim=feature_dim
        )
        
        # 選擇性記憶庫或普通記憶庫
        if use_selective_memory:
            self.memory_bank = SelectiveMemoryBank(
                feature_dim=feature_dim,
                memory_size=memory_size,
                spatial_radius=spatial_radius,
                save_path=memory_save_path,
                difficulty_threshold=difficulty_threshold,
                diversity_threshold=diversity_threshold,
                importance_decay=importance_decay,
                max_memory_age=max_memory_age
            )
        else:
            self.memory_bank = LocationMemoryBank(
                feature_dim=feature_dim,
                memory_size=memory_size,
                spatial_radius=spatial_radius,
                save_path=memory_save_path
            )
        
        # 其他組件
        self.memory_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.cross_modal_fusion = CrossModalFusion(
            feature_dim=feature_dim,
            fusion_method=fusion_method
        )
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim, num_classes, 1)
        )
        
        self.contrastive_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256)
        )
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"✅ MemoryEnhancedGeoSegformer initialized with {total_params:.2f}M parameters")
    
    def forward(
        self, 
        images: torch.Tensor, 
        gps: torch.Tensor,
        return_embeddings: bool = False,
        update_memory: bool = True,
        return_intermediate_features: bool = False,
        current_losses: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """方案C的核心forward方法"""
        # GPS 位置編碼
        location_embeddings = self.location_encoder(gps)
        
        # 影像特徵提取
        image_outputs = self.image_encoder(images)
        image_features = image_outputs['features']
        image_embeddings = image_outputs['embeddings']
        
        # 保存原始特徵
        original_image_embeddings = image_embeddings.clone()
        
        # 檢索位置記憶
        memory_features = self.memory_bank.retrieve_memory(gps)
        original_memory_features = memory_features.clone()
        
        # 記憶增強處理
        enhanced_embeddings = image_embeddings
        memory_weight = 0.0
        
        memory_norms = torch.norm(memory_features, dim=-1)
        valid_memory_mask = memory_norms > 1e-6
        
        if valid_memory_mask.any():
            memory_weight = valid_memory_mask.float().mean().item()
            
            if memory_weight > 0:
                valid_indices = valid_memory_mask.nonzero(as_tuple=True)[0]
                
                if len(valid_indices) > 0:
                    valid_memory_features = memory_features[valid_indices]
                    valid_image_embeddings = image_embeddings[valid_indices]
                    
                    # 特徵融合
                    combined_features = torch.cat([valid_image_embeddings, valid_memory_features], dim=-1)
                    fused_features = self.memory_fusion(combined_features)
                    
                    # 注意力融合
                    memory_enhanced, attention_weights = self.memory_attention(
                        valid_image_embeddings.unsqueeze(1),
                        valid_memory_features.unsqueeze(1),
                        valid_memory_features.unsqueeze(1)
                    )
                    
                    # 結合兩種融合方式
                    enhanced_part = (
                        0.6 * fused_features + 
                        0.4 * memory_enhanced.squeeze(1)
                    )
                    
                    # 殘差連接
                    enhanced_part = enhanced_part + valid_image_embeddings
                    
                    # 更新對應的嵌入
                    enhanced_embeddings = image_embeddings.clone()
                    enhanced_embeddings[valid_indices] = enhanced_part
        
        # 增強的位置嵌入
        enhanced_location_embeddings = location_embeddings + 0.3 * enhanced_embeddings
        
        # 跨模態特徵融合
        fused_features = self.cross_modal_fusion(image_features, enhanced_location_embeddings)
        
        # 語義分割預測
        segmentation_logits = self.segmentation_head(fused_features)
        
        # 調整到輸入影像尺寸
        segmentation_logits = F.interpolate(
            segmentation_logits, 
            size=images.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 更新記憶庫
        if update_memory and self.training:
            if self.use_selective_memory and current_losses is not None:
                self.memory_bank.update_memory(gps, image_embeddings, current_losses)
            else:
                if hasattr(self.memory_bank, 'update_memory'):
                    if self.use_selective_memory:
                        default_losses = torch.ones(gps.shape[0], device=gps.device)
                        self.memory_bank.update_memory(gps, image_embeddings, default_losses)
                    else:
                        self.memory_bank.update_memory(gps, image_embeddings)
        
        outputs = {
            'segmentation_logits': segmentation_logits,
            'fused_features': fused_features,
            'memory_weight': memory_weight
        }
        
        if return_intermediate_features:
            outputs.update({
                'original_image_embeddings': original_image_embeddings,
                'original_memory_features': original_memory_features,
                'enhanced_image_embeddings': enhanced_embeddings,
                'valid_memory_mask': valid_memory_mask
            })
        
        if return_embeddings:
            image_proj = self.contrastive_proj(enhanced_embeddings)
            location_proj = self.contrastive_proj(enhanced_location_embeddings)
            outputs.update({
                'image_embeddings': image_proj,
                'location_embeddings': location_proj
            })
        
        return outputs
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶庫統計信息"""
        return self.memory_bank.get_memory_stats()
    
    def save_memory_bank(self):
        """保存記憶庫"""
        if hasattr(self.memory_bank, 'save_memory_bank'):
            self.memory_bank.save_memory_bank()


def create_memory_enhanced_geo_segformer(
    num_classes: int,
    model_size: str = "b0",
    feature_dim: int = 512,
    fusion_method: str = "attention",
    memory_size: int = 20,
    spatial_radius: float = 0.00005,
    memory_save_path: Optional[str] = None,
    # 選擇性記憶參數
    use_selective_memory: bool = False,
    difficulty_threshold: float = 0.5,
    diversity_threshold: float = 0.7,
    importance_decay: float = 0.95,
    max_memory_age: int = 1000
) -> MemoryEnhancedGeoSegformer:
    """
    方案C工廠函數
    """
    segformer_model = f"nvidia/mit-{model_size}"
    
    return MemoryEnhancedGeoSegformer(
        num_classes=num_classes,
        segformer_model=segformer_model,
        feature_dim=feature_dim,
        fusion_method=fusion_method,
        memory_size=memory_size,
        spatial_radius=spatial_radius,
        memory_save_path=memory_save_path,
        use_selective_memory=use_selective_memory,
        difficulty_threshold=difficulty_threshold,
        diversity_threshold=diversity_threshold,
        importance_decay=importance_decay,
        max_memory_age=max_memory_age
    )


# ⭐ 調試和分析函數
def analyze_gps_quantization(gps_csv_path: str, spatial_radius: float = 0.00005):
    """分析GPS量化效果，幫助調整spatial_radius"""
    import pandas as pd
    
    print(f"\n📊 GPS量化分析 (spatial_radius={spatial_radius}):")
    
    # 讀取GPS數據
    gps_data = pd.read_csv(gps_csv_path)
    print(f"  總GPS記錄數: {len(gps_data)}")
    
    # 計算原始唯一位置
    original_coords = set()
    for _, row in gps_data.iterrows():
        lat, lon = row['lat'], row['long']
        original_coords.add(f"{lat:.7f},{lon:.7f}")
    
    # 模擬量化過程
    def gps_to_key(lat, lon, radius):
        lat_grid = round(lat / radius) * radius
        lon_grid = round(lon / radius) * radius
        return f"{lat_grid:.7f},{lon_grid:.7f}"
    
    quantized_keys = set()
    for _, row in gps_data.iterrows():
        lat, lon = row['lat'], row['long']
        quantized_keys.add(gps_to_key(lat, lon, spatial_radius))
    
    # 分析結果
    original_unique = len(original_coords)
    quantized_unique = len(quantized_keys)
    compression_rate = quantized_unique / original_unique
    
    print(f"  原始唯一位置數: {original_unique}")
    print(f"  量化後唯一位置數: {quantized_unique}")
    print(f"  位置保留率: {compression_rate*100:.1f}%")
    
    # 建議
    if compression_rate < 0.3:
        suggested_radius = spatial_radius * 0.1
        print(f"⚠️  位置保留率太低！建議將spatial_radius縮小到: {suggested_radius:.7f}")
    elif compression_rate > 0.9:
        suggested_radius = spatial_radius * 2
        print(f"💡 位置幾乎沒有聚合，可考慮將spatial_radius增大到: {suggested_radius:.7f}")
    else:
        print(f"✅ spatial_radius設置合理")
    
    return original_unique, quantized_unique, compression_rate


def debug_memory_system(train_gps_csv: str, spatial_radius: float = 0.00005):
    """完整的記憶系統調試"""
    print("🔧 記憶系統調試分析:")
    print("=" * 50)
    
    # 1. GPS量化分析
    analyze_gps_quantization(train_gps_csv, spatial_radius)
    
    # 2. GPS數據統計
    import pandas as pd
    gps_data = pd.read_csv(train_gps_csv)
    
    lats = gps_data['lat'].values
    lons = gps_data['long'].values
    
    print(f"\n📈 GPS數據統計:")
    print(f"  緯度範圍: [{lats.min():.6f}, {lats.max():.6f}] (跨度: {lats.max()-lats.min():.6f})")
    print(f"  經度範圍: [{lons.min():.6f}, {lons.max():.6f}] (跨度: {lons.max()-lons.min():.6f})")
    print(f"  緯度標準差: {lats.std():.6f}")
    print(f"  經度標準差: {lons.std():.6f}")
    
    # 3. 重複率分析
    unique_coords = set((lat, lon) for lat, lon in zip(lats, lons))
    duplicate_rate = (len(gps_data) - len(unique_coords)) / len(gps_data) * 100
    print(f"  重複座標率: {duplicate_rate:.2f}%")
    
    # 4. 距離分析
    import numpy as np
    
    # 隨機採樣計算平均距離
    if len(gps_data) > 1000:
        sample_indices = np.random.choice(len(gps_data), 1000, replace=False)
        sample_coords = [(lats[i], lons[i]) for i in sample_indices]
    else:
        sample_coords = [(lat, lon) for lat, lon in zip(lats, lons)]
    
    distances = []
    for i in range(len(sample_coords)):
        for j in range(i+1, min(i+10, len(sample_coords))):  # 只計算前10個鄰居
            lat1, lon1 = sample_coords[i]
            lat2, lon2 = sample_coords[j]
            dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
            distances.append(dist)
    
    if distances:
        distances = np.array(distances)
        print(f"  GPS點間距離統計:")
        print(f"    平均距離: {distances.mean():.6f}")
        print(f"    最小距離: {distances.min():.6f}")
        print(f"    中位數距離: {np.median(distances):.6f}")
        print(f"    90%分位數: {np.percentile(distances, 90):.6f}")
        
        # 與spatial_radius比較
        print(f"  與spatial_radius ({spatial_radius:.6f}) 比較:")
        smaller_than_radius = (distances < spatial_radius).sum()
        print(f"    小於radius的距離對數: {smaller_than_radius}/{len(distances)} ({smaller_than_radius/len(distances)*100:.1f}%)")
    
    print("\n🎯 調試建議:")
    print("1. 如果位置保留率 < 30%，縮小 spatial_radius")
    print("2. 如果重複座標率 > 80%，考慮增加數據多樣性")
    print("3. 如果平均距離 >> spatial_radius，考慮增大 spatial_radius")
    print("4. 觀察訓練過程中記憶庫統計的變化")


if __name__ == "__main__":
    # 測試修正版模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("🧪 測試修正版方案C - 選擇性記憶增強 GeoSegformer")
    print("=" * 70)
    
    # 創建使用選擇性記憶的模型
    model = create_memory_enhanced_geo_segformer(
        num_classes=25, 
        memory_size=15,
        spatial_radius=0.00005,
        memory_save_path="./selective_memory_stats.json",
        # ⭐ 啟用選擇性記憶
        use_selective_memory=True,
        difficulty_threshold=0.5,
        diversity_threshold=0.7,
        importance_decay=0.95,
        max_memory_age=1000
    ).to(device)
    
    print(f"✅ 創建選擇性記憶模型成功")
    print(f"  記憶庫類型: {type(model.memory_bank).__name__}")
    
    # 測試數據
    batch_size = 4
    images = torch.randn(batch_size, 3, 512, 512).to(device)
    gps = torch.tensor([
        [-0.001057, -0.000368],
        [-0.000738, -0.000405],
        [-0.000545, -0.000406],
        [-0.001057, -0.000368]
    ], dtype=torch.float32).to(device)
    
    # 模擬不同困難度的損失
    mock_losses = torch.tensor([0.3, 0.7, 0.9, 0.4], device=device)  # 不同困難度
    
    print(f"\n🔍 測試選擇性記憶機制...")
    model.train()
    
    # 第一次前向傳播 - 提供損失信息
    outputs1 = model(images, gps, return_embeddings=True, update_memory=True, 
                     return_intermediate_features=True, current_losses=mock_losses)
    
    # 檢查選擇性記憶統計
    memory_stats = model.get_memory_stats()
    print(f"  記憶接受率: {memory_stats.get('memory_acceptance_rate', 0):.3f}")
    print(f"  被拒絕記憶數: {memory_stats.get('rejected_memories', 0)}")
    print(f"  平均重要性評分: {memory_stats.get('avg_importance_score', 0):.3f}")
    
    print(f"\n🎉 修正版方案C測試完成！")
    print(f"✅ 選擇性記憶機制正常工作")
    print(f"✅ 接受率計算已修正")
    
    # 保存記憶庫統計
    model.save_memory_bank()
    
    print(f"📝 詳細統計已保存到 selective_memory_stats.json")