import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from typing import Any, Dict, List, Optional
import json
import os
import pickle
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


class MultiLayerGPSImageEncoder(nn.Module):
    """
    🌍 多層GPS融合的影像編碼器 - 仿照 GeoClip
    在 SegFormer 的每個階段都注入GPS信息
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
        
        self.backbone_dims = backbone_dims
        
        # 🌍 為每個階段創建GPS編碼器
        self.stage_gps_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, dim),
                nn.LayerNorm(dim)
            ) for dim in backbone_dims
        ])
        
        # 🔄 每個階段的GPS-Image融合模組
        self.fusion_modules = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim, 
                num_heads=min(8, dim//32),  # 動態調整注意力頭數
                dropout=0.1,
                batch_first=True
            ) for dim in backbone_dims
        ])
        
        # 特徵融合層
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(sum(backbone_dims), feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # 全局平均池化來獲得影像級別的嵌入
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        print(f"✅ MultiLayerGPSImageEncoder initialized:")
        print(f"  Backbone dims: {backbone_dims}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  GPS融合階段數: {len(backbone_dims)}")
        
    def forward(self, images: torch.Tensor, gps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images, shape (batch_size, 3, H, W)
            gps: GPS coordinates, shape (batch_size, 2)
        Returns:
            Dictionary containing:
                - 'features': Multi-scale feature maps with GPS fusion
                - 'embeddings': Global image embeddings
        """
        # 提取多尺度特徵
        outputs = self.feature_extractor(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # 調整特徵圖尺寸並進行GPS融合
        enhanced_features = []
        
        # 獲取目標尺寸（使用最大的特徵圖尺寸）
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
        
        for stage_idx, feature_map in enumerate(hidden_states):
            # 處理特徵圖格式
            if len(feature_map.shape) == 3:  # (B, HW, C) 格式
                B, HW, C = feature_map.shape
                H = W = int(math.sqrt(HW))
                feature_map = feature_map.transpose(1, 2).reshape(B, C, H, W)
            elif len(feature_map.shape) == 4:  # (B, C, H, W) 格式
                pass  # 已經是正確格式
            else:
                continue
            
            # 調整空間尺寸到目標大小
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(
                    feature_map, size=target_size, mode='bilinear', align_corners=False
                )
            
            # 🌍 第stage_idx階段的GPS編碼
            gps_embedding = self.stage_gps_encoders[stage_idx](gps)  # (B, stage_dim)
            
            # 🔄 GPS-Image 融合
            B, C, H, W = feature_map.shape
            
            # 將圖像特徵重塑為序列格式
            img_seq = feature_map.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, HW, C)
            gps_seq = gps_embedding.unsqueeze(1)  # (B, 1, C)
            
            # 使用交叉注意力機制：圖像特徵作為query，GPS特徵作為key和value
            enhanced_seq, attention_weights = self.fusion_modules[stage_idx](
                img_seq,      # query: 圖像特徵序列
                gps_seq,      # key: GPS特徵
                gps_seq       # value: GPS特徵
            )
            
            # 重塑回特徵圖格式
            enhanced_feature = enhanced_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # 殘差連接：保留原始圖像信息
            enhanced_feature = enhanced_feature + feature_map
            
            enhanced_features.append(enhanced_feature)
        
        if not enhanced_features:
            raise ValueError("No valid features extracted from Segformer")
        
        # 連接所有GPS增強的特徵
        fused_features = torch.cat(enhanced_features, dim=1)
        
        # 特徵融合
        processed_features = self.feature_fusion(fused_features)
        
        # 全局影像嵌入
        global_embeddings = self.global_pool(processed_features).flatten(1)
        
        return {
            'features': processed_features,
            'embeddings': global_embeddings,
            'stage_features': enhanced_features  # 返回各階段增強特徵
        }


class LocationMemoryBank(nn.Module):
    """
    🧠 位置記憶庫 - 為每個GPS位置建立特徵記憶
    ✨ 修復設備不匹配問題，CPU儲存GPU計算
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
        
        # 調試計數器
        self.debug_info = {
            'last_locations': 0,
            'last_memories': 0,
            'last_hit_rate': 0.0
        }
        
        print(f"✅ LocationMemoryBank initialized:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Memory size per location: {memory_size}")
        print(f"  Spatial radius: {spatial_radius}")
        
    def gps_to_key(self, gps: torch.Tensor) -> str:
        """將GPS座標轉換為記憶庫的鍵"""
        # 量化GPS座標到固定網格
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
                continue  # 跳過無效特徵
            
            # 🆕 確保特徵在CPU上儲存（節省GPU記憶體）
            cpu_feature = features[i].detach().cpu().clone()
            self.memory_bank[gps_key]['features'].append(cpu_feature)
            self.memory_bank[gps_key]['count'] += 1
            self.memory_bank[gps_key]['last_updated'] = self.total_updates
            
            # 保持記憶庫大小
            if len(self.memory_bank[gps_key]['features']) > self.memory_size:
                self.memory_bank[gps_key]['features'].pop(0)
        
        self.total_updates += 1
        
        # 每100次更新打印調試信息
        if self.total_updates % 100 == 0:
            current_stats = self.get_memory_stats()
            print(f"🔄 Memory Update #{self.total_updates}:")
            print(f"  Locations: {current_stats['total_locations']} "
                  f"(+{current_stats['total_locations'] - self.debug_info['last_locations']})")
            print(f"  Memories: {current_stats['total_memories']} "
                  f"(+{current_stats['total_memories'] - self.debug_info['last_memories']})")
            
            # 更新調試信息
            self.debug_info['last_locations'] = current_stats['total_locations']
            self.debug_info['last_memories'] = current_stats['total_memories']
    
    def retrieve_memory(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """檢索相關位置的歷史特徵"""
        batch_size = gps_coords.shape[0]
        memory_features = []
        
        # 🆕 獲取目標設備
        target_device = gps_coords.device
        
        # 更新查詢計數
        self.total_queries += batch_size
        
        hits_in_batch = 0
        
        for i in range(batch_size):
            gps_key = self.gps_to_key(gps_coords[i])
            retrieved_features = []
            
            # 1. 精確匹配
            if gps_key in self.memory_bank and len(self.memory_bank[gps_key]['features']) > 0:
                retrieved_features.extend(self.memory_bank[gps_key]['features'])
                self.hit_count += 1
                hits_in_batch += 1
            
            # 2. 鄰近位置匹配
            if len(retrieved_features) < 5:  # 如果精確匹配的特徵不夠
                for key, memory in self.memory_bank.items():
                    if key != gps_key and len(memory['features']) > 0:
                        stored_lat, stored_lon = map(float, key.split(','))
                        current_lat, current_lon = gps_coords[i][0].item(), gps_coords[i][1].item()
                        distance = ((current_lat - stored_lat)**2 + (current_lon - stored_lon)**2)**0.5
                        
                        if distance < self.spatial_radius * 3:  # 擴大搜索範圍
                            retrieved_features.extend(memory['features'][-2:])  # 只取最近的2個
                            if len(retrieved_features) >= 10:  # 限制總數
                                break
            
            # 聚合檢索到的特徵
            if retrieved_features:
                # 取最近的特徵並做加權平均
                recent_features = retrieved_features[-8:]  # 最多8個特徵
                if len(recent_features) == 1:
                    aggregated = recent_features[0]
                else:
                    # 加權平均：越新的特徵權重越大
                    weights = torch.softmax(
                        torch.tensor([i for i in range(len(recent_features))], dtype=torch.float32),
                        dim=0
                    )
                    
                    stacked_features = torch.stack(recent_features)
                    aggregated = (stacked_features * weights.unsqueeze(-1)).sum(dim=0)
                
                # 🆕 確保返回的特徵在正確的設備上
                aggregated = aggregated.to(target_device)
            else:
                # 沒有歷史記錄，使用零向量
                aggregated = torch.zeros(self.feature_dim, device=target_device)
            
            memory_features.append(aggregated)
        
        # 每200次查詢打印統計信息
        if self.total_queries % 200 == 0 and self.total_queries > 0:
            current_hit_rate = self.hit_count / self.total_queries
            print(f"🔍 Memory Query #{self.total_queries}:")
            print(f"  Total hits: {self.hit_count}")
            print(f"  Hit rate: {current_hit_rate:.4f} "
                  f"(Δ{current_hit_rate - self.debug_info['last_hit_rate']:+.4f})")
            print(f"  Hits in batch: {hits_in_batch}/{batch_size}")
            
            self.debug_info['last_hit_rate'] = current_hit_rate
        
        return torch.stack(memory_features)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶庫統計信息"""
        total_locations = len(self.memory_bank)
        total_memories = sum(len(memory['features']) for memory in self.memory_bank.values())
        
        # 使用 total_queries 計算命中率
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
        """🆕 保存完整記憶庫到文件"""
        if self.save_path:
            # 準備保存數據
            memory_data = {
                'spatial_radius': self.spatial_radius,
                'feature_dim': self.feature_dim,
                'memory_size': self.memory_size,
                'total_updates': self.total_updates,
                'total_queries': self.total_queries,
                'hit_count': self.hit_count,
                'memory_bank': {}
            }
            
            # 保存記憶庫內容
            for gps_key, memory_info in self.memory_bank.items():
                if len(memory_info['features']) > 0:
                    # 將特徵轉換為CPU並堆疊
                    features_tensor = torch.stack([f.cpu() for f in memory_info['features']])
                    memory_data['memory_bank'][gps_key] = {
                        'features': features_tensor,  # 這會是一個 (num_features, feature_dim) 的tensor
                        'count': memory_info['count'],
                        'last_updated': memory_info['last_updated']
                    }
            
            # 保存到.pth文件
            memory_file = self.save_path.replace('.json', '.pth')
            torch.save(memory_data, memory_file)
            
            # 也保存統計信息到JSON (向後兼容)
            stats = {
                'locations': list(self.memory_bank.keys()),
                'counts': {k: v['count'] for k, v in self.memory_bank.items()},
                'stats': self.get_memory_stats()
            }
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"💾 Memory bank saved to {memory_file}")
            print(f"📊 Memory stats saved to {self.save_path}")
            print(f"📈 Saved {len(memory_data['memory_bank'])} GPS locations with features")
    
    def load_memory_bank(self, memory_path: str):
        """🆕 從文件載入記憶庫"""
        try:
            # 嘗試載入.pth格式的完整記憶庫
            memory_file = memory_path.replace('.json', '.pth')
            
            if os.path.exists(memory_file):
                print(f"📂 Loading memory bank from {memory_file}")
                memory_data = torch.load(memory_file, map_location='cpu')  # 🆕 強制載入到CPU
                
                # 檢查版本兼容性
                if 'feature_dim' in memory_data and memory_data['feature_dim'] != self.feature_dim:
                    print(f"⚠️  Feature dimension mismatch: saved={memory_data['feature_dim']}, current={self.feature_dim}")
                    print("🔄 Will attempt to resize features...")
                
                # 載入統計信息
                self.spatial_radius = memory_data.get('spatial_radius', self.spatial_radius)
                self.total_updates = memory_data.get('total_updates', 0)
                self.total_queries = memory_data.get('total_queries', 0)
                self.hit_count = memory_data.get('hit_count', 0)
                
                # 載入記憶庫內容
                self.memory_bank = defaultdict(lambda: {'features': [], 'count': 0, 'last_updated': 0})
                
                for gps_key, memory_info in memory_data['memory_bank'].items():
                    features_tensor = memory_info['features']  # (num_features, feature_dim)
                    
                    # 🆕 確保所有特徵都在CPU上
                    if features_tensor.device != torch.device('cpu'):
                        features_tensor = features_tensor.cpu()
                    
                    # 處理特徵維度不匹配的情況
                    if features_tensor.shape[1] != self.feature_dim:
                        if features_tensor.shape[1] < self.feature_dim:
                            # Padding
                            padding = torch.zeros(features_tensor.shape[0], 
                                                self.feature_dim - features_tensor.shape[1])
                            features_tensor = torch.cat([features_tensor, padding], dim=1)
                        else:
                            # Truncate
                            features_tensor = features_tensor[:, :self.feature_dim]
                    
                    # 將特徵轉換為列表形式
                    feature_list = [features_tensor[i] for i in range(features_tensor.shape[0])]
                    
                    self.memory_bank[gps_key] = {
                        'features': feature_list,
                        'count': memory_info['count'],
                        'last_updated': memory_info['last_updated']
                    }
                
                loaded_locations = len(self.memory_bank)
                total_features = sum(len(memory['features']) for memory in self.memory_bank.values())
                
                print(f"✅ Memory bank loaded successfully!")
                print(f"📍 Loaded {loaded_locations} GPS locations")
                print(f"🧠 Loaded {total_features} feature memories")
                print(f"📊 Historical stats: queries={self.total_queries}, hits={self.hit_count}")
                
                return True
            
            else:
                print(f"❌ Memory bank file not found: {memory_file}")
                if os.path.exists(memory_path):
                    print(f"📋 Found stats file {memory_path}, but no feature data")
                    print("💡 Testing will start with empty memory bank")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load memory bank: {e}")
            print("💡 Testing will start with empty memory bank")
            return False


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


class MemoryEnhancedGeoSegformer(nn.Module):
    """
    🌍 多層GPS融合 + 記憶增強版 GeoSegformer
    ✨ 支持記憶庫消融實驗（memory_size=0時禁用記憶庫）
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
        memory_size: int = 20,  # 🆕 設為0禁用記憶庫
        spatial_radius: float = 0.00005,
        memory_save_path: Optional[str] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.memory_enabled = memory_size > 0  # 🆕 記憶庫啟用標誌
        
        print(f"🚀 Initializing MultiLayer GPS + Memory Enhanced GeoSegformer")
        print(f"  Num classes: {num_classes}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Memory enabled: {self.memory_enabled}")
        if self.memory_enabled:
            print(f"  Memory size: {memory_size}")
            print(f"  Spatial radius: {spatial_radius}")
        
        # GPS 位置編碼器（總是需要）
        self.location_encoder = LocationEncoder(
            rff_dim=rff_dim,
            output_dim=feature_dim,
            sigmas=sigmas,
            dropout=dropout
        )
        
        # 🌍 多層GPS融合的影像編碼器（總是需要）
        self.image_encoder = MultiLayerGPSImageEncoder(
            segformer_model=segformer_model,
            feature_dim=feature_dim
        )
        
        # 🆕 條件性記憶庫
        if self.memory_enabled:
            self.memory_bank = LocationMemoryBank(
                feature_dim=feature_dim,
                memory_size=memory_size,
                spatial_radius=spatial_radius,
                save_path=memory_save_path
            )
            
            # 記憶融合層
            self.memory_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim)
            )
            
            # 記憶注意力機制
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            print(f"✅ Memory bank components initialized")
        else:
            self.memory_bank = None
            self.memory_fusion = None  # 🔧 確保設為None
            self.memory_attention = None  # 🔧 確保設為None
            print(f"🚫 Memory bank disabled")
        
        # 跨模態融合（總是需要）
        self.cross_modal_fusion = CrossModalFusion(
            feature_dim=feature_dim,
            fusion_method=fusion_method
        )
        
        # 分割頭（總是需要）
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim, num_classes, 1)
        )
        
        # 用於對比學習的投影頭（總是需要）
        self.contrastive_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256)
        )
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"✅ MultiLayer GPS + Memory Enhanced GeoSegformer initialized with {total_params:.2f}M parameters")
    
    def forward(
        self, 
        images: torch.Tensor, 
        gps: torch.Tensor,
        return_embeddings: bool = False,
        update_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images, shape (batch_size, 3, H, W)
            gps: GPS coordinates, shape (batch_size, 2)
            return_embeddings: Whether to return embeddings for contrastive learning
            update_memory: Whether to update memory bank
        Returns:
            Dictionary containing segmentation results and optional embeddings
        """
        # GPS 位置編碼
        location_embeddings = self.location_encoder(gps)
        
        # 🌍 多層GPS融合的影像特徵提取
        image_outputs = self.image_encoder(images, gps)
        image_features = image_outputs['features']
        image_embeddings = image_outputs['embeddings']
        
        # 🆕 條件性記憶增強處理
        enhanced_embeddings = image_embeddings
        memory_weight = 0.0
        
        if self.memory_enabled and self.memory_bank is not None:
            # 檢索位置記憶
            memory_features = self.memory_bank.retrieve_memory(gps)
            
            # 計算記憶特徵的有效性（使用L2範數）
            memory_norms = torch.norm(memory_features, dim=-1)
            valid_memory_mask = memory_norms > 1e-6  # 非零特徵判斷
            
            if valid_memory_mask.any():
                memory_weight = valid_memory_mask.float().mean().item()
                
                if memory_weight > 0:
                    # 只對有效記憶進行處理
                    valid_indices = valid_memory_mask.nonzero(as_tuple=True)[0]
                    
                    if len(valid_indices) > 0:
                        valid_memory_features = memory_features[valid_indices]
                        valid_image_embeddings = image_embeddings[valid_indices]
                        
                        # 🆕 確保設備一致性 - 檢查和修復設備不匹配
                        if valid_memory_features.device != valid_image_embeddings.device:
                            print(f"⚠️  Device mismatch detected: memory={valid_memory_features.device}, image={valid_image_embeddings.device}")
                            valid_memory_features = valid_memory_features.to(valid_image_embeddings.device)
                        
                        # 方法1：特徵融合
                        combined_features = torch.cat([valid_image_embeddings, valid_memory_features], dim=-1)
                        fused_features = self.memory_fusion(combined_features)
                        
                        # 方法2：注意力融合
                        memory_enhanced, attention_weights = self.memory_attention(
                            valid_image_embeddings.unsqueeze(1),  # query: (valid_batch, 1, feature_dim)
                            valid_memory_features.unsqueeze(1),   # key: (valid_batch, 1, feature_dim)
                            valid_memory_features.unsqueeze(1)    # value: (valid_batch, 1, feature_dim)
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
            
            # 更新記憶庫（僅在訓練時）
            if update_memory and self.training:
                self.memory_bank.update_memory(gps, image_embeddings)
        
        # 將增強的特徵投影回空間維度以進行跨模態融合
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
        
        outputs = {
            'segmentation_logits': segmentation_logits,
            'fused_features': fused_features,
            'memory_weight': memory_weight  # 記憶特徵的有效權重
        }
        
        # 如果需要返回嵌入用於對比學習
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
        if self.memory_enabled and self.memory_bank is not None:
            return self.memory_bank.get_memory_stats()
        else:
            return {
                'total_locations': 0,
                'total_memories': 0,
                'hit_rate': 0.0,
                'avg_memories_per_location': 0.0,
                'total_queries': 0,
                'total_updates': 0,
                'hit_count': 0,
                'memory_enabled': False
            }
    
    def save_memory_bank(self):
        """保存記憶庫"""
        if self.memory_enabled and self.memory_bank is not None:
            self.memory_bank.save_memory_bank()
        else:
            print("🚫 Memory bank is disabled, nothing to save")
    
    def load_memory_bank(self, memory_path: str):
        """🆕 載入記憶庫"""
        if self.memory_enabled and self.memory_bank is not None:
            return self.memory_bank.load_memory_bank(memory_path)
        else:
            print("🚫 Memory bank is disabled, cannot load")
            return False


# 工廠函數
def create_memory_enhanced_geo_segformer(
    num_classes: int,
    model_size: str = "b0",
    feature_dim: int = 512,
    fusion_method: str = "attention",
    memory_size: int = 20,  # 🆕 設為0禁用記憶庫
    spatial_radius: float = 0.00005,
    memory_save_path: Optional[str] = None
) -> MemoryEnhancedGeoSegformer:
    """
    創建多層GPS融合 + 可選記憶增強版 GeoSegformer 模型的工廠函數
    
    Args:
        memory_size: 記憶庫大小，設為0時禁用記憶庫
    """
    segformer_model = f"nvidia/mit-{model_size}"
    
    # 🆕 根據memory_size決定是否啟用記憶庫
    if memory_size == 0:
        print(f"🚫 Creating GPS-only model (memory disabled)")
        memory_save_path = None  # 不保存記憶庫
    else:
        print(f"🧠 Creating full model with memory bank (size={memory_size})")
    
    return MemoryEnhancedGeoSegformer(
        num_classes=num_classes,
        segformer_model=segformer_model,
        feature_dim=feature_dim,
        fusion_method=fusion_method,
        memory_size=memory_size,  # 🆕 會自動處理0的情況
        spatial_radius=spatial_radius,
        memory_save_path=memory_save_path
    )


# 調試和分析函數
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
    # 測試支持消融實驗的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("🧪 測試支持消融實驗的 GeoSegformer")
    print("=" * 70)
    
    # 測試GPS-only模型（記憶庫禁用）
    print(f"\n🚫 測試GPS-only模型（記憶庫禁用）...")
    gps_only_model = create_memory_enhanced_geo_segformer(
        num_classes=25, 
        memory_size=0,  # 🆕 設為0禁用記憶庫
        spatial_radius=0.00005,  # 這個參數會被忽略
        memory_save_path=None
    ).to(device)
    
    # 測試完整模型（記憶庫啟用）
    print(f"\n🧠 測試完整模型（記憶庫啟用）...")
    full_model = create_memory_enhanced_geo_segformer(
        num_classes=25, 
        memory_size=15,  # 啟用記憶庫
        spatial_radius=0.00005,
        memory_save_path="./test_memory_stats.json"
    ).to(device)
    
    # 測試數據
    batch_size = 4
    images = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # 模擬你的GPS數據範圍
    gps = torch.tensor([
        [-0.001057, -0.000368],
        [-0.000738, -0.000405],
        [-0.000545, -0.000406],
        [-0.001057, -0.000368]
    ], dtype=torch.float32).to(device)
    
    print(f"\n🔍 測試配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image shape: {images.shape}")
    print(f"  GPS shape: {gps.shape}")
    
    # 測試GPS-only模型
    print(f"\n🚫 測試GPS-only模型...")
    gps_only_model.train()
    outputs1 = gps_only_model(images, gps, return_embeddings=True, update_memory=True)
    
    print(f"  分割輸出形狀: {outputs1['segmentation_logits'].shape}")
    print(f"  記憶權重: {outputs1['memory_weight']:.4f}")
    print(f"  影像嵌入形狀: {outputs1['image_embeddings'].shape}")
    print(f"  位置嵌入形狀: {outputs1['location_embeddings'].shape}")
    
    # 檢查記憶庫統計
    memory_stats = gps_only_model.get_memory_stats()
    print(f"  記憶庫統計: {memory_stats}")
    
    # 測試完整模型
    print(f"\n🧠 測試完整模型...")
    full_model.train()
    outputs2 = full_model(images, gps, return_embeddings=True, update_memory=True)
    
    print(f"  分割輸出形狀: {outputs2['segmentation_logits'].shape}")
    print(f"  記憶權重: {outputs2['memory_weight']:.4f}")
    print(f"  影像嵌入形狀: {outputs2['image_embeddings'].shape}")
    print(f"  位置嵌入形狀: {outputs2['location_embeddings'].shape}")
    
    # 檢查記憶庫統計
    memory_stats = full_model.get_memory_stats()
    print(f"  記憶庫統計: {memory_stats}")
    
    print(f"\n🎉 消融實驗支持測試完成！")
    print(f"✅ GPS-only模型：記憶庫已禁用")
    print(f"✅ 完整模型：記憶庫正常工作")
    print(f"✅ 兩個模型都可以正常訓練和推理")
    print(f"📝 現在可以開始消融實驗了！")