#!/usr/bin/env python3
"""
Complete GPS Parameter Analyzer for GeoSegformer
自動分析GPS數據並推薦最適合的GeoSegformer參數
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple
import json
from datetime import datetime
import sys
import os

class GPSParameterAnalyzer:
    def __init__(self, train_gps_csv: str, val_gps_csv: str):
        """
        GPS數據分析器，自動推薦GeoSegformer參數
        
        Args:
            train_gps_csv: 訓練集GPS CSV檔案路徑
            val_gps_csv: 驗證集GPS CSV檔案路徑
        """
        self.train_gps_csv = train_gps_csv
        self.val_gps_csv = val_gps_csv
        self.train_data = None
        self.val_data = None
        self.all_data = None
        self.analysis_results = {}
        
    def load_data(self):
        """載入GPS數據"""
        print("📂 Loading GPS data...")
        
        try:
            self.train_data = pd.read_csv(self.train_gps_csv)
            self.val_data = pd.read_csv(self.val_gps_csv)
            
            print(f"✅ Training data: {len(self.train_data)} samples")
            print(f"✅ Validation data: {len(self.val_data)} samples")
            
            # 檢查必要欄位
            required_cols = ['filename', 'lat', 'long']
            for df_name, df in [('train', self.train_data), ('val', self.val_data)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in {df_name} data: {missing_cols}")
            
            # 合併數據用於全局分析
            self.all_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
            print(f"✅ Total data: {len(self.all_data)} samples")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_spatial_distribution(self):
        """分析空間分布"""
        print("\n🗺️  Analyzing spatial distribution...")
        
        # 基本統計
        lat_stats = {
            'min': float(self.all_data['lat'].min()),
            'max': float(self.all_data['lat'].max()),
            'mean': float(self.all_data['lat'].mean()),
            'std': float(self.all_data['lat'].std()),
            'range': float(self.all_data['lat'].max() - self.all_data['lat'].min())
        }
        
        long_stats = {
            'min': float(self.all_data['long'].min()),
            'max': float(self.all_data['long'].max()),
            'mean': float(self.all_data['long'].mean()),
            'std': float(self.all_data['long'].std()),
            'range': float(self.all_data['long'].max() - self.all_data['long'].min())
        }
        
        print(f"📊 Latitude:  [{lat_stats['min']:.6f}, {lat_stats['max']:.6f}] (range: {lat_stats['range']:.6f})")
        print(f"📊 Longitude: [{long_stats['min']:.6f}, {long_stats['max']:.6f}] (range: {long_stats['range']:.6f})")
        
        # 計算空間密度
        spatial_area = lat_stats['range'] * long_stats['range']
        point_density = len(self.all_data) / spatial_area if spatial_area > 0 else 0
        
        self.analysis_results['spatial'] = {
            'lat_stats': lat_stats,
            'long_stats': long_stats,
            'spatial_area': float(spatial_area),
            'point_density': float(point_density)
        }
        
        return lat_stats, long_stats, spatial_area, point_density
    
    def analyze_location_clusters(self):
        """分析位置聚類"""
        print("\n🎯 Analyzing location clusters...")
        
        # 使用不同精度分析重複位置
        precisions = [4, 5, 6, 7]  # 小數點後位數
        cluster_analysis = {}
        
        for precision in precisions:
            # 四捨五入到指定精度
            rounded_coords = self.all_data[['lat', 'long']].round(precision)
            coord_strings = rounded_coords.apply(lambda x: f"{x['lat']:.{precision}f},{x['long']:.{precision}f}", axis=1)
            
            # 統計每個位置的圖像數量
            location_counts = coord_strings.value_counts()
            
            cluster_analysis[precision] = {
                'unique_locations': int(len(location_counts)),
                'avg_images_per_location': float(location_counts.mean()),
                'max_images_per_location': int(location_counts.max()),
                'locations_with_multiple_images': int((location_counts > 1).sum()),
                'duplicate_rate': float((location_counts > 1).sum() / len(location_counts) * 100)
            }
            
            print(f"📍 Precision {precision}: {cluster_analysis[precision]['unique_locations']} unique locations, "
                  f"avg {cluster_analysis[precision]['avg_images_per_location']:.1f} images/location, "
                  f"duplicate rate {cluster_analysis[precision]['duplicate_rate']:.1f}%")
        
        # 選擇最佳精度（平衡唯一性和實用性）
        best_precision = self.select_best_precision(cluster_analysis)
        
        self.analysis_results['clusters'] = {
            'precision_analysis': cluster_analysis,
            'best_precision': best_precision,
            'recommended_stats': cluster_analysis[best_precision]
        }
        
        return cluster_analysis, best_precision
    
    def select_best_precision(self, cluster_analysis: Dict) -> int:
        """選擇最佳精度"""
        # 目標：找到既有足夠唯一位置，又有合理重複率的精度
        scores = {}
        
        for precision, stats in cluster_analysis.items():
            # 評分標準：
            # 1. 唯一位置數量適中（不要太多也不要太少）
            # 2. 平均每位置圖像數在2-10之間較好
            # 3. 重複率在10-50%之間較好
            
            unique_score = min(stats['unique_locations'] / 100, 10)  # 不要太多位置
            avg_images_score = max(0, 10 - abs(stats['avg_images_per_location'] - 5))  # 理想是5張/位置
            duplicate_score = max(0, 10 - abs(stats['duplicate_rate'] - 30))  # 理想是30%重複率
            
            scores[precision] = unique_score + avg_images_score + duplicate_score
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def calculate_distance_statistics(self, precision: int):
        """計算距離統計"""
        print(f"\n📏 Calculating distance statistics (precision {precision})...")
        
        # 使用選定精度的坐標
        coords = self.all_data[['lat', 'long']].round(precision)
        unique_coords = coords.drop_duplicates().values
        
        if len(unique_coords) < 2:
            print("⚠️  Not enough unique coordinates for distance analysis")
            return None
        
        # 計算所有點對之間的距離（限制數量避免計算過久）
        distances = []
        max_pairs = min(10000, len(unique_coords) * (len(unique_coords) - 1) // 2)
        
        for i in range(len(unique_coords)):
            for j in range(i + 1, len(unique_coords)):
                if len(distances) >= max_pairs:
                    break
                    
                lat1, lon1 = unique_coords[i]
                lat2, lon2 = unique_coords[j]
                
                # 計算歐氏距離（適合小範圍）
                dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                distances.append(dist)
            
            if len(distances) >= max_pairs:
                break
        
        distances = np.array(distances)
        
        distance_stats = {
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'mean_distance': float(distances.mean()),
            'median_distance': float(np.median(distances)),
            'std_distance': float(distances.std()),
            'percentiles': {
                'p10': float(np.percentile(distances, 10)),
                'p25': float(np.percentile(distances, 25)),
                'p75': float(np.percentile(distances, 75)),
                'p90': float(np.percentile(distances, 90))
            }
        }
        
        print(f"📏 Distance statistics:")
        print(f"   Min: {distance_stats['min_distance']:.6f}")
        print(f"   Mean: {distance_stats['mean_distance']:.6f}")
        print(f"   Median: {distance_stats['median_distance']:.6f}")
        print(f"   P25: {distance_stats['percentiles']['p25']:.6f}")
        print(f"   P75: {distance_stats['percentiles']['p75']:.6f}")
        
        self.analysis_results['distances'] = distance_stats
        
        return distance_stats
    
    def recommend_parameters(self):
        """推薦GeoSegformer參數"""
        print("\n🎯 Recommending GeoSegformer parameters...")
        
        # 獲取分析結果
        spatial = self.analysis_results['spatial']
        clusters = self.analysis_results['clusters']
        distances = self.analysis_results.get('distances')
        
        recommended_stats = clusters['recommended_stats']
        
        # 1. 空間半徑 (spatial_radius)
        if distances:
            # 基於距離統計推薦
            spatial_radius = max(
                distances['percentiles']['p10'] * 0.5,  # 不要太小
                distances['mean_distance'] * 0.05,      # 平均距離的5%
                0.00001  # 最小值
            )
            spatial_radius = min(spatial_radius, 0.001)  # 最大值
        else:
            # 基於空間範圍推薦
            avg_range = (spatial['lat_stats']['range'] + spatial['long_stats']['range']) / 2
            spatial_radius = avg_range / (recommended_stats['unique_locations'] ** 0.5) * 0.1
        
        # 2. 空間閾值 (spatial_threshold)
        spatial_threshold = spatial_radius * 3
        
        # 3. 記憶庫大小 (memory_size)
        avg_imgs_per_location = recommended_stats['avg_images_per_location']
        if avg_imgs_per_location < 2:
            memory_size = 10
        elif avg_imgs_per_location < 5:
            memory_size = 15
        elif avg_imgs_per_location < 10:
            memory_size = 20
        else:
            memory_size = min(30, int(avg_imgs_per_location * 1.5))
        
        # 4. 對比學習權重 (contrastive_weight)
        duplicate_rate = recommended_stats['duplicate_rate']
        if duplicate_rate < 10:
            contrastive_weight = 0.01
        elif duplicate_rate < 30:
            contrastive_weight = 0.05
        elif duplicate_rate < 50:
            contrastive_weight = 0.03
        else:
            contrastive_weight = 0.02
        
        # 5. 記憶庫預熱輪數 (memory_warmup_epochs)
        total_samples = len(self.all_data)
        if total_samples < 1000:
            memory_warmup_epochs = 2
        elif total_samples < 5000:
            memory_warmup_epochs = 3
        else:
            memory_warmup_epochs = 5
        
        # 6. 模型大小 (model_size)
        if total_samples < 2000:
            model_size = "b0"
            feature_dim = 256
        elif total_samples < 8000:
            model_size = "b0"
            feature_dim = 512
        else:
            model_size = "b1"
            feature_dim = 512
        
        # 7. 融合方法 (fusion_method)
        if total_samples < 3000:
            fusion_method = "concat"  # 較快
        else:
            fusion_method = "attention"  # 較好效果
        
        # 8. 學習率推薦
        lr_scale = 1.0
        if total_samples < 2000:
            lr_scale = 0.7  # 小數據集用較小學習率
        elif total_samples > 10000:
            lr_scale = 1.3  # 大數據集可以用較大學習率
        
        recommendations = {
            'spatial_radius': float(round(spatial_radius, 6)),
            'spatial_threshold': float(round(spatial_threshold, 6)),
            'memory_size': int(memory_size),
            'contrastive_weight': float(contrastive_weight),
            'memory_warmup_epochs': int(memory_warmup_epochs),
            'model_size': model_size,
            'feature_dim': int(feature_dim),
            'fusion_method': fusion_method,
            'lr_backbone': float(round(6e-5 * lr_scale, 6)),
            'lr_head': float(round(6e-4 * lr_scale, 6)),
            'lr_gps': float(round(3e-4 * lr_scale, 6)),
            'lr_memory': float(round(6e-4 * lr_scale, 6)),
            'lr_fusion': float(round(6e-4 * lr_scale, 6)),
            'gps_norm_method': 'minmax',
            'seg_weight': float(1.0),
            'temperature': float(0.07)
        }
        
        self.analysis_results['recommendations'] = recommendations
        
        return recommendations
    
    def print_recommendations(self):
        """打印推薦結果"""
        recommendations = self.analysis_results['recommendations']
        
        print(f"\n🎯 RECOMMENDED GEOSEGFORMER PARAMETERS")
        print("=" * 60)
        
        print(f"\n🏗️  Model Architecture:")
        print(f"--model-size {recommendations['model_size']}")
        print(f"--feature-dim {recommendations['feature_dim']}")
        print(f"--fusion-method {recommendations['fusion_method']}")
        
        print(f"\n🗺️  Spatial Parameters:")
        print(f"--spatial-radius {recommendations['spatial_radius']}")
        print(f"--spatial-threshold {recommendations['spatial_threshold']}")
        print(f"--gps-norm-method {recommendations['gps_norm_method']}")
        
        print(f"\n🧠 Memory Parameters:")
        print(f"--memory-size {recommendations['memory_size']}")
        print(f"--memory-warmup-epochs {recommendations['memory_warmup_epochs']}")
        
        print(f"\n⚖️  Loss Parameters:")
        print(f"--seg-weight {recommendations['seg_weight']}")
        print(f"--contrastive-weight {recommendations['contrastive_weight']}")
        print(f"--temperature {recommendations['temperature']}")
        
        print(f"\n🎓 Learning Rate Parameters:")
        print(f"--lr-backbone {recommendations['lr_backbone']}")
        print(f"--lr-head {recommendations['lr_head']}")
        print(f"--lr-gps {recommendations['lr_gps']}")
        print(f"--lr-memory {recommendations['lr_memory']}")
        print(f"--lr-fusion {recommendations['lr_fusion']}")
        
        print(f"\n📋 Complete Command Line:")
        print("=" * 60)
        cmd_parts = []
        for key, value in recommendations.items():
            if key in ['gps_norm_method', 'model_size', 'fusion_method']:
                cmd_parts.append(f"--{key.replace('_', '-')} {value}")
            else:
                cmd_parts.append(f"--{key.replace('_', '-')} {value}")
        
        print(" \\\n  ".join(cmd_parts))
    
    def save_analysis_report(self, output_path: str = "gps_analysis_report.json"):
        """儲存分析報告"""
        
        def convert_numpy_types(obj):
            """將numpy類型轉換為Python原生類型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'input_files': {
                'train_gps_csv': self.train_gps_csv,
                'val_gps_csv': self.val_gps_csv
            },
            'data_summary': {
                'train_samples': len(self.train_data),
                'val_samples': len(self.val_data),
                'total_samples': len(self.all_data)
            },
            'analysis_results': convert_numpy_types(self.analysis_results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analysis report saved to: {output_path}")
    
    def create_visualization(self, output_dir: str = "gps_analysis_plots"):
        """創建視覺化圖表"""
        print(f"\n📊 Creating visualizations...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        try:
            # 設定圖表風格
            plt.style.use('default')
            
            # 創建圖表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. GPS點分布圖
            axes[0, 0].scatter(self.train_data['long'], self.train_data['lat'], 
                           alpha=0.6, s=30, c='blue', label='Training')
            axes[0, 0].scatter(self.val_data['long'], self.val_data['lat'], 
                           alpha=0.6, s=30, c='red', label='Validation')
            axes[0, 0].set_xlabel('Longitude')
            axes[0, 0].set_ylabel('Latitude')
            axes[0, 0].set_title('GPS Points Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 距離分布直方圖
            if 'distances' in self.analysis_results:
                distances = []
                coords = self.all_data[['lat', 'long']].values
                for i in range(min(100, len(coords))):
                    for j in range(i + 1, min(i + 20, len(coords))):
                        lat1, lon1 = coords[i]
                        lat2, lon2 = coords[j]
                        dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                        distances.append(dist)
                
                axes[0, 1].hist(distances, bins=30, alpha=0.7, color='green')
                axes[0, 1].set_xlabel('Distance')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Pairwise Distance Distribution')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 每個位置的圖像數量分布
            coords = self.all_data[['lat', 'long']].round(5)
            coord_strings = coords.apply(lambda x: f"{x['lat']:.5f},{x['long']:.5f}", axis=1)
            location_counts = coord_strings.value_counts()
            
            axes[1, 0].hist(location_counts, bins=min(20, len(location_counts)//2), 
                    alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Images per Location')
            axes[1, 0].set_ylabel('Number of Locations')
            axes[1, 0].set_title('Images per Location Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 推薦參數總結
            axes[1, 1].axis('off')
            
            recommendations = self.analysis_results['recommendations']
            text_content = f"""Recommended Parameters:

Spatial Radius: {recommendations['spatial_radius']}
Memory Size: {recommendations['memory_size']}
Contrastive Weight: {recommendations['contrastive_weight']}
Model Size: {recommendations['model_size']}
Feature Dim: {recommendations['feature_dim']}
Fusion Method: {recommendations['fusion_method']}

Data Summary:
Total Samples: {len(self.all_data)}
Unique Locations: {self.analysis_results['clusters']['recommended_stats']['unique_locations']}
Avg Images/Location: {self.analysis_results['clusters']['recommended_stats']['avg_images_per_location']:.1f}
Duplicate Rate: {self.analysis_results['clusters']['recommended_stats']['duplicate_rate']:.1f}%
            """
            
            axes[1, 1].text(0.05, 0.95, text_content, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/gps_analysis_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved to: {output_dir}/gps_analysis_summary.png")
            
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
    
    def run_analysis(self, save_report: bool = True, create_plots: bool = True):
        """運行完整分析"""
        print("🚀 Starting GPS Parameter Analysis...")
        print("=" * 60)
        
        # 載入數據
        self.load_data()
        
        # 分析空間分布
        self.analyze_spatial_distribution()
        
        # 分析位置聚類
        _, best_precision = self.analyze_location_clusters()
        
        # 計算距離統計
        self.calculate_distance_statistics(best_precision)
        
        # 推薦參數
        self.recommend_parameters()
        
        # 打印推薦結果
        self.print_recommendations()
        
        # 保存報告
        if save_report:
            self.save_analysis_report()
        
        # 創建視覺化
        if create_plots:
            self.create_visualization()
        
        print(f"\n✅ Analysis completed!")
        return self.analysis_results['recommendations']


def check_gps_csv_format(csv_path: str):
    """檢查GPS CSV格式"""
    print(f"\n🔍 Checking GPS CSV format: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ File loaded successfully: {len(df)} rows")
        
        # 檢查欄位
        print(f"📋 Columns: {list(df.columns)}")
        
        required_cols = ['filename', 'lat', 'long']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print("Required columns: filename, lat, long")
            return False
        else:
            print("✅ All required columns present")
        
        # 檢查數據範圍
        print(f"📊 Latitude range: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
        print(f"📊 Longitude range: [{df['long'].min():.6f}, {df['long'].max():.6f}]")
        
        # 檢查是否有缺失值
        if df[required_cols].isnull().any().any():
            print("⚠️  Warning: Found missing values in GPS data")
            print(df[required_cols].isnull().sum())
        else:
            print("✅ No missing values in GPS data")
        
        # 顯示前幾行
        print("\n📋 First 5 rows:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def main():
    """主程式"""
    print("🚀 GPS Parameter Analyzer for GeoSegformer")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        print("Usage options:")
        print("1. python gps_analyzer.py check <gps_csv_file>")
        print("2. python gps_analyzer.py analyze <train_gps_csv> <val_gps_csv>")
        print()
        print("Example:")
        print("python gps_analyzer.py analyze data/train_gps.csv data/val_gps.csv")
        return
        
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        # 檢查單個GPS檔案格式
        csv_file = sys.argv[2]
        check_gps_csv_format(csv_file)
        
    elif len(sys.argv) == 4 and sys.argv[1] == "analyze":
        # 分析指定的GPS檔案
        train_gps_csv = sys.argv[2]
        val_gps_csv = sys.argv[3]
        
        print(f"📂 Training GPS: {train_gps_csv}")
        print(f"📂 Validation GPS: {val_gps_csv}")
        
        # 檢查檔案存在性
        if not os.path.exists(train_gps_csv):
            print(f"❌ Training GPS file not found: {train_gps_csv}")
            sys.exit(1)
            
        if not os.path.exists(val_gps_csv):
            print(f"❌ Validation GPS file not found: {val_gps_csv}")
            sys.exit(1)
        
        # 先檢查格式
        print("\n🔍 Checking file formats...")
        if not check_gps_csv_format(train_gps_csv):
            print("❌ Training GPS file format error")
            sys.exit(1)
            
        if not check_gps_csv_format(val_gps_csv):
            print("❌ Validation GPS file format error")
            sys.exit(1)
        
        # 運行分析
        print("\n🚀 Starting parameter analysis...")
        analyzer = GPSParameterAnalyzer(train_gps_csv, val_gps_csv)
        
        try:
            recommendations = analyzer.run_analysis(
                save_report=True,
                create_plots=True
            )
            
            print("\n🎉 Analysis completed successfully!")
            
            # 生成可以直接使用的命令
            print("\n📋 Ready-to-use training command:")
            print("=" * 60)
            
            # 生成完整的訓練命令
            cmd = f"""python universal_enhanced_train.py \\
  [train_img_dir] [train_ann_dir] [val_img_dir] [val_ann_dir] [category_csv] [max_epochs] [logdir] \\
  --use-geo \\
  --train-gps-csv {train_gps_csv} \\
  --val-gps-csv {val_gps_csv} \\
  --model-size {recommendations['model_size']} \\
  --feature-dim {recommendations['feature_dim']} \\
  --fusion-method {recommendations['fusion_method']} \\
  --spatial-radius {recommendations['spatial_radius']} \\
  --spatial-threshold {recommendations['spatial_threshold']} \\
  --memory-size {recommendations['memory_size']} \\
  --memory-warmup-epochs {recommendations['memory_warmup_epochs']} \\
  --contrastive-weight {recommendations['contrastive_weight']} \\
  --lr-backbone {recommendations['lr_backbone']} \\
  --lr-head {recommendations['lr_head']} \\
  --lr-gps {recommendations['lr_gps']} \\
  --lr-memory {recommendations['lr_memory']} \\
  --lr-fusion {recommendations['lr_fusion']} \\
  --early-stop --patience 50 --batch-size 8"""
            
            print(cmd)
            
            # 保存命令到檔案
            with open("recommended_geosegformer_command.txt", "w") as f:
                f.write("# Recommended GeoSegformer Training Command\n")
                f.write("# Generated by GPS Parameter Analyzer\n\n")
                f.write(cmd)
                f.write("\n\n# Please replace the following placeholders:\n")
                f.write("# [train_img_dir] - path to training images\n")
                f.write("# [train_ann_dir] - path to training annotations\n")
                f.write("# [val_img_dir] - path to validation images\n")
                f.write("# [val_ann_dir] - path to validation annotations\n")
                f.write("# [category_csv] - path to category definition CSV\n")
                f.write("# [max_epochs] - maximum training epochs (e.g., 1000)\n")
                f.write("# [logdir] - output directory for logs and models\n")
                
            print(f"\n💾 Command saved to: recommended_geosegformer_command.txt")
            
            # 額外的建議
            print(f"\n💡 Additional suggestions:")
            print(f"   - Start with a small test run (e.g., 10 epochs) to validate parameters")
            print(f"   - Monitor 'Memory Hit Rate' during training (should be 0.3-0.8)")
            print(f"   - If hit rate is too low, increase --memory-size")
            print(f"   - If hit rate is too high, decrease --spatial-radius")
            print(f"   - Consider using --multi-seed for robust results")
            
            # 根據數據特徵給出特殊建議
            total_samples = len(analyzer.all_data)
            unique_locations = recommendations['memory_size']
            duplicate_rate = analyzer.analysis_results['clusters']['recommended_stats']['duplicate_rate']
            
            print(f"\n📊 Dataset Analysis Summary:")
            print(f"   Dataset size: {total_samples} images")
            print(f"   Unique locations: {analyzer.analysis_results['clusters']['recommended_stats']['unique_locations']}")
            print(f"   Duplicate rate: {duplicate_rate:.1f}%")
            print(f"   Avg images per location: {analyzer.analysis_results['clusters']['recommended_stats']['avg_images_per_location']:.1f}")
            
            # 根據數據特徵給出個性化建議
            print(f"\n🎯 Personalized Recommendations:")
            if duplicate_rate < 10:
                print("   - Your dataset has low location overlap")
                print("   - Consider using lower contrastive weight")
                print("   - Focus on spatial accuracy rather than memory efficiency")
            elif duplicate_rate > 50:
                print("   - Your dataset has high location overlap")
                print("   - Memory system will be very effective")
                print("   - Consider increasing memory size for better performance")
            
            if total_samples < 2000:
                print("   - Small dataset: use conservative parameters")
                print("   - Consider data augmentation")
                print("   - Start with lower learning rates")
            elif total_samples > 10000:
                print("   - Large dataset: can use more aggressive parameters")
                print("   - Consider using larger model (b1 or b2)")
                print("   - Multi-seed training highly recommended")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    else:
        print("❌ Invalid arguments")
        print("Usage:")
        print("  python gps_analyzer.py check <gps_csv_file>")
        print("  python gps_analyzer.py analyze <train_gps_csv> <val_gps_csv>")
        sys.exit(1)


if __name__ == "__main__":
    main()