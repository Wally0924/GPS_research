#!/usr/bin/env python3
"""
記憶庫參數智能分析器
根據GPS數據自動推薦最佳的 spatial_radius 和 memory_size 參數
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path
import json


class MemoryBankParameterAnalyzer:
    """記憶庫參數分析器"""
    
    def __init__(self, train_gps_csv: str, val_gps_csv: str = None):
        """
        Args:
            train_gps_csv: 訓練集GPS CSV文件路徑
            val_gps_csv: 驗證集GPS CSV文件路徑（可選）
        """
        self.train_gps_csv = train_gps_csv
        self.val_gps_csv = val_gps_csv
        
        # 載入GPS數據
        print("📂 載入GPS數據...")
        self.train_gps = pd.read_csv(train_gps_csv)
        print(f"✅ 訓練集: {len(self.train_gps)} 個GPS點")
        
        if val_gps_csv:
            self.val_gps = pd.read_csv(val_gps_csv)
            print(f"✅ 驗證集: {len(self.val_gps)} 個GPS點")
            self.all_gps = pd.concat([self.train_gps, self.val_gps], ignore_index=True)
        else:
            self.all_gps = self.train_gps
        
        print(f"📊 總共: {len(self.all_gps)} 個GPS點")
        
        # 基本統計
        self.gps_stats = self._compute_basic_stats()
        self._print_basic_stats()
    
    def _compute_basic_stats(self) -> Dict[str, float]:
        """計算基本GPS統計信息"""
        lats = self.all_gps['lat'].values
        lons = self.all_gps['long'].values
        
        return {
            'lat_min': lats.min(),
            'lat_max': lats.max(),
            'lat_range': lats.max() - lats.min(),
            'lat_std': lats.std(),
            'lon_min': lons.min(),
            'lon_max': lons.max(),
            'lon_range': lons.max() - lons.min(),
            'lon_std': lons.std(),
            'total_points': len(lats),
            'unique_coords': len(set(zip(lats, lons)))
        }
    
    def _print_basic_stats(self):
        """打印基本統計信息"""
        stats = self.gps_stats
        print(f"\n📈 GPS數據基本統計:")
        print(f"  緯度範圍: [{stats['lat_min']:.6f}, {stats['lat_max']:.6f}] (跨度: {stats['lat_range']:.6f})")
        print(f"  經度範圍: [{stats['lon_min']:.6f}, {stats['lon_max']:.6f}] (跨度: {stats['lon_range']:.6f})")
        print(f"  緯度標準差: {stats['lat_std']:.6f}")
        print(f"  經度標準差: {stats['lon_std']:.6f}")
        print(f"  總GPS點數: {stats['total_points']}")
        print(f"  唯一位置數: {stats['unique_coords']}")
        print(f"  重複率: {(1 - stats['unique_coords']/stats['total_points'])*100:.1f}%")
    
    def analyze_spatial_clustering(self, radius_candidates: List[float] = None) -> Dict[float, Dict]:
        """分析不同空間半徑下的聚類效果"""
        
        if radius_candidates is None:
            # 自動生成候選半徑：從GPS範圍的0.5%到20%
            base_range = max(self.gps_stats['lat_range'], self.gps_stats['lon_range'])
            radius_candidates = [
                base_range * factor for factor in 
                [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
            ]
        
        print(f"\n🔍 分析空間聚類效果...")
        print(f"測試半徑候選: {[f'{r:.6f}' for r in radius_candidates]}")
        
        clustering_results = {}
        
        for radius in radius_candidates:
            result = self._analyze_single_radius(radius)
            clustering_results[radius] = result
            
            print(f"  半徑 {radius:.6f}: "
                  f"{result['clusters']} 個聚類, "
                  f"平均 {result['avg_points_per_cluster']:.1f} 點/聚類, "
                  f"命中率 {result['hit_rate']:.3f}")
        
        return clustering_results
    
    def _analyze_single_radius(self, radius: float) -> Dict:
        """分析單個半徑的聚類效果"""
        
        # 模擬GPS量化過程
        def gps_to_key(lat, lon):
            lat_grid = round(lat / radius) * radius
            lon_grid = round(lon / radius) * radius
            return f"{lat_grid:.8f},{lon_grid:.8f}"
        
        # 構建聚類
        clusters = defaultdict(list)
        for _, row in self.all_gps.iterrows():
            key = gps_to_key(row['lat'], row['long'])
            clusters[key].append((row['lat'], row['long']))
        
        # 計算統計
        cluster_sizes = [len(points) for points in clusters.values()]
        
        # 模擬記憶庫命中率
        hit_count = sum(1 for size in cluster_sizes if size > 1)
        hit_rate = hit_count / len(cluster_sizes) if cluster_sizes else 0
        
        return {
            'radius': radius,
            'clusters': len(clusters),
            'total_points': sum(cluster_sizes),
            'avg_points_per_cluster': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_points_per_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'min_points_per_cluster': min(cluster_sizes) if cluster_sizes else 0,
            'cluster_sizes': cluster_sizes,
            'hit_rate': hit_rate,
            'clusters_with_multiple_points': hit_count,
            'compression_ratio': len(clusters) / self.gps_stats['total_points']
        }
    
    def compute_distance_statistics(self) -> Dict[str, Any]:
        """計算GPS點之間的距離統計"""
        print(f"\n📏 計算GPS點距離統計...")
        
        # 隨機採樣以加速計算（如果數據太大）
        sample_size = min(2000, len(self.all_gps))
        if len(self.all_gps) > sample_size:
            sampled_gps = self.all_gps.sample(n=sample_size, random_state=42)
            print(f"  採樣 {sample_size} 個點進行距離分析")
        else:
            sampled_gps = self.all_gps
        
        distances = []
        coords = [(row['lat'], row['long']) for _, row in sampled_gps.iterrows()]
        
        # 計算所有點對之間的距離
        for i in range(len(coords)):
            for j in range(i+1, min(i+50, len(coords))):  # 限制每個點最多比較50個鄰居
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
                distances.append(dist)
        
        distances = np.array(distances)
        
        distance_stats = {
            'min_distance': distances.min(),
            'max_distance': distances.max(),
            'mean_distance': distances.mean(),
            'median_distance': np.median(distances),
            'std_distance': distances.std(),
            'percentiles': {
                '5%': np.percentile(distances, 5),
                '10%': np.percentile(distances, 10),
                '25%': np.percentile(distances, 25),
                '75%': np.percentile(distances, 75),
                '90%': np.percentile(distances, 90),
                '95%': np.percentile(distances, 95),
            }
        }
        
        print(f"  最小距離: {distance_stats['min_distance']:.6f}")
        print(f"  平均距離: {distance_stats['mean_distance']:.6f}")
        print(f"  中位數距離: {distance_stats['median_distance']:.6f}")
        print(f"  10%分位數: {distance_stats['percentiles']['10%']:.6f}")
        print(f"  90%分位數: {distance_stats['percentiles']['90%']:.6f}")
        
        return distance_stats
    
    def recommend_optimal_parameters(self) -> Dict[str, Any]:
        """推薦最佳記憶庫參數"""
        print(f"\n🎯 生成記憶庫參數推薦...")
        
        # 1. 分析空間聚類
        clustering_results = self.analyze_spatial_clustering()
        
        # 2. 計算距離統計
        distance_stats = self.compute_distance_statistics()
        
        # 3. 基於多個指標評估最佳參數
        recommendations = self._evaluate_parameter_combinations(clustering_results, distance_stats)
        
        return recommendations
    
    def _evaluate_parameter_combinations(self, clustering_results: Dict, distance_stats: Dict) -> Dict:
        """評估參數組合並生成推薦"""
        
        # 評估標準
        def score_radius(result):
            """為每個半徑計算分數"""
            # 目標：
            # 1. 合理的聚類數量 (不要太多也不要太少)
            # 2. 每個聚類有足夠的點 (記憶庫才有用)
            # 3. 命中率要高 (能經常匹配到記憶)
            
            clusters = result['clusters']
            avg_points = result['avg_points_per_cluster']
            hit_rate = result['hit_rate']
            compression = result['compression_ratio']
            
            # 理想的聚類數量：總點數的10%-50%
            ideal_clusters_min = self.gps_stats['total_points'] * 0.1
            ideal_clusters_max = self.gps_stats['total_points'] * 0.5
            
            if clusters < ideal_clusters_min:
                cluster_score = clusters / ideal_clusters_min  # 太少聚類，懲罰
            elif clusters > ideal_clusters_max:
                cluster_score = ideal_clusters_max / clusters  # 太多聚類，懲罰
            else:
                cluster_score = 1.0  # 理想範圍
            
            # 每個聚類的平均點數分數 (2-10個點比較理想)
            if avg_points < 2:
                avg_points_score = avg_points / 2
            elif avg_points > 10:
                avg_points_score = 10 / avg_points
            else:
                avg_points_score = 1.0
            
            # 命中率分數 (越高越好)
            hit_rate_score = hit_rate
            
            # 綜合分數
            total_score = (cluster_score * 0.3 + 
                          avg_points_score * 0.3 + 
                          hit_rate_score * 0.4)
            
            return total_score
        
        # 評估所有半徑
        scored_results = []
        for radius, result in clustering_results.items():
            score = score_radius(result)
            scored_results.append((score, radius, result))
        
        # 排序得到最佳結果
        scored_results.sort(reverse=True)
        
        # 生成推薦
        best_score, best_radius, best_result = scored_results[0]
        
        # 根據最佳聚類結果推薦記憶庫大小
        avg_cluster_size = best_result['avg_points_per_cluster']
        max_cluster_size = best_result['max_points_per_cluster']
        
        # 記憶庫大小建議：能容納大部分聚類的點
        recommended_memory_sizes = {
            'conservative': max(10, int(avg_cluster_size * 1.5)),
            'moderate': max(20, int(avg_cluster_size * 2.5)),
            'aggressive': max(30, int(max_cluster_size * 0.8))
        }
        
        recommendations = {
            'optimal_spatial_radius': best_radius,
            'spatial_radius_score': best_score,
            'recommended_memory_sizes': recommended_memory_sizes,
            'clustering_analysis': best_result,
            'all_radius_scores': [(score, radius) for score, radius, _ in scored_results],
            'distance_based_suggestions': {
                'min_useful_radius': distance_stats['percentiles']['10%'],
                'max_useful_radius': distance_stats['percentiles']['90%'],
                'sweet_spot_radius': distance_stats['percentiles']['25%']
            },
            'performance_prediction': self._predict_memory_performance(best_result, recommended_memory_sizes)
        }
        
        return recommendations
    
    def _predict_memory_performance(self, clustering_result: Dict, memory_sizes: Dict) -> Dict:
        """預測記憶庫性能"""
        
        cluster_sizes = clustering_result['cluster_sizes']
        
        performance = {}
        for size_name, memory_size in memory_sizes.items():
            # 計算能被完全存儲的聚類比例
            fully_stored = sum(1 for size in cluster_sizes if size <= memory_size)
            fully_stored_ratio = fully_stored / len(cluster_sizes)
            
            # 計算平均存儲利用率
            avg_utilization = np.mean([min(size, memory_size) / memory_size for size in cluster_sizes])
            
            performance[size_name] = {
                'memory_size': memory_size,
                'fully_stored_clusters_ratio': fully_stored_ratio,
                'average_utilization': avg_utilization,
                'expected_hit_rate': clustering_result['hit_rate'] * fully_stored_ratio
            }
        
        return performance
    
    def print_recommendations(self, recommendations: Dict):
        """打印推薦結果"""
        print(f"\n🎯 記憶庫參數推薦結果")
        print(f"=" * 60)
        
        print(f"\n📍 最佳空間半徑:")
        print(f"  推薦值: {recommendations['optimal_spatial_radius']:.6f}")
        print(f"  評分: {recommendations['spatial_radius_score']:.3f}")
        
        clustering = recommendations['clustering_analysis']
        print(f"\n📊 聚類效果:")
        print(f"  聚類數量: {clustering['clusters']}")
        print(f"  平均每聚類點數: {clustering['avg_points_per_cluster']:.1f}")
        print(f"  最大聚類點數: {clustering['max_points_per_cluster']}")
        print(f"  記憶命中率: {clustering['hit_rate']:.3f}")
        print(f"  壓縮比: {clustering['compression_ratio']:.3f}")
        
        print(f"\n🧠 推薦記憶庫大小:")
        memory_sizes = recommendations['recommended_memory_sizes']
        performance = recommendations['performance_prediction']
        
        for strategy, size in memory_sizes.items():
            perf = performance[strategy]
            print(f"  {strategy.capitalize():12s}: {size:3d} "
                  f"(預期命中率: {perf['expected_hit_rate']:.3f}, "
                  f"利用率: {perf['average_utilization']:.3f})")
        
        print(f"\n🎲 距離分析建議:")
        dist_suggestions = recommendations['distance_based_suggestions']
        print(f"  最小有用半徑: {dist_suggestions['min_useful_radius']:.6f}")
        print(f"  最大有用半徑: {dist_suggestions['max_useful_radius']:.6f}")
        print(f"  甜蜜點半徑: {dist_suggestions['sweet_spot_radius']:.6f}")
        
        print(f"\n✨ 最終建議:")
        best_memory = memory_sizes['moderate']
        best_radius = recommendations['optimal_spatial_radius']
        
        print(f"  --spatial-radius {best_radius:.6f}")
        print(f"  --memory-size {best_memory}")
        
        print(f"\n📋 訓練命令範例:")
        print(f"  python geotrain_v2_early_v1.py \\")
        print(f"    --spatial-radius {best_radius:.6f} \\")
        print(f"    --memory-size {best_memory} \\")
        print(f"    其他參數...")
    
    def save_analysis_report(self, recommendations: Dict, output_path: str):
        """保存分析報告"""
        report = {
            'gps_data_info': {
                'train_csv': self.train_gps_csv,
                'val_csv': self.val_gps_csv,
                'total_points': self.gps_stats['total_points'],
                'unique_coords': self.gps_stats['unique_coords']
            },
            'gps_statistics': self.gps_stats,
            'recommendations': recommendations,
            'suggested_commands': {
                'conservative': f"--spatial-radius {recommendations['optimal_spatial_radius']:.6f} --memory-size {recommendations['recommended_memory_sizes']['conservative']}",
                'moderate': f"--spatial-radius {recommendations['optimal_spatial_radius']:.6f} --memory-size {recommendations['recommended_memory_sizes']['moderate']}",
                'aggressive': f"--spatial-radius {recommendations['optimal_spatial_radius']:.6f} --memory-size {recommendations['recommended_memory_sizes']['aggressive']}"
            }
        }
        
        # 保存為JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 分析報告已保存到: {output_path}")
    
    def visualize_clustering_effects(self, clustering_results: Dict, save_path: str = None):
        """可視化聚類效果"""
        try:
            import matplotlib.pyplot as plt
            
            # 提取數據用於繪圖
            radii = list(clustering_results.keys())
            clusters = [result['clusters'] for result in clustering_results.values()]
            avg_points = [result['avg_points_per_cluster'] for result in clustering_results.values()]
            hit_rates = [result['hit_rate'] for result in clustering_results.values()]
            
            # 創建子圖
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # 圖1: 聚類數量 vs 半徑
            ax1.plot(radii, clusters, 'b-o')
            ax1.set_xlabel('Spatial Radius')
            ax1.set_ylabel('Number of Clusters')
            ax1.set_title('Clusters vs Spatial Radius')
            ax1.grid(True)
            
            # 圖2: 平均每聚類點數 vs 半徑
            ax2.plot(radii, avg_points, 'g-o')
            ax2.set_xlabel('Spatial Radius')
            ax2.set_ylabel('Avg Points per Cluster')
            ax2.set_title('Cluster Size vs Spatial Radius')
            ax2.grid(True)
            
            # 圖3: 命中率 vs 半徑
            ax3.plot(radii, hit_rates, 'r-o')
            ax3.set_xlabel('Spatial Radius')
            ax3.set_ylabel('Hit Rate')
            ax3.set_title('Memory Hit Rate vs Spatial Radius')
            ax3.grid(True)
            
            # 圖4: 聚類大小分布
            best_radius = max(clustering_results.keys(), 
                            key=lambda r: clustering_results[r]['hit_rate'])
            cluster_sizes = clustering_results[best_radius]['cluster_sizes']
            ax4.hist(cluster_sizes, bins=20, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Cluster Size')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Cluster Size Distribution (radius={best_radius:.6f})')
            ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 可視化圖表已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("⚠️  matplotlib未安裝，跳過可視化")


def main():
    parser = argparse.ArgumentParser(description="記憶庫參數智能分析器")
    parser.add_argument("train_gps_csv", help="訓練集GPS CSV文件路徑")
    parser.add_argument("--val-gps-csv", help="驗證集GPS CSV文件路徑")
    parser.add_argument("--output-dir", default="./memory_analysis", help="輸出目錄")
    parser.add_argument("--visualize", action="store_true", help="生成可視化圖表")
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🧠 記憶庫參數智能分析器")
    print("=" * 50)
    
    # 創建分析器
    analyzer = MemoryBankParameterAnalyzer(
        train_gps_csv=args.train_gps_csv,
        val_gps_csv=args.val_gps_csv
    )
    
    # 生成推薦
    recommendations = analyzer.recommend_optimal_parameters()
    
    # 打印推薦
    analyzer.print_recommendations(recommendations)
    
    # 保存報告
    report_path = output_dir / "memory_bank_analysis_report.json"
    analyzer.save_analysis_report(recommendations, str(report_path))
    
    # 可視化
    if args.visualize:
        clustering_results = analyzer.analyze_spatial_clustering()
        viz_path = output_dir / "clustering_analysis.png"
        analyzer.visualize_clustering_effects(clustering_results, str(viz_path))
    
    print(f"\n✅ 分析完成！")
    print(f"📁 結果保存在: {output_dir}")


if __name__ == "__main__":
    main()