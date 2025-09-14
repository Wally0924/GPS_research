import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import seaborn as sns

class GPSParameterAnalyzer:
    """GPS參數分析器：自動計算optimal spatial_radius和spatial_threshold"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.analysis_results = {}
        
    def analyze_gps_distribution(self):
        """分析GPS數據分佈"""
        print("🔍 GPS數據分析")
        print("=" * 50)
        
        # 檢查是否有正規化欄位
        has_normalized = 'lat_norm' in self.data.columns and 'long_norm' in self.data.columns
        
        if has_normalized:
            print("✅ 發現正規化GPS數據")
            lat_col, long_col = 'lat_norm', 'long_norm'
            lat_data = self.data[lat_col].dropna()
            long_data = self.data[long_col].dropna()
        else:
            print("📍 使用原始GPS數據")
            lat_col, long_col = 'lat', 'long'
            lat_data = self.data[lat_col].dropna()
            long_data = self.data[long_col].dropna()
        
        # 基本統計
        stats = {
            'total_records': len(self.data),
            'valid_gps_records': len(lat_data),
            'lat_min': lat_data.min(),
            'lat_max': lat_data.max(),
            'lat_range': lat_data.max() - lat_data.min(),
            'lat_std': lat_data.std(),
            'long_min': long_data.min(),
            'long_max': long_data.max(),
            'long_range': long_data.max() - long_data.min(),
            'long_std': long_data.std(),
            'lat_col': lat_col,
            'long_col': long_col
        }
        
        self.analysis_results['basic_stats'] = stats
        
        print(f"📊 基本統計:")
        print(f"  總記錄數: {stats['total_records']}")
        print(f"  有效GPS記錄: {stats['valid_gps_records']}")
        print(f"  {lat_col}範圍: [{stats['lat_min']:.8f}, {stats['lat_max']:.8f}] (跨度: {stats['lat_range']:.8f})")
        print(f"  {long_col}範圍: [{stats['long_min']:.8f}, {stats['long_max']:.8f}] (跨度: {stats['long_range']:.8f})")
        print(f"  緯度標準差: {stats['lat_std']:.8f}")
        print(f"  經度標準差: {stats['long_std']:.8f}")
        
        return stats
    
    def calculate_distance_statistics(self, sample_size=2000):
        """計算GPS距離統計"""
        print(f"\n📏 GPS距離統計分析 (樣本數: {sample_size})")
        print("-" * 40)
        
        stats = self.analysis_results['basic_stats']
        lat_col, long_col = stats['lat_col'], stats['long_col']
        
        # 取樣以加快計算
        valid_data = self.data[[lat_col, long_col]].dropna()
        if len(valid_data) > sample_size:
            sample_data = valid_data.sample(n=sample_size, random_state=42)
        else:
            sample_data = valid_data
        
        # 計算歐幾里得距離矩陣
        coords = sample_data[[lat_col, long_col]].values
        distances = cdist(coords, coords, metric='euclidean')
        
        # 取上三角矩陣（排除對角線和重複）
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        distance_values = distances[upper_tri_indices]
        
        # 距離統計
        distance_stats = {
            'min_distance': distance_values.min(),
            'max_distance': distance_values.max(),
            'mean_distance': distance_values.mean(),
            'median_distance': np.median(distance_values),
            'std_distance': distance_values.std(),
            'percentiles': {
                '5%': np.percentile(distance_values, 5),
                '10%': np.percentile(distance_values, 10),
                '25%': np.percentile(distance_values, 25),
                '75%': np.percentile(distance_values, 75),
                '90%': np.percentile(distance_values, 90),
                '95%': np.percentile(distance_values, 95),
                '99%': np.percentile(distance_values, 99)
            },
            'zero_distances': np.sum(distance_values == 0),
            'total_pairs': len(distance_values)
        }
        
        self.analysis_results['distance_stats'] = distance_stats
        
        print(f"  最小距離: {distance_stats['min_distance']:.8f}")
        print(f"  最大距離: {distance_stats['max_distance']:.8f}")
        print(f"  平均距離: {distance_stats['mean_distance']:.8f}")
        print(f"  中位數距離: {distance_stats['median_distance']:.8f}")
        print(f"  距離標準差: {distance_stats['std_distance']:.8f}")
        print(f"  零距離對數: {distance_stats['zero_distances']}/{distance_stats['total_pairs']}")
        
        print(f"\n  距離分位數:")
        for k, v in distance_stats['percentiles'].items():
            print(f"    {k}: {v:.8f}")
        
        return distance_stats
    
    def find_optimal_spatial_radius(self):
        """尋找最佳spatial_radius (用於記憶庫)"""
        print(f"\n🧠 尋找最佳spatial_radius")
        print("-" * 40)
        
        distance_stats = self.analysis_results['distance_stats']
        
        # spatial_radius建議範圍：應該比最小非零距離小，但不要太小
        min_nonzero_dist = distance_stats['min_distance']
        mean_dist = distance_stats['mean_distance']
        
        # 建議值：5%分位數到10%分位數之間
        suggested_radius = distance_stats['percentiles']['5%'] * 0.1
        
        # 確保不會太小
        if suggested_radius < 1e-8:
            suggested_radius = 1e-6
        
        # 也不要太大
        if suggested_radius > mean_dist * 0.1:
            suggested_radius = mean_dist * 0.1
        
        spatial_radius_analysis = {
            'suggested_radius': suggested_radius,
            'min_nonzero_distance': min_nonzero_dist,
            'reasoning': f"設為5%分位數的10%，確保相近位置能聚類但不會過度聚合"
        }
        
        self.analysis_results['spatial_radius'] = spatial_radius_analysis
        
        print(f"  建議spatial_radius: {suggested_radius:.8f}")
        print(f"  理由: {spatial_radius_analysis['reasoning']}")
        
        return suggested_radius
    
    def find_optimal_spatial_threshold(self):
        """尋找最佳spatial_threshold (用於對比學習)"""
        print(f"\n🎯 尋找最佳spatial_threshold")
        print("-" * 40)
        
        distance_stats = self.analysis_results['distance_stats']
        
        # spatial_threshold建議：應該讓大部分樣本對成為負樣本
        # 但也要避免把真正相近的位置強制分離
        
        # 建議範圍：10%到25%分位數之間
        p10 = distance_stats['percentiles']['10%']
        p25 = distance_stats['percentiles']['25%']
        
        # 選擇15%分位數作為閾值
        suggested_threshold = np.percentile([p10, p25], 30)  # 介於10%和25%之間
        
        # 計算使用此閾值時的負樣本比例
        sample_data = self.data[[self.analysis_results['basic_stats']['lat_col'], 
                                self.analysis_results['basic_stats']['long_col']]].dropna()
        if len(sample_data) > 500:
            test_sample = sample_data.sample(n=500, random_state=42)
        else:
            test_sample = sample_data
        
        test_coords = test_sample.values
        test_distances = cdist(test_coords, test_coords, metric='euclidean')
        
        # 計算超過閾值的比例
        upper_tri = np.triu_indices_from(test_distances, k=1)
        test_distance_values = test_distances[upper_tri]
        negative_ratio = np.sum(test_distance_values > suggested_threshold) / len(test_distance_values)
        
        spatial_threshold_analysis = {
            'suggested_threshold': suggested_threshold,
            'negative_sample_ratio': negative_ratio,
            'total_test_pairs': len(test_distance_values),
            'negative_pairs': np.sum(test_distance_values > suggested_threshold),
            'reasoning': f"選擇讓{negative_ratio:.1%}的樣本對成為負樣本，平衡對比學習效果"
        }
        
        self.analysis_results['spatial_threshold'] = spatial_threshold_analysis
        
        print(f"  建議spatial_threshold: {suggested_threshold:.8f}")
        print(f"  負樣本比例: {negative_ratio:.1%}")
        print(f"  負樣本對數: {spatial_threshold_analysis['negative_pairs']}/{spatial_threshold_analysis['total_test_pairs']}")
        print(f"  理由: {spatial_threshold_analysis['reasoning']}")
        
        return suggested_threshold
    
    def test_dbscan_clustering(self, eps_candidates=None):
        """測試DBSCAN聚類效果"""
        print(f"\n🔬 DBSCAN聚類測試")
        print("-" * 40)
        
        if eps_candidates is None:
            distance_stats = self.analysis_results['distance_stats']
            # 測試不同的eps值
            eps_candidates = [
                distance_stats['percentiles']['5%'] * 0.1,
                distance_stats['percentiles']['5%'] * 0.5,
                distance_stats['percentiles']['10%'],
                distance_stats['percentiles']['25%'],
                distance_stats['mean_distance'] * 0.1
            ]
        
        lat_col = self.analysis_results['basic_stats']['lat_col']
        long_col = self.analysis_results['basic_stats']['long_col']
        
        coords = self.data[[lat_col, long_col]].dropna().values
        
        clustering_results = []
        
        for eps in eps_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(coords)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            result = {
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels),
                'avg_cluster_size': (len(labels) - n_noise) / max(n_clusters, 1)
            }
            clustering_results.append(result)
            
            print(f"  eps={eps:.8f}: {n_clusters}個聚類, {n_noise}個噪聲點 ({result['noise_ratio']:.1%})")
        
        self.analysis_results['clustering_results'] = clustering_results
        
        # 推薦最佳eps值
        best_result = min(clustering_results, 
                         key=lambda x: abs(x['noise_ratio'] - 0.1))  # 目標噪聲比例10%
        
        print(f"\n  推薦eps (spatial_radius): {best_result['eps']:.8f}")
        print(f"  將產生 {best_result['n_clusters']} 個聚類，噪聲比例 {best_result['noise_ratio']:.1%}")
        
        return best_result['eps']
    
    def generate_recommendations(self):
        """生成最終建議"""
        print(f"\n🎯 最終建議")
        print("=" * 50)
        
        spatial_radius = self.analysis_results['spatial_radius']['suggested_radius']
        spatial_threshold = self.analysis_results['spatial_threshold']['suggested_threshold']
        
        print(f"推薦參數設置:")
        print(f"  --spatial-radius {spatial_radius:.8f}")
        print(f"  --spatial-threshold {spatial_threshold:.8f}")
        
        print(f"\n命令行範例:")
        print(f"python -m tools.your_script \\")
        print(f"  [其他參數] \\")
        print(f"  --spatial-radius {spatial_radius:.8f} \\")
        print(f"  --spatial-threshold {spatial_threshold:.8f} \\")
        print(f"  --contrastive-weight 0.7")
        
        # 生成不同scenario的建議
        print(f"\n📋 不同場景建議:")
        
        print(f"🎯 保守設置 (更少負樣本):")
        conservative_threshold = spatial_threshold * 2
        print(f"  --spatial-threshold {conservative_threshold:.8f}")
        
        print(f"🚀 激進設置 (更多負樣本):")
        aggressive_threshold = spatial_threshold * 0.5
        print(f"  --spatial-threshold {aggressive_threshold:.8f}")
        
        print(f"🧠 記憶庫優化:")
        memory_radius = spatial_radius * 0.5
        print(f"  --spatial-radius {memory_radius:.8f}")
        
        return {
            'spatial_radius': spatial_radius,
            'spatial_threshold': spatial_threshold,
            'conservative_threshold': conservative_threshold,
            'aggressive_threshold': aggressive_threshold,
            'memory_radius': memory_radius
        }
    
    def plot_distance_distribution(self, save_path=None):
        """繪製距離分佈圖"""
        if 'distance_stats' not in self.analysis_results:
            print("請先執行 calculate_distance_statistics()")
            return
        
        # 重新計算距離用於繪圖
        lat_col = self.analysis_results['basic_stats']['lat_col']
        long_col = self.analysis_results['basic_stats']['long_col']
        
        sample_data = self.data[[lat_col, long_col]].dropna().sample(n=min(1000, len(self.data)), random_state=42)
        coords = sample_data.values
        distances = cdist(coords, coords, metric='euclidean')
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        distance_values = distances[upper_tri_indices]
        
        plt.figure(figsize=(12, 8))
        
        # 子圖1: 距離直方圖
        plt.subplot(2, 2, 1)
        plt.hist(distance_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.analysis_results['spatial_threshold']['suggested_threshold'], 
                   color='red', linestyle='--', label=f'建議threshold: {self.analysis_results["spatial_threshold"]["suggested_threshold"]:.6f}')
        plt.axvline(self.analysis_results['spatial_radius']['suggested_radius'], 
                   color='green', linestyle='--', label=f'建議radius: {self.analysis_results["spatial_radius"]["suggested_radius"]:.6f}')
        plt.xlabel('GPS距離')
        plt.ylabel('頻率')
        plt.title('GPS距離分佈')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子圖2: 累積分佈
        plt.subplot(2, 2, 2)
        sorted_distances = np.sort(distance_values)
        cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        plt.plot(sorted_distances, cumulative, 'b-', linewidth=2)
        plt.axvline(self.analysis_results['spatial_threshold']['suggested_threshold'], 
                   color='red', linestyle='--', label=f'建議threshold')
        plt.axvline(self.analysis_results['spatial_radius']['suggested_radius'], 
                   color='green', linestyle='--', label=f'建議radius')
        plt.xlabel('GPS距離')
        plt.ylabel('累積機率')
        plt.title('GPS距離累積分佈')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子圖3: 對數尺度直方圖
        plt.subplot(2, 2, 3)
        plt.hist(distance_values[distance_values > 0], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.yscale('log')
        plt.xlabel('GPS距離')
        plt.ylabel('頻率 (對數尺度)')
        plt.title('GPS距離分佈 (對數尺度)')
        plt.grid(True, alpha=0.3)
        
        # 子圖4: 箱型圖
        plt.subplot(2, 2, 4)
        plt.boxplot(distance_values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        plt.ylabel('GPS距離')
        plt.title('GPS距離箱型圖')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存到: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self):
        """執行完整分析"""
        print("🚀 啟動GPS參數完整分析")
        print("=" * 60)
        
        # 1. 基本分佈分析
        self.analyze_gps_distribution()
        
        # 2. 距離統計分析
        self.calculate_distance_statistics()
        
        # 3. 尋找最佳spatial_radius
        self.find_optimal_spatial_radius()
        
        # 4. 尋找最佳spatial_threshold
        self.find_optimal_spatial_threshold()
        
        # 5. DBSCAN聚類測試
        self.test_dbscan_clustering()
        
        # 6. 生成最終建議
        recommendations = self.generate_recommendations()
        
        print(f"\n✅ 分析完成！")
        
        return recommendations


# 使用範例
def analyze_your_gps_data(csv_path):
    """分析您的GPS數據"""
    analyzer = GPSParameterAnalyzer(csv_path)
    
    # 執行完整分析
    recommendations = analyzer.run_complete_analysis()
    
    # 繪製分佈圖
    try:
        analyzer.plot_distance_distribution(save_path='gps_distance_analysis.png')
    except Exception as e:
        print(f"繪圖時發生錯誤: {e}")
    
    return recommendations, analyzer

if __name__ == "__main__":
    # 使用您的CSV檔案
    csv_file = "YOUR_DATA_norm.csv"
    
    print("🔍 分析GPS數據以找到最佳參數...")
    recommendations, analyzer = analyze_your_gps_data(csv_file)
    
    print(f"\n🎯 快速設置:")
    print(f"--spatial-radius {recommendations['spatial_radius']:.8f}")
    print(f"--spatial-threshold {recommendations['spatial_threshold']:.8f}")