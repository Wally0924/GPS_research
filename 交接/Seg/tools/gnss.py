#!/usr/bin/env python3
"""
GPS參數分析工具
幫助確定最佳的 spatial_radius 和 spatial_threshold 參數
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def analyze_gps_data(csv_path):
    """分析GPS數據的基本統計"""
    print("📊 GPS數據基本分析")
    print("=" * 50)
    
    # 讀取GPS數據
    gps_data = pd.read_csv(csv_path)
    print(f"總GPS記錄數: {len(gps_data)}")
    
    lats = gps_data['lat'].values
    lons = gps_data['long'].values
    
    # 基本統計
    print(f"\n📈 GPS統計信息:")
    print(f"  緯度範圍: [{lats.min():.6f}, {lats.max():.6f}] (跨度: {lats.max()-lats.min():.6f})")
    print(f"  經度範圍: [{lons.min():.6f}, {lons.max():.6f}] (跨度: {lons.max()-lons.min():.6f})")
    print(f"  緯度標準差: {lats.std():.6f}")
    print(f"  經度標準差: {lons.std():.6f}")
    
    # 重複率分析
    unique_coords = set((lat, lon) for lat, lon in zip(lats, lons))
    duplicate_rate = (len(gps_data) - len(unique_coords)) / len(gps_data) * 100
    print(f"  唯一位置數: {len(unique_coords)}")
    print(f"  重複座標率: {duplicate_rate:.2f}%")
    
    # 距離分析
    distances = []
    sample_size = min(1000, len(gps_data))
    sample_indices = np.random.choice(len(gps_data), sample_size, replace=False)
    
    for i in range(sample_size):
        for j in range(i+1, min(i+10, sample_size)):
            lat1, lon1 = lats[sample_indices[i]], lons[sample_indices[i]]
            lat2, lon2 = lats[sample_indices[j]], lons[sample_indices[j]]
            dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
            distances.append(dist)
    
    distances = np.array(distances)
    print(f"\n📏 GPS點間距離分析:")
    print(f"  平均距離: {distances.mean():.6f}")
    print(f"  最小距離: {distances.min():.6f}")
    print(f"  中位數距離: {np.median(distances):.6f}")
    print(f"  90%分位數: {np.percentile(distances, 90):.6f}")
    print(f"  95%分位數: {np.percentile(distances, 95):.6f}")
    
    return {
        'lat_range': (lats.min(), lats.max()),
        'lon_range': (lons.min(), lons.max()),
        'unique_coords': len(unique_coords),
        'total_coords': len(gps_data),
        'distances': distances
    }


def test_spatial_radius(csv_path, radius_candidates=None):
    """測試不同spatial_radius的位置保留率"""
    print("\n🔧 Spatial Radius 測試")
    print("=" * 50)
    
    if radius_candidates is None:
        # 根據數據自動生成候選值
        gps_data = pd.read_csv(csv_path)
        lat_range = gps_data['lat'].max() - gps_data['lat'].min()
        lon_range = gps_data['long'].max() - gps_data['long'].min()
        avg_range = (lat_range + lon_range) / 2
        
        radius_candidates = [
            avg_range * 0.001,  # 很小
            avg_range * 0.005,  # 小
            avg_range * 0.01,   # 中等偏小
            avg_range * 0.02,   # 中等
            avg_range * 0.05,   # 中等偏大
            avg_range * 0.1,    # 大
            avg_range * 0.2,    # 很大
        ]
    
    gps_data = pd.read_csv(csv_path)
    total_coords = len(gps_data)
    
    print("Radius\t\t原始位置\t量化位置\t保留率\t\t建議")
    print("-" * 70)
    
    best_radius = None
    best_score = 0
    
    for radius in radius_candidates:
        # 模擬量化過程
        quantized_keys = set()
        for _, row in gps_data.iterrows():
            lat, lon = row['lat'], row['long']
            lat_grid = round(lat / radius) * radius
            lon_grid = round(lon / radius) * radius
            quantized_keys.add(f"{lat_grid:.7f},{lon_grid:.7f}")
        
        quantized_unique = len(quantized_keys)
        retention_rate = quantized_unique / total_coords
        
        # 評分標準：30-70%為最佳範圍
        if 0.3 <= retention_rate <= 0.7:
            recommendation = "✅ 推薦"
            score = 1.0 - abs(retention_rate - 0.5)  # 50%為最佳
            if score > best_score:
                best_score = score
                best_radius = radius
        elif 0.2 <= retention_rate < 0.3 or 0.7 < retention_rate <= 0.8:
            recommendation = "⚠️  可接受"
        elif retention_rate < 0.2:
            recommendation = "❌ 太大"
        else:
            recommendation = "❌ 太小"
        
        print(f"{radius:.6f}\t{total_coords}\t\t{quantized_unique}\t\t{retention_rate:.1%}\t\t{recommendation}")
    
    if best_radius:
        print(f"\n🎯 推薦的最佳 spatial_radius: {best_radius:.6f}")
    else:
        print(f"\n⚠️  所有候選值都不理想，建議手動調整")
    
    return best_radius


def test_spatial_threshold(csv_path, threshold_candidates=None):
    """測試不同spatial_threshold對對比學習的影響"""
    print("\n🎯 Spatial Threshold 測試")
    print("=" * 50)
    
    gps_data = pd.read_csv(csv_path)
    
    if threshold_candidates is None:
        # 基於距離分析生成候選值
        distances = []
        sample_size = min(500, len(gps_data))
        for i in range(sample_size):
            for j in range(i+1, min(i+10, sample_size)):
                lat1, lon1 = gps_data.iloc[i]['lat'], gps_data.iloc[i]['long']
                lat2, lon2 = gps_data.iloc[j]['lat'], gps_data.iloc[j]['long']
                dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
                distances.append(dist)
        
        distances = np.array(distances)
        threshold_candidates = [
            np.percentile(distances, 10),   # 很小
            np.percentile(distances, 25),   # 小
            np.percentile(distances, 50),   # 中位數
            np.percentile(distances, 75),   # 大
            np.percentile(distances, 90),   # 很大
        ]
    
    print("Threshold\t平均負樣本數\t正樣本率\t負樣本率\t建議")
    print("-" * 70)
    
    best_threshold = None
    best_balance = 0
    
    for threshold in threshold_candidates:
        total_pairs = 0
        negative_pairs = 0
        
        # 隨機採樣計算
        sample_size = min(100, len(gps_data))
        sample_indices = np.random.choice(len(gps_data), sample_size, replace=False)
        
        for i in range(sample_size):
            lat1, lon1 = gps_data.iloc[sample_indices[i]]['lat'], gps_data.iloc[sample_indices[i]]['long']
            
            neg_count = 0
            for j in range(sample_size):
                if i != j:
                    lat2, lon2 = gps_data.iloc[sample_indices[j]]['lat'], gps_data.iloc[sample_indices[j]]['long']
                    dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5
                    total_pairs += 1
                    if dist > threshold:
                        neg_count += 1
                        negative_pairs += 1
        
        avg_negatives = negative_pairs / sample_size if sample_size > 0 else 0
        positive_rate = 1 - (negative_pairs / max(total_pairs, 1))
        negative_rate = negative_pairs / max(total_pairs, 1)
        
        # 評分：希望負樣本率在60-80%之間
        if 0.6 <= negative_rate <= 0.8:
            recommendation = "✅ 推薦"
            balance_score = 1.0 - abs(negative_rate - 0.7)
            if balance_score > best_balance:
                best_balance = balance_score
                best_threshold = threshold
        elif 0.4 <= negative_rate < 0.6 or 0.8 < negative_rate <= 0.9:
            recommendation = "⚠️  可接受"
        elif negative_rate < 0.4:
            recommendation = "❌ 太小"
        else:
            recommendation = "❌ 太大"
        
        print(f"{threshold:.6f}\t{avg_negatives:.1f}\t\t{positive_rate:.1%}\t\t{negative_rate:.1%}\t\t{recommendation}")
    
    if best_threshold:
        print(f"\n🎯 推薦的最佳 spatial_threshold: {best_threshold:.6f}")
    else:
        print(f"\n⚠️  所有候選值都不理想，建議手動調整")
    
    return best_threshold


def explain_parameters():
    """解釋兩個參數的差異"""
    print("\n📚 參數說明")
    print("=" * 50)
    print("""
🔧 spatial_radius (記憶庫量化參數):
   • 用途: 將GPS座標量化到網格，決定記憶庫的位置精度
   • 原理: 將相近的GPS位置合併為同一個記憶slot
   • 影響: 
     - 太小 → 位置過於分散，記憶庫效率低
     - 太大 → 位置過度聚合，丟失空間細節
   • 建議: 讓位置保留率在30-70%之間
   
🎯 spatial_threshold (對比學習參數):
   • 用途: 決定哪些GPS位置對被視為"負樣本"
   • 原理: 距離 > threshold 的位置對用於對比學習
   • 影響:
     - 太小 → 大部分樣本都是負樣本，學習困難
     - 太大 → 負樣本太少，學習不到位置差異
   • 建議: 讓負樣本率在60-80%之間

🔗 兩者關係:
   • spatial_radius 影響記憶庫結構
   • spatial_threshold 影響訓練過程
   • 通常 spatial_threshold > spatial_radius
""")


def generate_recommended_config(csv_path):
    """生成推薦的配置"""
    print("\n⚙️  推薦配置生成")
    print("=" * 50)
    
    # 分析數據
    stats = analyze_gps_data(csv_path)
    
    # 測試參數
    best_radius = test_spatial_radius(csv_path)
    best_threshold = test_spatial_threshold(csv_path)
    
    print(f"\n🎯 推薦的訓練配置:")
    print(f"--spatial-radius {best_radius:.6f}")
    print(f"--spatial-threshold {best_threshold:.6f}")
    
    # 根據數據特性調整其他參數
    unique_rate = stats['unique_coords'] / stats['total_coords']
    if unique_rate < 0.3:
        print(f"--memory-size 50  # GPS重複率高，增加記憶容量")
    elif unique_rate > 0.8:
        print(f"--memory-size 15  # GPS變化大，減少記憶容量")
    else:
        print(f"--memory-size 20  # 標準記憶容量")
    
    avg_distance = np.mean(stats['distances'])
    if avg_distance < 0.0001:
        print(f"--contrastive-weight 0.1  # GPS變化小，增加對比學習權重")
        print(f"--memory-warmup-epochs 5  # 延長預熱期")
    else:
        print(f"--contrastive-weight 0.05  # 標準對比學習權重")
        print(f"--memory-warmup-epochs 3   # 標準預熱期")


def main():
    parser = argparse.ArgumentParser(description="GPS參數分析工具")
    parser.add_argument("gps_csv", help="GPS CSV文件路徑")
    parser.add_argument("--explain", action="store_true", help="解釋參數含義")
    parser.add_argument("--test-radius", action="store_true", help="測試spatial_radius")
    parser.add_argument("--test-threshold", action="store_true", help="測試spatial_threshold")
    parser.add_argument("--all", action="store_true", help="執行所有分析")
    
    args = parser.parse_args()
    
    if args.explain or args.all:
        explain_parameters()
    
    if args.test_radius or args.all:
        test_spatial_radius(args.gps_csv)
    
    if args.test_threshold or args.all:
        test_spatial_threshold(args.gps_csv)
    
    if args.all:
        generate_recommended_config(args.gps_csv)


if __name__ == "__main__":
    # 如果直接執行，提供簡單的測試介面
    import sys
    if len(sys.argv) == 2:
        csv_path = sys.argv[1]
        explain_parameters()
        generate_recommended_config(csv_path)
    else:
        main()