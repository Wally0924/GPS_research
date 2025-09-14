import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# 讀取 minmax 訓練參數
minmax = np.load('gps_minmax.npy', allow_pickle=True).item()  # 這是 dict 格式

# 讀你要分析的 GPS CSV
csv_file = 'data/gnss/train_gnss.csv'
df = pd.read_csv(csv_file)

# 用訓練集的 minmax 做正規化（和你訓練時完全一致）
lat_norm = (df['lat'] - minmax['lat_min']) / (minmax['lat_max'] - minmax['lat_min'])
lon_norm = (df['long'] - minmax['lon_min']) / (minmax['lon_max'] - minmax['lon_min'])
gps_norm = np.stack([lat_norm, lon_norm], axis=1)  # shape (N, 2)

# 印範圍（可以確認有無小於0或大於1，是不是極端點）
print(f'lat_norm 範圍: {lat_norm.min():.4f} ~ {lat_norm.max():.4f}')
print(f'lon_norm 範圍: {lon_norm.min():.4f} ~ {lon_norm.max():.4f}')

# 計算距離分布
dists = pdist(gps_norm, metric='euclidean')
print(f"距離 min: {dists.min():.6f}")
print(f"距離 1% 分位: {np.quantile(dists, 0.01):.6f}")
print(f"距離 10% 分位: {np.quantile(dists, 0.10):.6f}")
print(f"距離 50% 分位: {np.quantile(dists, 0.50):.6f}")
print(f"距離 90% 分位: {np.quantile(dists, 0.90):.6f}")
print(f"距離 max: {dists.max():.6f}")

plt.hist(dists, bins=100)
plt.xlabel('Normalized GPS Distance')
plt.ylabel('Count')
plt.title(f'Pairwise Normalized GPS Distance ({csv_file})')
plt.show()
