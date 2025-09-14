import pandas as pd
import numpy as np

# 讀訓練集 minmax 參數
minmax = np.load('gps_minmax.npy', allow_pickle=True).item()

# 讀你的 csv
df = pd.read_csv('data/gnss/train_gnss.csv')

# 正規化
df['lat_norm'] = (df['lat'] - minmax['lat_min']) / (minmax['lat_max'] - minmax['lat_min'])
df['long_norm'] = (df['long'] - minmax['lon_min']) / (minmax['lon_max'] - minmax['lon_min'])

# 存檔（包含原始欄位+正規化欄位）
df.to_csv('YOUR_DATA_norm.csv', index=False)

# 預覽部分正規化後的值
print(df[['lat', 'long', 'lat_norm', 'long_norm']].head(10))
