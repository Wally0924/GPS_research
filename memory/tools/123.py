import pandas as pd
import numpy as np

def gps_minmax_normalize_with_padding(csv_file, save_norm_file=None, save_minmax_file=None, padding=0.01):
    """
    讀取 GPS CSV，進行 min-max normalization（含 1% padding），回傳正規化結果
    Args:
        csv_file: 包含 'lat', 'long' 欄位的 CSV 檔路徑
        save_norm_file: 要儲存正規化後的檔案路徑（可為 None）
        save_minmax_file: 要儲存 min/max 的檔案路徑（可為 None）
        padding: padding 百分比（預設 0.01）
    Returns:
        norm_gps: shape (N, 2) numpy array
        minmax: dict with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    """
    df = pd.read_csv(csv_file)
    # 計算 min/max
    lat_min = df['lat'].min()
    lat_max = df['lat'].max()
    lon_min = df['long'].min()
    lon_max = df['long'].max()

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # 1% padding
    lat_min_pad = lat_min - lat_range * padding
    lat_max_pad = lat_max + lat_range * padding
    lon_min_pad = lon_min - lon_range * padding
    lon_max_pad = lon_max + lon_range * padding

    # min-max normalization
    df['lat_norm'] = (df['lat'] - lat_min_pad) / (lat_max_pad - lat_min_pad)
    df['long_norm'] = (df['long'] - lon_min_pad) / (lon_max_pad - lon_min_pad)

    norm_gps = df[['lat_norm', 'long_norm']].values

    # 儲存正規化結果
    if save_norm_file:
        df.to_csv(save_norm_file, index=False)
    # 儲存 minmax 參數
    if save_minmax_file:
        minmax_dict = {
            'lat_min': lat_min_pad,
            'lat_max': lat_max_pad,
            'lon_min': lon_min_pad,
            'lon_max': lon_max_pad,
        }
        np.save(save_minmax_file, minmax_dict)

    print(f"正規化後緯度範圍: {df['lat_norm'].min():.6f} ~ {df['lat_norm'].max():.6f}")
    print(f"正規化後經度範圍: {df['long_norm'].min():.6f} ~ {df['long_norm'].max():.6f}")

    return norm_gps, {
        'lat_min': lat_min_pad,
        'lat_max': lat_max_pad,
        'lon_min': lon_min_pad,
        'lon_max': lon_max_pad
    }

# ====== 使用範例 ======
# 假設你的 GPS CSV 為 train_gps.csv
csv_path = 'data/gnss/train_gnss.csv'
norm_gps, minmax = gps_minmax_normalize_with_padding(
    csv_file=csv_path,
    save_norm_file='data/gnss/train_gnss.csv',
    save_minmax_file='gps_minmax.npy',  # 儲存成 numpy 檔，下次推論可直接讀
    padding=0.01
)

# 如果要用 minmax 做新資料正規化
def normalize_new_gps(lat, lon, minmax):
    lat_norm = (lat - minmax['lat_min']) / (minmax['lat_max'] - minmax['lat_min'])
    lon_norm = (lon - minmax['lon_min']) / (minmax['lon_max'] - minmax['lon_min'])
    return lat_norm, lon_norm

# 例如：normalize_new_gps(23.0, 121.0, minmax)
