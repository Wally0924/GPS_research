import os
import pandas as pd

root_folder = '/home/rvl122-4090/SynologyDrive/dataset/random'  # 改成你的根目錄

dfs = []
for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file == 'gnss_data.csv':
            file_path = os.path.join(subdir, file)
            print("找到檔案：", file_path)    # 一定要有這行，幫助你 debug
            df = pd.read_csv(file_path)
            dfs.append(df)

if not dfs:
    print("⚠️ 沒有找到任何 gnss_data.csv 檔案，請確認路徑或檔案存在！")
else:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('merged_gnss_data.csv', index=False, encoding='utf-8-sig')
    print(f"合併完成，共有 {len(merged_df)} 筆資料，儲存為 merged_gnss_data.csv")

