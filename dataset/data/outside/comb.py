import os
import shutil

# 設定你的根目錄
root_dir = './'    # 你可以改成絕對路徑
dst_dir = os.path.join(root_dir, 'imgs')
os.makedirs(dst_dir, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(root_dir):
    # 只抓名為 images 的資料夾
    if os.path.basename(dirpath) == 'images':
        print(f"找到 images 資料夾：{dirpath}")
        for file in os.listdir(dirpath):
            src_file = os.path.join(dirpath, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.isfile(src_file):
                # 如果有同名檔案，會自動覆蓋
                shutil.copy2(src_file, dst_file)
                print(f"複製 {src_file} 到 {dst_file}")

print("全部複製完成！")

