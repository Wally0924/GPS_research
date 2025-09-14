import os
import shutil

# 設定根目錄（要搜尋的資料夾）
root_dir = './'   # 改成你要搜尋的根目錄
dst_dir = os.path.join(root_dir, 'mask')
os.makedirs(dst_dir, exist_ok=True)

found_any = False

for dirpath, dirnames, filenames in os.walk(root_dir):
    # 跳過根目錄本身
    if dirpath == root_dir or os.path.abspath(dirpath) == os.path.abspath(dst_dir):
        continue
    for file in filenames:
        if file.lower().endswith('.png'):
            src_file = os.path.join(dirpath, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.isfile(src_file):
                print(f"複製 {src_file} 到 {dst_file}")
                shutil.copy2(src_file, dst_file)
                found_any = True

if not found_any:
    print("⚠️ 沒有找到任何 .png 檔案")
else:
    print("全部複製完成！")

