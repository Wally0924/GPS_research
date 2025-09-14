import os

def compare_files(txt_file_path, img_folder_path):
    """
    比較txt檔案中的檔案名稱與img資料夾中的圖片檔案名稱
    找出txt檔案中多出來的檔案名稱
    
    Args:
        txt_file_path (str): txt檔案的路徑
        img_folder_path (str): img資料夾的路徑
    
    Returns:
        list: txt檔案中多出來的檔案名稱
    """
    
    # 讀取txt檔案中的檔案名稱
    txt_files = set()
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if filename:  # 忽略空行
                    # 移除副檔名，只保留檔案名稱
                    name_without_ext = os.path.splitext(filename)[0]
                    txt_files.add(name_without_ext)
    except FileNotFoundError:
        print(f"找不到txt檔案: {txt_file_path}")
        return []
    except Exception as e:
        print(f"讀取txt檔案時發生錯誤: {e}")
        return []
    
    # 獲取img資料夾中的圖片檔案名稱
    img_files = set()
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    try:
        if os.path.exists(img_folder_path):
            for filename in os.listdir(img_folder_path):
                file_path = os.path.join(img_folder_path, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in supported_extensions:
                        img_files.add(name)
        else:
            print(f"找不到img資料夾: {img_folder_path}")
            return []
    except Exception as e:
        print(f"讀取img資料夾時發生錯誤: {e}")
        return []
    
    # 找出txt檔案中多出來的檔案名稱
    extra_files = txt_files - img_files
    
    # 顯示結果
    print(f"txt檔案中的檔案數量: {len(txt_files)}")
    print(f"img資料夾中的圖片數量: {len(img_files)}")
    print(f"txt檔案中多出來的檔案數量: {len(extra_files)}")
    print()
    
    if extra_files:
        print("txt檔案中多出來的檔案名稱:")
        for filename in sorted(extra_files):
            print(f"  - {filename}")
    else:
        print("txt檔案中沒有多出來的檔案名稱")
    
    return sorted(list(extra_files))

def main():
    # 設定檔案路徑 - 請根據實際情況修改
    txt_file_path = "clean_filenames.txt"  # txt檔案路徑
    img_folder_path = "imgs"          # img資料夾路徑
    
    # 執行比較
    extra_files = compare_files(txt_file_path, img_folder_path)
    
    # 可選：將結果寫入新的txt檔案
    if extra_files:
        output_file = "extra_files.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for filename in extra_files:
                    f.write(filename + '\n')
            print(f"\n結果已儲存至: {output_file}")
        except Exception as e:
            print(f"儲存結果時發生錯誤: {e}")

if __name__ == "__main__":
    main()
