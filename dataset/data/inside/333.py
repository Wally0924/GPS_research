import os

def compare_files(txt_file_path, img_folder_path):
    """
    比較txt檔案中的檔案名稱與img資料夾中的圖片檔案名稱
    找出txt檔案中多出來的檔案名稱，並分析重複項目
    
    Args:
        txt_file_path (str): txt檔案的路徑
        img_folder_path (str): img資料夾的路徑
    
    Returns:
        dict: 包含分析結果的字典
    """
    
    # 讀取txt檔案中的檔案名稱（包含重複統計）
    txt_files_list = []  # 保存所有檔案名稱（包含重複）
    txt_files_set = set()  # 去重後的檔案名稱
    file_count = {}  # 統計每個檔案名稱出現的次數
    total_lines = 0
    empty_lines = 0
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                filename = line.strip()
                if filename:  # 忽略空行
                    # 移除副檔名，只保留檔案名稱
                    name_without_ext = os.path.splitext(filename)[0]
                    txt_files_list.append(name_without_ext)
                    txt_files_set.add(name_without_ext)
                    
                    # 統計出現次數
                    file_count[name_without_ext] = file_count.get(name_without_ext, 0) + 1
                else:
                    empty_lines += 1
    except FileNotFoundError:
        print(f"找不到txt檔案: {txt_file_path}")
        return {}
    except Exception as e:
        print(f"讀取txt檔案時發生錯誤: {e}")
        return {}
    
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
            return {}
    except Exception as e:
        print(f"讀取img資料夾時發生錯誤: {e}")
        return {}
    
    # 找出重複的檔案名稱
    duplicates = {name: count for name, count in file_count.items() if count > 1}
    
    # 找出txt檔案中多出來的檔案名稱
    extra_files = txt_files_set - img_files
    missing_files = img_files - txt_files_set
    
    # 顯示詳細分析結果
    print("=" * 60)
    print("檔案分析報告")
    print("=" * 60)
    print(f"txt檔案總行數: {total_lines}")
    print(f"空行數量: {empty_lines}")
    print(f"有效檔案名稱總數: {len(txt_files_list)}")
    print(f"去重後檔案名稱數量: {len(txt_files_set)}")
    print(f"img資料夾中的圖片數量: {len(img_files)}")
    print(f"重複檔案名稱數量: {len(duplicates)}")
    print(f"重複項目總計: {sum(duplicates.values()) - len(duplicates)}")
    print()
    
    if duplicates:
        print("重複的檔案名稱:")
        for filename, count in sorted(duplicates.items()):
            print(f"  - {filename}: 出現 {count} 次")
        print()
    
    if extra_files:
        print(f"txt檔案中多出來的檔案名稱 ({len(extra_files)} 個):")
        for filename in sorted(extra_files):
            print(f"  - {filename}")
        print()
    else:
        print("txt檔案中沒有多出來的檔案名稱")
        print()
    
    if missing_files:
        print(f"img資料夾中有但txt檔案中沒有的檔案 ({len(missing_files)} 個):")
        for filename in sorted(missing_files):
            print(f"  - {filename}")
        print()
    
    return {
        'total_lines': total_lines,
        'empty_lines': empty_lines,
        'valid_files_count': len(txt_files_list),
        'unique_files_count': len(txt_files_set),
        'img_files_count': len(img_files),
        'duplicates': duplicates,
        'extra_files': sorted(list(extra_files)),
        'missing_files': sorted(list(missing_files))
    }

def main():
    # 設定檔案路徑 - 請根據實際情況修改
    txt_file_path = "clean_filenames.txt"  # txt檔案路徑
    img_folder_path = "imgs"          # img資料夾路徑
    
    # 執行比較
    result = compare_files(txt_file_path, img_folder_path)
    
    if not result:
        return
    
    # 可選：將重複檔案名稱寫入檔案
    if result['duplicates']:
        duplicates_file = "duplicate_files.txt"
        try:
            with open(duplicates_file, 'w', encoding='utf-8') as f:
                f.write("重複的檔案名稱:\n")
                for filename, count in sorted(result['duplicates'].items()):
                    f.write(f"{filename}: 出現 {count} 次\n")
            print(f"重複檔案清單已儲存至: {duplicates_file}")
        except Exception as e:
            print(f"儲存重複檔案清單時發生錯誤: {e}")
    
    # 可選：將多出來的檔案名稱寫入檔案
    if result['extra_files']:
        extra_file = "extra_files.txt"
        try:
            with open(extra_file, 'w', encoding='utf-8') as f:
                for filename in result['extra_files']:
                    f.write(filename + '\n')
            print(f"多出來的檔案清單已儲存至: {extra_file}")
        except Exception as e:
            print(f"儲存多出來的檔案清單時發生錯誤: {e}")

if __name__ == "__main__":
    main()
