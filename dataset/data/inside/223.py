import re

input_txt_path = 'merged_gnss_data.txt'     # 請改成你的 txt 檔案路徑
output_txt_path = 'clean_filenames.txt' # 輸出只包含純檔名的 txt

filenames = []

# 讀取每行並擷取出 .jpg 檔名
with open(input_txt_path, 'r') as file:
    for line in file:
        match = re.search(r'(\d{5,}\.jpg)', line.lower())  # 擷取像 00012345.jpg 的格式
        if match:
            filenames.append(match.group(1))

# 儲存成新的 txt
with open(output_txt_path, 'w') as file:
    for name in filenames:
        file.write(name + '\n')

print(f"✅ 已輸出 {len(filenames)} 筆純檔名至 {output_txt_path}")

