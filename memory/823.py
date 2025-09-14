import pandas as pd

input_csv = '/home/rvl122-4090/An/test_Final/ALL/test_gnss.csv'
output_csv = '/home/rvl122-4090/An/test_Final/ALL/test_gnss_fixed.csv'

df = pd.read_csv(input_csv)
df['lat'] = -0.0011679
df['long'] = -0.000201

df.to_csv(output_csv, index=False)
print(f"已完成！輸出檔案：{output_csv}")

