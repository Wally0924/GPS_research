from ultralytics import YOLO
import csv, os

# ✅ 載入模型
model = YOLO("runs/detect/my_run5/weights/best.pt")

# ✅ 執行 val
results = model.val(
    data="data_gps.yaml",
    split="val",
    imgsz=640,
    batch=16
)

# ✅ 取得 YOLO 的輸出資料夾 (runs/val/exp)
csv_filename = os.path.join(results.save_dir, "val_results.csv")

# ✅ 輸出總體結果
print("\n========== Validation Results ==========")
print(f"mAP@50: {results.box.map50:.4f}")
print(f"mAP@50-95: {results.box.map:.4f}")
print("========================================\n")

# ✅ 存每類別結果 (mAP50 + mAP50-95)
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "mAP50", "mAP50-95"])
    for cls_id, cls_name in results.names.items():
        map50 = results.box.maps50[cls_id] if hasattr(results.box, "maps50") else "N/A"
        writer.writerow([cls_name, map50, results.box.maps[cls_id]])

print(f"Validation results saved to: {os.path.abspath(csv_filename)}")
