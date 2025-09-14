import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.metric import Metrics
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.geo_v2 import create_memory_enhanced_geo_segformer
from tools.geotrain_v2 import MemoryEnhancedGeoSegDataset, setup_gps_normalization


class GeoSlideInferencer:
    """
    專為GeoSegformer設計的滑窗推理器
    """
    def __init__(
        self, 
        crop_size: tuple[int, int], 
        stride: tuple[int, int], 
        num_categories: int
    ) -> None:
        self.crop_size = crop_size
        self.stride = stride
        self.num_categories = num_categories

    def inference(self, model: torch.nn.Module, img: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        """
        對GeoSegformer進行滑窗推理
        """
        return self.slide_inference(model, img, gps, self.crop_size, self.stride, self.num_categories)
    
    def slide_inference(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        gps: torch.Tensor,
        crop_size: tuple[int, int],
        stride: tuple[int, int],
        num_classes: int,
    ):
        """滑窗推理實現"""
        import torch.nn.functional as F
        
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                
                # 對於GeoSegformer，需要傳入GPS座標
                crop_outputs = model(crop_img, gps, return_embeddings=False, update_memory=False)
                crop_seg_logit = crop_outputs['segmentation_logits']
                
                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1
        
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Memory-Enhanced GeoSegformer Testing")
    # 基本參數
    parser.add_argument("img_dir", type=str, help="測試影像目錄")
    parser.add_argument("ann_dir", type=str, help="測試標註目錄")
    parser.add_argument("category_csv", type=str, help="類別定義CSV文件")
    parser.add_argument("checkpoint", type=str, help="模型檢查點路徑")
    parser.add_argument("test_gps_csv", type=str, help="測試集GPS CSV文件")
    
    # GPS正規化參數（需要與訓練時一致）
    parser.add_argument("--train-gps-csv", type=str, required=True, 
                       help="訓練集GPS CSV（用於計算正規化參數）")
    parser.add_argument("--val-gps-csv", type=str, required=True,
                       help="驗證集GPS CSV（用於計算正規化參數）") 
    parser.add_argument("--gps-norm-method", type=str, default="minmax",
                       choices=["minmax", "zscore"],
                       help="GPS正規化方法（必須與訓練時一致）")
    
    # 模型參數（需要與訓練時一致）
    parser.add_argument("--model-size", type=str, default="b0", choices=["b0", "b1", "b2"])
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--fusion-method", type=str, default="attention", 
                       choices=["add", "concat", "attention"])
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--spatial-radius", type=float, default=0.00005)
    
    # 測試參數
    parser.add_argument("--save-dir", type=str, default=None, help="結果保存目錄")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    
    # 推理參數
    parser.add_argument("--use-slide-inference", action="store_true", default=True,
                       help="使用滑窗推理")
    
    return parser.parse_args()


def main(args: Namespace):
    print("🧪 記憶增強版 GeoSegformer 測試開始")
    print("=" * 60)
    
    # 圖像尺寸設定
    image_size = 1080, 1920  # H x W
    crop_size = 512, 512
    stride = 384, 384
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用設備: {device}")
    
    # 載入類別定義
    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(categories)
    print(f"📊 類別數量: {num_categories}")
    
    # 設置GPS正規化（與訓練時保持一致）
    print(f"🌍 設置GPS正規化 (方法: {args.gps_norm_method})")
    gps_normalizer = setup_gps_normalization(
        args.train_gps_csv, 
        args.val_gps_csv,
        method=args.gps_norm_method
    )
    
    # 創建模型
    print(f"🚀 創建記憶增強版模型...")
    model = create_memory_enhanced_geo_segformer(
        num_classes=num_categories,
        model_size=args.model_size,
        feature_dim=args.feature_dim,
        fusion_method=args.fusion_method,
        memory_size=args.memory_size,
        spatial_radius=args.spatial_radius,
        memory_save_path=None  # 測試時不保存記憶庫
    ).to(device)
    
    # 載入檢查點
    print(f"📂 載入檢查點: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # 顯示模型參數統計
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 模型載入完成，參數量: {total_params:.2f}M")
    
    # 如果檢查點中有記憶庫統計，顯示訓練時的統計
    if "memory_stats" in checkpoint:
        memory_stats = checkpoint["memory_stats"]
        print(f"📈 訓練時記憶庫統計:")
        print(f"  總位置數: {memory_stats.get('total_locations', 'N/A')}")
        print(f"  總記憶數: {memory_stats.get('total_memories', 'N/A')}")
        print(f"  命中率: {memory_stats.get('hit_rate', 'N/A'):.4f}")
    
    # 創建保存目錄
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"📁 創建保存目錄: {args.save_dir}")
    
    # 數據變換（與訓練時的驗證集變換一致）
    transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        gps_normalizer,  # ⭐ 重要：必須與訓練時一致
        transform.Resize(image_size),
        transform.Normalize(),
    ]
    
    # 創建推理器
    if args.use_slide_inference:
        inferencer = GeoSlideInferencer(crop_size, stride, num_categories)
        print(f"🔍 使用滑窗推理 (crop: {crop_size}, stride: {stride})")
    else:
        # 如果不使用滑窗推理，直接調用模型
        inferencer = None
        print(f"🔍 使用直接推理")
    
    # 可視化工具
    if args.save_dir:
        visualizer = IdMapVisualizer(categories)
        img_saver = ImgSaver(args.save_dir, visualizer)
    
    # 損失函數和評估指標
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(num_categories, nan_to_num=0)
    
    # 創建測試數據集
    print(f"📁 載入測試數據集...")
    test_dataset = MemoryEnhancedGeoSegDataset(
        transforms=transforms,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        gps_csv=args.test_gps_csv,
        max_len=args.max_len,
    )
    
    test_dataloader = test_dataset.get_loader(
        batch_size=args.batch_size, 
        pin_memory=False, 
        num_workers=args.num_workers
    )
    
    print(f"📊 測試數據: {len(test_dataset)} 張影像")
    
    # 開始測試
    print(f"\n🧪 開始測試...")
    with Progress() as prog:
        with torch.no_grad():
            task = prog.add_task("Testing", total=len(test_dataloader))
            avg_loss = 0
            memory_weight_sum = 0
            
            for batch_idx, data in enumerate(test_dataloader):
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]
                gps = data["gps"].to(device)
                
                # 推理
                if inferencer is not None:
                    # 滑窗推理
                    pred = inferencer.inference(model, img, gps)
                else:
                    # 直接推理
                    outputs = model(img, gps, return_embeddings=False, update_memory=False)
                    pred = outputs['segmentation_logits']
                    memory_weight_sum += outputs.get('memory_weight', 0)
                
                # 計算損失
                loss = criterion(pred, ann)
                avg_loss += loss.item()
                
                # 計算評估指標
                metrics.compute_and_accum(pred.argmax(1), ann)
                
                # 保存結果
                if args.save_dir:
                    filenames = data.get("filename", [f"test_{batch_idx}"])
                    for i, (filename, p) in enumerate(zip(filenames, pred)):
                        save_name = f"{filename}_pred.png"
                        img_saver.save_pred(p[None, :], save_name)
                
                prog.update(task, advance=1)
            
            # 獲取最終結果
            result = metrics.get_and_reset()
            avg_loss /= len(test_dataloader)
            avg_memory_weight = memory_weight_sum / len(test_dataloader) if inferencer is None else 0
            
            prog.remove_task(task)
    
    # 創建結果表格
    print(f"\n📊 測試結果:")
    table = Table()
    table.add_column("Category")
    table.add_column("Acc")
    table.add_column("IoU")
    table.add_column("Dice")
    table.add_column("Fscore")
    table.add_column("Precision")
    table.add_column("Recall")
    
    for cat, acc, iou, dice, fs, pre, rec in zip(
        categories,
        result["Acc"],
        result["IoU"],
        result["Dice"],
        result["Fscore"],
        result["Precision"],
        result["Recall"],
    ):
        table.add_row(
            cat.name,
            "{:.5f}".format(acc),
            "{:.5f}".format(iou),
            "{:.5f}".format(dice),
            "{:.5f}".format(fs),
            "{:.5f}".format(pre),
            "{:.5f}".format(rec),
        )
    
    # 添加平均值
    table.add_row(
        "Avg.",
        "{:.5f}".format(result["Acc"].mean()),
        "{:.5f}".format(result["IoU"].mean()),
        "{:.5f}".format(result["Dice"].mean()),
        "{:.5f}".format(result["Fscore"].mean()),
        "{:.5f}".format(result["Precision"].mean()),
        "{:.5f}".format(result["Recall"].mean()),
    )
    
    print(table)
    print(f"\n📈 總體指標:")
    print(f"  平均損失: {avg_loss:.5f}")
    print(f"  平均IoU: {result['IoU'].mean():.5f}")
    print(f"  平均Dice: {result['Dice'].mean():.5f}")
    if avg_memory_weight > 0:
        print(f"  平均記憶權重: {avg_memory_weight:.4f}")
    
    # 獲取測試時記憶庫統計
    test_memory_stats = model.get_memory_stats()
    print(f"\n🧠 測試時記憶庫統計:")
    print(f"  總位置數: {test_memory_stats['total_locations']}")
    print(f"  總記憶數: {test_memory_stats['total_memories']}")
    print(f"  命中率: {test_memory_stats['hit_rate']:.4f}")
    print(f"  總查詢數: {test_memory_stats['total_queries']}")
    
    print(f"\n🎉 記憶增強版 GeoSegformer 測試完成！")
    if args.save_dir:
        print(f"📁 結果已保存到: {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)