# 程式碼文件: `geotrain_v2_early_memory_v1.py`

## 1. 總覽 (Overview)

`geotrain_v2_early_memory_v1.py` 是一個功能強大且高度可配置的訓練腳本，用於**地理位置感知的語意分割 (Geo-aware Semantic Segmentation)** 任務。

其核心目標是訓練一個不僅能理解影像內容，還能利用 **GPS 地理位置資訊** 和 **歷史經驗 (外部記憶體庫)** 來進行更精準像素級分類的模型。這個腳本是為學術研究和實驗而設計的，提供了極大的靈活性。

### 核心思想

1.  **模型核心**: `MemoryEnhancedGeoSegformer` (記憶增強的地理位置 Segformer)。
    - **基礎模型**: 使用 `Segformer`，一個強大的基於 Transformer 的分割模型作為影像特徵提取器。
    - **GPS 融合**: 額外建立一個 GPS 編碼器，將 2D 的經緯度座標轉換為高維度特徵，並將其與影像特徵融合，讓模型具備地理位置感知能力。
    - **記憶體庫 (Memory Bank)**: 這是此模型最創新的部分。模型會建立一個外部「記憶體」，記錄在不同地理位置上出現過的典型視覺特徵。當模型處理一張新圖片時，它可以查詢這個記憶體庫，看看「過去在這個位置附近，我看過什麼樣的東西？」，並利用這些「記憶」來輔助當前的判斷。

2.  **雙重學習目標 (Dual Learning Objectives)**:
    - **分割任務**: 模型的主要目標，即準確地為每個像素分類。這部分由標準的 `CrossEntropyLoss` 來監督。
    - **對比學習任務**: 一個輔助目標，用來讓模型更好地理解地理位置。它透過 `ContrastiveLoss` 來強迫模型學習一個統一的特徵空間，在這個空間裡，一張圖片的「視覺特徵」和它的「地理位置特徵」是相互靠近的。

---

## 2. 程式碼結構詳解

- **`main_training_logic`**: 整個訓練流程的核心，包含了初始化、訓練迴圈、驗證、儲存等所有步驟。
- **`MemoryEnhancedGeoSegDataset`**: 負責準備模型所需的訓練資料，將圖片、分割標籤和 GPS 座標打包在一起。
- **`ContrastiveLoss` & `MemoryAwareContrastiveLoss`**: 實現對比學習的損失函數。後者是進階版，會考慮樣本間的物理距離，使學習更合理。
- **`EarlyStopping`**: 一個實用工具類，用於監控驗證指標，在模型性能不再提升時自動停止訓練，防止過擬合。
- **`parse_args()`**: 定義了所有可透過命令列配置的參數，是腳本靈活性的來源。
- **輔助函式**: 如 `save_checkpoint_geo` (儲存模型)、`create_training_summary_file` (儲存實驗配置) 等，極大提高了實驗的可追溯性和便利性。

---

## 3. 如何執行

要執行此訓練腳本，您需要在命令列中提供一系列參數。

### 完整指令範例

```bash
python tools/geotrain_v2_early_memory_v1.py \
  # 1. Positional Arguments (9個必要的位置參數)
  Seg_dataset_train/img \
  Seg_dataset_train/mask \
  Seg_dataset_test/img \
  Seg_dataset_test/mask \
  Seg_dataset_train/category.csv \
  Seg_dataset_train/gps.csv \
  Seg_dataset_test/gps.csv \
  200 \
  runs/seg_experiment_1 \
  # 2. Optional Arguments (可選的超參數)
  --model-size b2 \
  --feature-dim 256 \
  --fusion-method attention \
  --memory-size 20 \
  --spatial-radius 0.00005 \
  --seg-weight 1.0 \
  --contrastive-weight 0.1 \
  --batch-size 4 \
  --lr-backbone 1e-5 \
  --lr-head 1e-4 \
  --lr-gps 1e-4 \
  --early-stop \
  --patience 15
```

---

## 4. 參數詳細說明

### 位置參數 (Positional Arguments)
| # | 參數名稱 | 說明 |
|---|---|---|
| 1 | `train_img_dir` | 訓練集圖片資料夾路徑。 |
| 2 | `train_ann_dir` | 訓練集標籤 (mask) 資料夾路徑。 |
| 3 | `val_img_dir` | 驗證集圖片資料夾路徑。 |
| 4 | `val_ann_dir` | 驗證集標籤 (mask) 資料夾路徑。 |
| 5 | `category_csv` | 類別定義檔 (`.csv`) 的路徑。 |
| 6 | `train_gps_csv` | 訓練集 GPS 座標檔 (`.csv`) 的路徑。 |
| 7 | `val_gps_csv` | 驗證集 GPS 座標檔 (`.csv`) 的路徑。 |
| 8 | `max_epochs` | 最大訓練輪數。 |
| 9 | `logdir` | 儲存日誌、模型權重和視覺化結果的目錄。 |

### 可選參數 (Optional Arguments)

#### 模型架構
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--model-size` | Segformer Backbone 的大小，可選 `b0`, `b1`, `b2`。 | `b0` |
| `--feature-dim` | 模型內部特徵向量的維度。 | `512` |
| `--fusion-method` | 影像與 GPS 特徵的融合方式，可選 `add`, `concat`, `attention`。 | `attention` |

#### 記憶體庫 (Memory Bank)
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--memory-size` | 每個地理位置在記憶體庫中儲存的特徵數量。**設為 0 可完全禁用記憶體庫功能**。 | `20` |
| `--spatial-radius` | 定義記憶體庫查詢時的「地理鄰近範圍」。 | `0.00005` |
| `--memory-warmup-epochs` | 記憶體庫的「預熱」輪數。在此期間，記憶體庫只寫入不讀取。 | `3` |

#### 訓練策略與損失函數
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--seg-weight` | 分割損失 (`CrossEntropyLoss`) 的權重。 | `1.0` |
| `--contrastive-weight` | 對比學習損失 (`ContrastiveLoss`) 的權重。 | `0.05` |
| `--temperature` | 對比學習損失中的溫度係數。 | `0.07` |
| `--spatial-threshold` | `MemoryAwareContrastiveLoss` 中用來判斷地理位置是否相近的距離閾值。 | `0.15` |

#### 學習率 (Learning Rate)
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--lr-backbone` | Backbone (Segformer) 的學習率。 | `6e-5` |
| `--lr-head` | 分割頭 (Decoder Head) 的學習率。 | `6e-4` |
| `--lr-gps` | GPS 編碼器的學習率。 | `3e-4` |
| `--lr-memory` | 記憶體庫相關組件的學習率。 | `6e-4` |
| `--lr-fusion` | 特徵融合組件的學習率。 | `6e-4` |

#### 訓練流程控制
| 參數 | 說明 | 預設值 |
|---|---|---|
| `--batch-size` | 批次大小。 | `2` |
| `--num-workers` | 讀取資料的子行程數量。 | `0` |
| `--early-stop` | 啟用「提早結束」功能。 | `False` |
| `--patience` | 若啟用提早結束，模型性能連續多少輪沒有提升後就停止。 | `10` |
| `--monitor` | 提早結束所監控的指標，可選 `mIoU` 或 `loss`。 | `mIoU` |
| `--seed` | 全域隨機種子，用於保證實驗的可再現性。 | `42` |
| `--resume` | 從指定的 epoch 檢查點恢復訓練。 | `0` |
| `--keep-only-best` | 僅儲存驗證集上表現最好的模型，以節省硬碟空間。 | `False` |
