# GPS_research
# 交接 Detection 部份 
# 程式碼分析: `true_gps_training_v4.2.py`

這份 Python 腳本的核心目標是將 GPS 地理位置資訊與 YOLOv10 物件偵測模型相結合，以提升偵測的準確性。它透過在模型的特定網路層中「注入」GPS 特徵來實現此目的。

以下是檔案中各個組件的詳細說明：

---

## 1. `class GPSDataManager` (GPS 資料管理器)

> **主要職責**:
> - 負責從 CSV 檔案中讀取、管理和提供 GPS 資料。
> - 計算 GPS 座標的統計數據（最大/最小值、平均值），用於後續的資料正規化。

#### 核心方法:
- `__init__(self, gps_csv_path)`:
  - 初始化時，會接收一個 GPS 資料的 CSV 檔案路徑。
  - 呼叫 `load_gps_data` 載入資料，並呼叫 `_compute_gps_statistics` 計算統計數據。
- `load_gps_data(self, gps_csv_path)`:
  - 使用 `pandas` 函式庫讀取 CSV 檔案。
  - 將資料轉換成一個字典 (`gps_mapping`)，其中 **key** 是圖片的檔名 (e.g., `00004602.jpg`)，**value** 是包含緯度 (lat) 和經度 (long) 的 PyTorch 張量 (Tensor)。
- `_compute_gps_statistics(self)`:
  - 計算所有 GPS 座標的緯度和經度的最大值、最小值和平均值。這些統計數據對於後續將 GPS 座標正規化到一個固定的範圍 (例如 `[-1, 1]`) 至關重要。
- `get_gps_stats(self)`:
  - 回傳計算好的統計數據字典。
- `get_gps_batch(self, image_paths)`:
  - 接收一個批次的圖片路徑列表。
  - 根據圖片檔名，從 `gps_mapping` 中查找對應的 GPS 座標。
  - 如果某張圖片找不到對應的 GPS 座標，會使用資料集的平均 GPS 座標作為預設值。
  - 最後，將這批次的 GPS 座標堆疊成一個張量並回傳。

---

## 2. `class ImprovedGPSEncoder` (改良版 GPS 編碼器)

> **主要職責**:
> - 將原始的 2D GPS 座標 (緯度, 經度) 轉換為高維度的特徵向量 (feature vector)。神經網路很難直接理解 `[40.7128, -74.0060]` 這樣的原始座標，因此需要將其編碼成更豐富的表示形式。

#### 核心邏輯:
- **正規化 (Normalization)**:
  - 在 `forward` 方法中，它首先使用 `GPSDataManager` 提供的 `min/max` 統計數據，將 GPS 座標正規化到 `[-1, 1]` 的範圍內。這有助於模型穩定訓練。
- **隨機傅立葉特徵 (Random Fourier Features, RFF)**:
  - 這是此編碼器的核心技術。它透過一系列固定的隨機投影 (`omega_list`) 和偏移 (`b_list`)，將 2D 的正規化座標映射到一個非常高維度的空間。這使得模型能夠從 GPS 數據中學習複雜的非線性關係。
- **正弦位置編碼 (Sinusoid Positional Encoding)**:
  - 類似於 Transformer 模型中的位置編碼，它會根據 GPS 座標的值，從一個預先計算好的正弦/餘弦表中查找一個對應的編碼。這為模型提供了關於 GPS 點在整個地理空間中「絕對位置」的資訊。
- **小型神經網路 (Encoder)**:
  - 將 RFF 編碼後的特徵再通過一個小型的全連接神經網路 (`nn.Sequential`)，進行進一步的特徵提取和轉換。
- **`forward(self, gps_coords)`**:
  - 執行上述所有步驟（正規化 -> RFF -> MLP -> 加上位置編碼），最終輸出一批高維度的 GPS 特徵向量。

---

## 3. `class AdaptiveGPSFeatureFusion` (自適應 GPS 特徵融合模組)

> **主要職責**:
> - 將 `ImprovedGPSEncoder` 產生的 GPS 特徵向量與 YOLO 模型中的影像特徵圖 (feature map) 進行融合。

#### 核心邏輯:
- `__init__(...)`:
  - 初始化時，會創建一個 `ImprovedGPSEncoder` 實例。
  - 根據 `fusion_type` 參數（預設為 `attention`）來決定融合策略。
- **注意力機制 (Attention Mechanism)**:
  - 當 `fusion_type` 為 `attention` 時，這是主要的融合方式。
  - **`gps_projection`**: 首先，將 GPS 特徵向量透過一個 1x1 卷積層，轉換成與影像特徵圖相同的通道數。
  - **`attention`**: 接著，讓影像特徵圖自己通過一個小型的卷積網路，產生一個「注意力權重圖」。這個圖的每個像素值介於 0 到 1 之間，代表了在該位置，GPS 資訊的重要性。
  - **融合**: 最後，將轉換後的 GPS 特徵與注意力權重圖相乘，再加回到原始的影像特徵圖上。公式可以簡化為：`新特徵 = 舊特徵 + (GPS特徵 * 注意力權重)`。這讓模型可以「自適應地」決定在影像的哪個區域（例如，天空區域可能不需要 GPS，而道路區域可能很需要）以及多大程度上使用 GPS 資訊。
- `forward(self, features, gps_coords)`:
  - 接收 YOLO 的影像特徵圖 (`features`) 和原始的 GPS 座標 (`gps_coords`)。
  - 呼叫內部的 `gps_encoder` 將 GPS 座標編碼。
  - 將編碼後的 GPS 特徵擴展成與影像特徵圖相同的空間維度 (H, W)。
  - 執行注意力融合機制，並回傳融合後的新特徵圖。

---

## 4. `def get_yolov10_fusion_layers(model_name)` (獲取 YOLOv10 融合層)

> **主要職責**:
> - 這是一個輔助函式，根據 YOLOv10 模型的大小（例如 `yolov10n`, `yolov10s`），回傳一個預先定義好的網路層索引列表。

#### 核心邏輯:
- 它包含一個字典，映射了不同模型尺寸到一組數字（層的索引）。
- GPS 特徵將會被注入到這些指定索引的網路層中。選擇這些層通常是基於經驗，目標是在模型的不同深度（淺層、中層、深層）都引入 GPS 資訊。

---

## 5. `class GPSEnhancedTrainer` (GPS 增強訓練器)

> **主要職責**:
> - 這是整個流程的總指揮。它封裝了標準的 `ultralytics.YOLO` 模型，並為其添加了 GPS 融合的功能。

#### 核心方法:
- `__init__(self, model_path, gps_manager)`:
  - 載入一個標準的 YOLO 模型。
  - 保存傳入的 `gps_manager`（訓練集和驗證集的資料管理器）。
  - 獲取要注入特徵的層索引 (`fusion_layers`)。
- `_register_hooks(self)`:
  - **這是實現 GPS 注入的關鍵**。它使用 PyTorch 的 `register_forward_hook` 機制。
  - "Hook" 就像一個掛鉤，可以掛在神經網路的任何一層上。當模型進行前向傳播（計算預測結果）時，數據流經這一層後，會觸發這個掛鉤函式。
  - 這個方法會為 `fusion_layers` 列表中的每一個層索引註冊一個 `hook_fn`。
  - `hook_fn` 的作用是：在該層的輸出（即影像特徵圖）上，呼叫 `AdaptiveGPSFeatureFusion` 模組來進行特徵融合，然後用融合後的新特徵取代原始輸出。
- `_set_batch(self, trainer, is_val=False)`:
  - 這是一個回呼函式 (Callback)，會在每個訓練或驗證批次開始時被 `ultralytics` 框架自動呼叫。
  - 它的作用是從當前批次的資料中獲取圖片路徑列表，然後呼叫 `set_gps_batch` 來準備好對應的 GPS 數據 (`self.current_gps_batch`)，以供 `hook_fn` 使用。
- `set_gps_batch(self, image_paths)`:
  - 呼叫 `gps_manager` 的 `get_gps_batch` 方法來獲取當前批次的 GPS 座標。
- `train(self, **kwargs)`, `validate(self, **kwargs)`, `predict(self, ...)`:
  - 這些是公開的介面方法。
  - 它們在內部首先呼叫 `_register_hooks` 來設置好特徵注入點。
  - 接著，註冊 `_set_batch` 回呼函式，以確保每個批次都有對應的 GPS 數據。
  - 最後，呼叫 `ultralytics.YOLO` 模型對應的 `train`, `val`, `predict` 方法來執行標準的訓練、驗證或預測流程。

---

## 6. `def main()` (主函式)

> **主要職責**:
> - 程式的進入點。
> - 解析命令列參數（例如資料集路徑、模型路徑、訓練輪數等）。
> - 根據參數初始化 `GPSDataManager` 和 `GPSEnhancedTrainer`。
> - 根據使用者是想進行訓練 (`--train`)、純驗證 (`--val`) 還是預測 (`--predict`)，來呼叫 `GPSEnhancedTrainer` 的相應方法。
> - 在驗證結束後，會印出詳細的評估指標（如 mAP50, mAP50-95），並將每個類別的結果儲存到一個 CSV 檔案中。

---
### 總結

整個工作流程如下：
1.  `main` 函式啟動，解析參數並建立 `GPSDataManager` 和 `GPSEnhancedTrainer`。
2.  使用者執行訓練、驗證或預測。`GPSEnhancedTrainer` 呼叫 `_register_hooks` 在 YOLO 模型的骨幹網路中設定好「掛鉤」。
3.  當一個批次的圖片送入模型時：
    a. `_set_batch` 回呼函式被觸發，從 `GPSDataManager` 獲取這批圖片對應的 GPS 座標。
    b. 模型進行前向傳播，當數據流經被掛鉤的層時，`hook_fn` 被觸發。
    c. `hook_fn` 內的 `AdaptiveGPSFeatureFusion` 模組將 GPS 座標編碼並與影像特徵融合。
    d. 融合後的新特徵取代原始特徵，繼續在網路中傳播。
4.  模型最終基於融合了 GPS 資訊的特徵來進行物件偵測，從而可能達到比僅使用影像更好的效果。
