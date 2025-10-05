# GPS-Enhanced Computer Vision: A Research Framework

## 1. 研究總覽 (Research Overview)

本專案是一個專注於將**地理位置資訊 (GPS)** 融入深度學習電腦視覺模型的研究框架。我們的核心假設是：透過為模型提供地理空間上下文 (Geo-spatial Context)，可以顯著提升其在特定視覺任務上的表現，尤其是在處理視覺外觀相似但地理位置不同的場景時。

本研究框架主要探索兩條並行的技術路線：

1.  **物件偵測 (Object Detection)**:
    - **目標**: 增強 YOLOv10 模型，使其在偵測物件時能參考其地理位置。
    - **應用場景**: 例如，在特定區域（如工業區 vs. 住宅區）對相同物件（如車輛）進行更精細的分類或行為預測。

2.  **語意分割 (Semantic Segmentation)**:
    - **目標**: 增強 Segformer 模型，不僅融入 GPS 資訊，更引入一個**外部記憶體庫 (Memory Bank)** 的創新機制。
    - **核心創新**: 模型不僅能感知「我在哪裡」，還能查詢「過去在這一帶我看過什麼？」。這種「歷史經驗」機制對於處理光線變化、季節更迭等複雜場景具有巨大潛力。

## 2. 核心方法論 (Core Methodologies)

本專案採用了多種先進技術來實現地理與視覺的融合：

-   **GPS 特徵編碼 (GPS Feature Encoding)**:
    我們沒有直接使用原始的經緯度座標，而是透過**隨機傅立葉特徵 (Random Fourier Features, RFF)** 等技術將 2D 座標轉換為高維度的特徵向量，使其更容易被神經網路理解和利用。

-   **多模態特徵融合 (Multi-Modal Fusion)**:
    專案探索了多種融合策略（如注意力機制 `Attention`），以智慧地將視覺特徵與地理特徵結合，讓模型可以自適應地決定在影像的不同區域應該賦予地理資訊多大的權重。

-   **記憶增強網路 (Memory-Augmented Networks)**:
    在語意分割任務中，我們引入了外部記憶體庫。這個記憶體庫儲存了大量「地理位置 -> 視覺特徵」的對應關係，使模型具備了長期的、跨場景的記憶能力。

-   **對比學習 (Contrastive Learning)**:
    作為一項輔助學習任務，我們使用對比學習來對齊視覺和地理兩種模態的特征空間。這鼓勵模型學習到一個更有意義的、統一的地理-視覺聯合表示。

## 3. 環境安裝 (Environment Setup)

本專案使用 Conda 進行環境管理。請依照以下步驟安裝並啟用專案所需的環境。

1.  **安裝 Conda**:
    如果您尚未安裝 Conda，請參考官方文件進行安裝 ([Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 或 [Anaconda](https://www.anaconda.com/download))。

2.  **從 `environment.yml` 建立環境**:
    本專案根目錄下已包含 `environment.yml` 檔案。請執行以下指令來建立名為 `gps_research` 的 Conda 環境並安裝所有相依套件。

    ```bash
    conda env create -f environment.yml
    ```

3.  **啟用環境**:
    每次要執行專案中的程式碼前，請務必先啟用此環境。

    ```bash
    conda activate gps_research
    ```

## 4. 專案結構與重點程式 (Project Structure & Key Programs)

以下是本專案中最重要的目錄和腳本：

-   `environment.yml`:
    - **Conda 環境定義檔**，包含了所有必要的 Python 套件與版本，是再現本專案研究環境的基礎。

-   `交接/Detection/`: **物件偵測研究目錄**
    - `true_gps_training_v4.2.py`: **核心訓練腳本**。用於訓練、驗證和測試 GPS 增強的 YOLOv10 模型。
    - `README.hd`: 該目錄下的說明文件，包含更詳細的指令與參數說明。

-   `交接/Seg/`: **語意分割研究目錄**
    - `tools/geotrain_v2_early_memory_v1.py`: **核心訓練腳本**。用於訓練、驗證結合了 GPS 和記憶體庫的 Segformer 模型。此腳本高度可配置，是進行消融實驗的關鍵。
    - `engine/geo_v2_memory.py`: **核心模型定義**。包含了 `MemoryEnhancedGeoSegformer` 的完整實作。
    - `README.md`: 該目錄下的說明文件，對分割任務的參數和執行方式有更深入的介紹。

## 5. 快速開始 (Getting Started)

在成功安裝並啟用 Conda 環境後，您可以參考以下指令快速開始您的第一個實驗。

### 執行物件偵測訓練

```bash
# 進入 Detection 工作目錄
cd 交接/Detection

# 執行訓練
  python true_gps_training_v4.2.py   
  --data data_gps.yaml \   
  --gps-train-csv Dete_dataset/train_gps.csv \
  --gps-val-csv Dete_dataset/val_gps.csv \
  --model yolov10n.pt \
  --epochs 2000 \
  --batch-size 64 \
  --imgsz 640
```

### 執行語意分割訓練

```bash
# 回到專案根目錄
cd ../..

# 執行訓練
python 交接/Seg/tools/geotrain_v2_early_memory_v1.py \
  交接/Seg/Seg_dataset_train/img \
  交接/Seg/Seg_dataset_train/mask \
  交接/Seg/Seg_dataset_test/img \
  交接/Seg/Seg_dataset_test/mask \
  交接/Seg/Seg_dataset_train/category.csv \
  交接/Seg/Seg_dataset_train/gps.csv \
  交接/Seg/Seg_dataset_test/gps.csv \
  200 \
  runs/seg_experiment_1 \
  --model-size b2 \
  --memory-size 20 \
  --batch-size 4
```

> **提示**: 關於每個任務更詳細的參數配置和說明，請參考對應子目錄 (`Detection` 和 `Seg`) 下的 `README` 文件。
