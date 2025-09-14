import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as t
from torchvision.io import read_image
from torchvision.transforms import functional as tf


class Transform(ABC):
    @abstractmethod
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


class LoadImg(Transform):
    def __init__(self, mode: str = "rgb") -> None:
        assert mode in ["rgb", "bgr"]
        self.to_bgr = mode == "bgr"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "img_path" in data, "No path given to load image!"

        img = read_image(data["img_path"])
        if self.to_bgr:
            img = img[[2, 1, 0], :]
        data["img"] = img.float() / 255.0
        return data


class LoadAnn(Transform):
    def __init__(self, ignored_index: Optional[list[int]] = None) -> None:
        self.ignored_index = ignored_index

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "ann_path" in data, "No path given to load annotation!"

        img = Image.open(data["ann_path"])
        data["ann"] = tf.pil_to_tensor(img).long()
        return data


class Resize(Transform):
    def __init__(self, size: tuple[int, int]) -> None:
        self.img_transform = t.Resize(size, antialias=True)
        self.ann_transform = t.Resize(
            size, interpolation=tf.InterpolationMode.NEAREST, antialias=True
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.img_transform(data["img"])
        if "ann" in data:
            data["ann"] = self.ann_transform(data["ann"])
        return data


class RandomResizeCrop:
    def __init__(
        self,
        image_scale: tuple[int, int],
        scale: tuple[float, float],
        crop_size: tuple[int, int],
        antialias: bool = True,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = np.array(crop_size)
        self.antialias = antialias

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_random_crop(self, scaled_height, scaled_width, crop_size):
        crop_y0 = random.randint(0, scaled_height - crop_size[0])
        crop_x0 = random.randint(0, scaled_width - crop_size[1])

        return crop_y0, crop_x0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        height, width = self.get_random_size()
        y0, x0 = self.get_random_crop(height, width, self.crop_size)

        if "img" in data:
            data["img"] = tf.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        if "ann" in data:
            data["ann"] = tf.resize(
                data["ann"],
                (height, width),
                interpolation=tf.InterpolationMode.NEAREST,
            )[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return data


class RandomHorizontalFlip(Transform):
    def __init__(self) -> None:
        self.flip = t.RandomHorizontalFlip()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.flip(data["img"])
        if "ann" in data:
            data["ann"] = self.flip(data["ann"])
        return data


class ColorJitter:
    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ) -> None:
        self.jitter = t.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.jitter(data["img"])

        return data


class Normalize(Transform):
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.normalize = t.Normalize(mean, std, True)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.normalize(data["img"])
        return data
        
class RandomErasing:
    """使用torchvision內建的RandomErasing"""
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
        inplace: bool = False
    ):
        from torchvision.transforms import RandomErasing as TorchRandomErasing
        self.random_erase = TorchRandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=inplace
        )
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.random_erase(data["img"])
        return data

class Sequence(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


class GPSToRFF(Transform):
    """
    把 data['gps'] = Tensor([lat,lon]) 經 Random Fourier Features 轉成高維向量
    使用方法：在 transform pipeline 裡加入 GPSToRFF(rff_dim=256, sigma=10.0)
    會在 data["gps_embed"] 加入 RFF 高維特徵
    """
    def __init__(self, rff_dim: int = 256, sigma: float = 1.0) -> None:
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.omega = torch.randn((rff_dim//2, 2)) / sigma
        self.b = 2 * math.pi * torch.rand(rff_dim//2)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        gps: torch.Tensor = data.get("gps")  # shape: (2,)
        if gps is None:
            raise ValueError("No gps found in data dict!")
        proj = gps.unsqueeze(0) @ self.omega.T  # (1, rff_dim//2)
        y = proj + self.b  # (1, rff_dim//2)
        rff = torch.cat([torch.cos(y), torch.sin(y)], dim=-1)  # (1, rff_dim)
        data["gps_embed"] = rff.squeeze(0)  # shape: (rff_dim,)
        return data


class GPSNormalize(Transform):
    """
    GPS座標正規化Transform
    將GPS座標正規化到[-1, 1]範圍
    """
    def __init__(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float]):
        """
        Args:
            lat_range: (min_lat, max_lat) 緯度範圍
            lon_range: (min_lon, max_lon) 經度範圍
        """
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        
        print(f"✅ GPSNormalize initialized:")
        print(f"  Lat range: [{self.lat_min:.6f}, {self.lat_max:.6f}]")
        print(f"  Lon range: [{self.lon_min:.6f}, {self.lon_max:.6f}]")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if 'gps' in data:
            gps = data['gps']
            if isinstance(gps, torch.Tensor):
                lat, lon = gps[0].item(), gps[1].item()
            else:
                lat, lon = gps[0], gps[1]
            
            # 正規化到[0, 1]再映射到[-1, 1]
            lat_norm = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1
            lon_norm = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
            
            # 確保在範圍內
            lat_norm = max(-1, min(1, lat_norm))
            lon_norm = max(-1, min(1, lon_norm))
            
            data['gps'] = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)
            
        return data

    @staticmethod
    def compute_ranges_from_csv(csv_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        從CSV文件計算GPS範圍
        
        Args:
            csv_path: GPS CSV文件路徑
            
        Returns:
            ((min_lat, max_lat), (min_lon, max_lon))
        """
        import pandas as pd
        
        gps_data = pd.read_csv(csv_path)
        
        lat_min = gps_data['lat'].min()
        lat_max = gps_data['lat'].max()
        lon_min = gps_data['long'].min()
        lon_max = gps_data['long'].max()
        
        print(f"📊 從 {csv_path} 計算的GPS範圍:")
        print(f"  緯度: [{lat_min:.6f}, {lat_max:.6f}] (範圍: {lat_max-lat_min:.6f})")
        print(f"  經度: [{lon_min:.6f}, {lon_max:.6f}] (範圍: {lon_max-lon_min:.6f})")
        
        return (lat_min, lat_max), (lon_min, lon_max)


class GPSStandardize(Transform):
    """
    GPS座標標準化Transform (Z-score normalization)
    將GPS座標標準化為均值0，標準差1
    """
    def __init__(self, lat_stats: Tuple[float, float], lon_stats: Tuple[float, float]):
        """
        Args:
            lat_stats: (mean_lat, std_lat) 緯度統計
            lon_stats: (mean_lon, std_lon) 經度統計
        """
        self.lat_mean, self.lat_std = lat_stats
        self.lon_mean, self.lon_std = lon_stats
        
        print(f"✅ GPSStandardize initialized:")
        print(f"  Lat: mean={self.lat_mean:.6f}, std={self.lat_std:.6f}")
        print(f"  Lon: mean={self.lon_mean:.6f}, std={self.lon_std:.6f}")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if 'gps' in data:
            gps = data['gps']
            if isinstance(gps, torch.Tensor):
                lat, lon = gps[0].item(), gps[1].item()
            else:
                lat, lon = gps[0], gps[1]
            
            # Z-score標準化
            lat_norm = (lat - self.lat_mean) / (self.lat_std + 1e-8)
            lon_norm = (lon - self.lon_mean) / (self.lon_std + 1e-8)
            
            data['gps'] = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)
            
        return data

    @staticmethod
    def compute_stats_from_csv(csv_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        從CSV文件計算GPS統計信息
        
        Returns:
            ((mean_lat, std_lat), (mean_lon, std_lon))
        """
        import pandas as pd
        
        gps_data = pd.read_csv(csv_path)
        
        lat_mean = gps_data['lat'].mean()
        lat_std = gps_data['lat'].std()
        lon_mean = gps_data['long'].mean()
        lon_std = gps_data['long'].std()
        
        print(f"📊 從 {csv_path} 計算的GPS統計:")
        print(f"  緯度: mean={lat_mean:.6f}, std={lat_std:.6f}")
        print(f"  經度: mean={lon_mean:.6f}, std={lon_std:.6f}")
        
        return (lat_mean, lat_std), (lon_mean, lon_std)


# 測試GPS正規化效果的函數
def test_gps_normalization():
    """測試GPS正規化效果"""
    
    # 使用實際GPS範圍（基於你的數據）
    lat_range = (-0.001207, -0.000212)
    lon_range = (-0.000407, 0.000924)
    
    gps_normalizer = GPSNormalize(lat_range, lon_range)
    
    # 測試數據
    test_gps_coords = [
        [-0.001057, -0.000368],  # 實際數據
        [-0.000738, -0.000405],
        [-0.000545, -0.000406],
        [-0.001207, -0.000407],  # 邊界值
        [-0.000212, 0.000924],
    ]
    
    print("🧪 GPS正規化測試:")
    print("原始座標 → 正規化座標")
    print("-" * 35)
    
    for i, coords in enumerate(test_gps_coords):
        data = {'gps': torch.tensor(coords, dtype=torch.float32)}
        normalized_data = gps_normalizer(data)
        
        original = coords
        normalized = normalized_data['gps'].tolist()
        
        print(f"{i+1}. [{original[0]:8.6f}, {original[1]:8.6f}] → [{normalized[0]:6.3f}, {normalized[1]:6.3f}]")
    
    print("\n✅ 正規化後的GPS座標都在[-1, 1]範圍內")
    print("✅ 可以更有效地被RFF編碼")


if __name__ == "__main__":
    from torchvision.io import write_jpeg

    # 原有的測試代碼
    transforms = [LoadImg(), Resize((512, 512))]

    data = {
        "img_path": "./data/rlmd/clear/images/train/10.jpg",
    }

    for transform in transforms:
        data = transform(data)

    write_jpeg(data["img"], "test.png")
    
    # 測試GPS正規化
    print("\n" + "="*50)
    test_gps_normalization()