# Geospatial AI: Multi-Temporal Semantic Segmentation Pipeline

An end-to-end Machine Learning pipeline designed for scalable Earth observation analysis. This framework integrates Python spatial raster processing with deep learning (PyTorch U-Net) and ensemble methods to perform automated land cover and crop-type classification on multi-sensor satellite imagery.

## 🚀 Key Features
- **Spatial Feature Engineering:** Automated derivation of indices (NDVI, NDWI, EVI) from multispectral rasters via `rasterio` and `numpy`.
- **Spatial Block Cross-Validation:** Eliminates spatial autocorrelation leakage during model validation.
- **Memory-Efficient Tiled Inference:** Processes multi-gigabyte GeoTIFFs using windowed matrix streaming.
- **GIS Interoperability:** Automatically aligns Coordinate Reference Systems (CRS) and preserves spatial affine transforms in model outputs.

## 🛠️ Tech Stack
- **Languages & Core:** Python 3.11, NumPy, SciPy
- **GIS & Remote Sensing:** Rasterio, GeoPandas, Shapely, GDAL, Fiona
- **Machine Learning:** PyTorch, Scikit-Learn, LightGBM
- **Automation & Quality:** Pytest, Flake8, Docker

## 📊 Benchmark Results

| Model Architecture | Spatial F1-Score | Overall Accuracy | Inference Speed (km²/sec) |
|:-------------------|:-----------------|:-----------------|:--------------------------|
| Random Forest      | 0.82             | 85.1%            | 12.4                      |
| U-Net (ResNet-34)  | 0.91             | 93.4%            | 4.2 (NVIDIA T4)           |

## ⚡ Quickstart

### 1. Installation
\`\`\`bash
git clone https://github.com/your-username/geospatial-ml-landcover.git
cd geospatial-ml-landcover
conda env create -f environment.yml
conda activate geo-ml
\`\`\`

### 2. Run Spatial Feature Generation & Training
\`\`\`bash
python -m src.pipeline.train --config configs/config.yaml
\`\`\`

### 3. Predict & Export Classified GeoTIFF
\`\`\`bash
python -m src.pipeline.inference --input data/raw/sentinel_scene.tif --output outputs/classified_map.tif
\`\`\`
