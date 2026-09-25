import numpy as np
import rasterio
import torch
from src.spatial.raster_ops import generate_window_tiles, compute_spectral_indices

def run_spatial_inference(
    input_geotiff: str,
    output_geotiff: str,
    model: torch.nn.Module,
    device: str = "cpu"
):
    """Integrates deep learning model prediction with spatial geospatial writing."""
    model.eval()
    model.to(device)

    with rasterio.open(input_geotiff) as src:
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": 0
        })

        with rasterio.open(output_geotiff, "w", **meta) as dst:
            for window, _ in generate_window_tiles(src, tile_size=512):
                # Read specific multispectral bands (e.g., B4: Red, B8: NIR)
                data = src.read(window=window).astype(np.float32)
                
                # Check for empty/padded tiles
                if np.all(data == src.nodata):
                    dst.write(np.zeros((1, window.height, window.width), dtype=np.uint8), window=window)
                    continue

                # Normalization and Tensor formatting: [C, H, W] -> [1, C, H, W]
                tensor_input = torch.from_numpy(data / 10000.0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor_input)
                    predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                dst.write(predictions, indexes=1, window=window)
