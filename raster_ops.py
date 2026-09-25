import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Generator, Tuple

def compute_spectral_indices(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index (NDVI) with zero-division safety."""
    denom = nir + red
    ndvi = np.where(denom == 0, 0, (nir - red) / denom)
    return np.clip(ndvi, -1.0, 1.0)

def generate_window_tiles(
    src: rasterio.io.DatasetReader, 
    tile_size: int = 512
) -> Generator[Tuple[Window, rasterio.transform.Affine], None, None]:
    """Yields non-overlapping raster windows and local affine transforms for scalable inference."""
    for y in range(0, src.height, tile_size):
        for x in range(0, src.width, tile_size):
            window = Window(
                col_off=x,
                row_off=y,
                width=min(tile_size, src.width - x),
                height=min(tile_size, src.height - y),
            )
            transform = rasterio.windows.transform(window, src.transform)
            yield window, transform
