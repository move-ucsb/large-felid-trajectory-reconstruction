"""Dataset configuration for training your own model.

Edit this file when you want to train or adapt the model with your own high-resolution
tracking data.

Expected folder layout:

    data/raw/       high-resolution track CSV files
    data/raster/    optional raster covariates

For zero-shot use of the included pretrained model, you do not need to edit this file.
"""

from pathlib import Path
from carnivore_reconstruction import DatasetSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
RASTER_ROOT = PROJECT_ROOT / "data" / "raster"

# Replace this example with your own high-resolution training datasets.
DATASETS = [
    DatasetSpec(
        path=DATA_ROOT / "your_high_resolution_tracks.csv",
        dataset="Your_dataset_name",
        taxon="puma",  # e.g., "puma", "cougar", "tiger", "leopard", or your study taxon
        id_col="animal_id",
        time_col="timestamp",
        x_col="x",
        y_col="y",
        env_cols=[],  # optional columns already in the CSV, e.g. ["elevation", "slope"]
        # Optional raster covariates, if you want to train/evaluate with rasters:
        # raster_dir=RASTER_ROOT,
        # raster_paths={"elevation": RASTER_ROOT / "elevation.tif"},
        # raster_epsg=32610,
    ),

    # If your table has longitude/latitude instead of projected x/y, use this pattern:
    # DatasetSpec(
    #     path=DATA_ROOT / "your_lonlat_tracks.csv",
    #     dataset="Your_lonlat_dataset",
    #     taxon="cougar",
    #     id_col="animal_id",
    #     time_col="timestamp",
    #     x_col=None,
    #     y_col=None,
    #     lon_col="longitude",
    #     lat_col="latitude",
    #     epsg=32610,  # target projected CRS for distance calculations
    # ),
]
