"""
Annotated County Raster Utilities - PRISM
==========================================

Notes
-----
* Expects a single-band precipitation raster readable via rasterio.
* CRS/coords: geometries passed to `calculate_weighted_average` must already be in the raster's CRS.
* Pixels intersecting the geometry are masked via `rasterio.mask.mask` and an equal-weight average is computed
  over valid (non-nodata) cells. Replace the placeholder weights with your scheme if needed.
* GCS libraries (`gcsfs`, `google-cloud-storage`) are imported for surrounding I/O but not used inside these functions.
* Progress uses `print`; switch to `logging` for structured logs in production.
"""
from google.cloud import storage
import gcsfs
import os
import shutil
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union  # annotations only

# Function to calculate weighted average precipitation
def calculate_weighted_average(
    raster: "rasterio.io.DatasetReader",
    geometry: Union[Sequence[dict], Sequence["gpd.GeoSeries"], Sequence["gpd.GeoDataFrame"], Sequence["object"]],
) -> float:
    """
    Compute an equal-weight average of raster cell values inside the provided geometry.

    Parameters
    ----------
    raster : rasterio.io.DatasetReader
        An open rasterio dataset (assumed single-band for this function) with a valid `nodata` value.
        The geometry must be expressed in this raster's CRS.
    geometry : sequence
        Geometry (or geometries) to crop/mask with. Accepts GeoJSON-like mappings or shapely geometries
        that `rasterio.mask.mask` can consume.

    Returns
    -------
    float
        Mean of valid (non-nodata) pixels within the masked window; NaN if no valid pixels intersect.
    """
    try:
        # Mask the raster with the geometry
        out_image, out_transform = mask(raster, geometry, crop=True)
        values = out_image.flatten()
        values = values[values != raster.nodata]  # Exclude no data values

        if len(values) == 0:
            return np.nan  # Return NaN if no valid data
        else:
            weights = np.ones(values.shape)  # Default weights, adjust as needed
            return float(np.average(values, weights=weights))
    except ValueError as e:
        print(f"ValueError: {e}")
        return np.nan


# Define a function to calculate percentile based on the annual max distribution
def calculate_percentile_vectorized(
    precipitation_values: np.ndarray,
    county: Union[str, int],
    county_max_dist: Mapping[Union[str, int], Sequence[float]],
) -> np.ndarray:
    """
    Empirical percentile (fraction in [0, 1]) of values relative to a county's annual-max distribution.

    This ranks each `precipitation_values` element against the sorted annual maxima for `county`
    using `np.searchsorted` (right side), i.e., P = (# of maxima <= value) / N.

    Parameters
    ----------
    precipitation_values : np.ndarray
        Array of values to score (e.g., daily or rolling precipitation metrics).
    county : str | int
        County key used to index `county_max_dist`.
    county_max_dist : Mapping[str|int, Sequence[float]]
        Mapping from county -> sequence of annual maximum values. Must be non-empty.

    Returns
    -------
    np.ndarray
        Array of percentile fractions in [0, 1] (same shape as `precipitation_values`).
        Multiply by 100 if you want percent.
    """
    # Get the annual max distribution for the given county
    max_values = county_max_dist[county]
    # Convert the list of max values into a numpy array for faster percentile calculations
    max_values = np.array(max_values)
    # Calculate the percentile rank of the precipitation value within the max values
    return np.searchsorted(np.sort(max_values), precipitation_values, side='right') / len(max_values)
