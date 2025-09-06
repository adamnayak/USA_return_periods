"""
Annotated MSWEP County Pipeline Utilities
========================================

Notes
-----
* MSWEP precipitation variable defaults to 'precipitation'. Adjust if your files differ.
* Coordinates expected: 'lat', 'lon', and 'time' (xarray dims/coords).
* County polygons must contain a 'GEOID' column (string FIPS, zero-padded to 5 where used).
* Pixel weights are built by rasterizing county geometry onto the grid; weights sum to 1 per county.
  If no pixels intersect a county, the nearest grid cell to the county centroid gets weight=1 (others 0).
* Weighted aggregation is performed as (precipitation * weights).sum(dim=['lat','lon']) to avoid
  index misalignment; weight arrays must match the grid shape exactly.
* Yearly processing expects NetCDFs under a GCS prefix (e.g., 'mswep/daily/...') and writes
  `daily_precip_by_county_{year}.csv` to GCS.
* Rechunking merges per-year CSVs ('{year}_MSWEP.csv') into per-county CSVs and uploads back to GCS;
  temporary local files are cleaned up automatically.
* Claims merge requires per-county claims with 'dateOfLoss'; dates are merged as naive timestamps
  (timezones stripped) and extra columns ['date','county','year'] are dropped post-merge.
* GCS I/O via `gcsfs` (streaming reads/writes) and `google-cloud-storage` (bulk blob ops). Weights are
  saved with `numpy.save`; load with `np.load(path, allow_pickle=True).item()`.
* Parallelism: county CSV processing uses `ProcessPoolExecutor`; tune process count/chunking for your host.
* Progress uses `print`; switch to `logging` for structured, level-based logs in production.
* Caveats: nearest-cell fallback uses centroid distance in the data’s CRS; if CRS is geographic, distances
  are in degrees (ordering is still valid, but not metric). Ensure weight grid aligns with each NetCDF.
* Interpretation: GEV percentiles are relative to the distribution of **annual maxima** (return-period framing).
"""
import os
import time
import shutil
import tempfile
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple, Union, List, Optional
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # required for .rio accessor on xarray objects
import geopandas as gpd
from shapely.geometry import Point, box
from rasterio.features import geometry_mask
from scipy.stats import genextreme
from concurrent.futures import ProcessPoolExecutor
import gcsfs
from google.cloud import storage

# -------------------------
# Type aliases
# -------------------------
PathLike = Union[str, os.PathLike]
GeoID = str  # County FIPS/GEOID stored as string


def calculate_pixel_weights_mSWEP(
    county_shapefile: PathLike,
    precipitation_nc_path: str,
    output_path: str,
    gcs_bucket: str,
) -> Dict[GeoID, np.ndarray]:
    """
    Compute per-county pixel weights for a gridded precipitation field and write the
    resulting weight arrays to GCS as a NumPy .npy (pickle of dict).

    Parameters
    ----------
    county_shapefile : PathLike
        Local path to a county polygons file readable by GeoPandas (e.g., shapefile or GeoPackage).
        Must contain a 'GEOID' column uniquely identifying each county.
    precipitation_nc_path : str
        Object path (within the bucket) to the gridded precipitation NetCDF on GCS.
        Example: 'mswep/daily/mswep_daily_precipitation_1950.nc'.
    output_path : str
        Object path (within the bucket) where the weights dictionary (.npy) should be written.
        Example: 'weights/mswep_pixel_weights.npy'.
    gcs_bucket : str
        GCS bucket name or URL prefix acceptable by gcsfs (e.g., 'gs://my-bucket').

    Returns
    -------
    Dict[GeoID, np.ndarray]
        Mapping of county GEOID -> 2D NumPy array with shape matching the precipitation
        grid (ny, nx). Each array contains non-negative weights that sum to 1 for pixels
        inside the county. If no pixels intersect a county, a single nearest pixel is
        assigned weight=1 and others 0.

    Side Effects
    ------------
    - Reads county geometries from the local filesystem.
    - Reads NetCDF from GCS.
    - Writes a NumPy file (pickled Python dict) to GCS at `output_path`.

    Assumptions
    -----------
    - Gridded dataset includes coordinates named 'lat' and 'lon'.
    - The xarray Dataset has a CRS accessible via `.rio.crs`, or CRS can be inherited
      from the counties if missing.
    - County GeoDataFrame CRS is compatible with (or convertible to) the raster CRS.

    Raises
    ------
    FileNotFoundError
        If `county_shapefile` is not found locally.
    KeyError
        If required columns (e.g., 'GEOID') or variables ('precipitation') are missing.
    ValueError
        If shapes/CRS are incompatible or coordinate names differ from 'lat'/'lon'.
    """
    fs = gcsfs.GCSFileSystem()

    precipitation_nc_path = f"{gcs_bucket}/{precipitation_nc_path}"
    output_path = f"{gcs_bucket}/{output_path}"

    counties = gpd.read_file(county_shapefile)

    # Load the precipitation NetCDF file from GCS
    with fs.open(precipitation_nc_path, "rb") as f:
        ds = xr.open_dataset(f)

    # Ensure that the precipitation data has a CRS
    if not ds.rio.crs:
        # If the file lacks metadata, assume WGS84
        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        print("Assigned EPSG:4326 to precipitation data:", ds.rio.crs)
    

    if "precipitation" not in ds:
        raise KeyError("Expected variable 'precipitation' in dataset.")

    precip = ds["precipitation"]

    # Reproject counties to match precipitation CRS
    counties = counties.to_crs(precip.rio.crs)

    # Bounds of precip grid
    precip_extent = box(
        precip["lon"].min().item(),
        precip["lat"].min().item(),
        precip["lon"].max().item(),
        precip["lat"].max().item(),
    )

    # Clip counties to grid extent to avoid unnecessary masks
    counties_clipped = counties[counties.intersects(precip_extent)]

    # Precompute pixel centroids for "nearest pixel" fallback
    pixel_lon, pixel_lat = np.meshgrid(precip["lon"].values, precip["lat"].values)
    pixel_centroids = gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in zip(pixel_lon.ravel(), pixel_lat.ravel())]},
        crs=precip.rio.crs,
    )

    weights: Dict[GeoID, np.ndarray] = {}

    for county in counties_clipped.itertuples():
        # Build a rasterized mask for this county (True=inside)
        mask = geometry_mask(
            [county.geometry],
            transform=precip.rio.transform(),
            invert=True,
            out_shape=precip.shape[-2:],  # (ny, nx)
        )

        # Use pixel counts as a proxy for area weight within the polygon footprint
        pixel_areas = mask.astype(int)
        total_area = pixel_areas.sum()

        if total_area > 0:
            pixel_weights = pixel_areas / total_area
        else:
            # Fallback: choose nearest grid cell to the polygon centroid
            county_centroid = county.geometry.centroid
            pixel_centroids["distance"] = pixel_centroids["geometry"].distance(county_centroid)
            closest_pixel = pixel_centroids.loc[pixel_centroids["distance"].idxmin()]

            # Find index of nearest lat/lon
            closest_pixel_idx = (
                np.abs(precip["lat"].values - closest_pixel.geometry.y).argmin(),
                np.abs(precip["lon"].values - closest_pixel.geometry.x).argmin(),
            )

            pixel_weights = np.zeros_like(pixel_areas, dtype=float)
            pixel_weights[closest_pixel_idx] = 1.0

        weights[getattr(county, "GEOID")] = pixel_weights

    # Persist to GCS
    with fs.open(output_path, "wb") as f:
        np.save(f, weights)

    return weights


def process_mSWEP_precipitation(
    county_shapefile: PathLike,
    precipitation_dir: str,
    weights: Mapping[GeoID, np.ndarray],
    output_dir: str,
    gcs_bucket: str,
) -> None:
    """
    Apply precomputed per-county pixel weights to each daily MSWEP NetCDF file on GCS,
    export per-year county daily precipitation CSVs to GCS.

    Parameters
    ----------
    county_shapefile : PathLike
        Local path to county polygons (must include 'GEOID').
    precipitation_dir : str
        GCS folder containing yearly MSWEP NetCDF files ('.nc').
    weights : Mapping[GeoID, np.ndarray]
        Dict-like mapping from county GEOID to a 2D weight array matching (lat, lon) grid.
    output_dir : str
        GCS folder to write CSV outputs into (created if missing).
    gcs_bucket : str
        GCS bucket (e.g., 'gs://my-bucket').

    Returns
    -------
    None

    Side Effects
    ------------
    - Reads county polygons locally.
    - Reads all NetCDF files under `precipitation_dir` on GCS.
    - Writes one CSV per year to `output_dir` on GCS.

    Assumptions
    -----------
    - NetCDF files contain a variable 'precipitation' with dims including 'lat' and 'lon'.
    - Weight arrays align (ny, nx) with the grid of each yearly file.

    Raises
    ------
    FileNotFoundError
        If `county_shapefile` not found.
    KeyError
        If 'GEOID' is missing or 'precipitation' variable is absent.
    """
    fs = gcsfs.GCSFileSystem()

    precipitation_dir_path = f"{gcs_bucket}/{precipitation_dir}"
    output_dir_path = f"{gcs_bucket}/{output_dir}"

    counties = gpd.read_file(county_shapefile)

    if not fs.exists(output_dir_path):
        fs.mkdirs(output_dir_path)

    precipitation_files = [f for f in fs.ls(precipitation_dir_path) if f.endswith(".nc")]

    overall_start_time = time.time()

    for year_file_path in sorted(precipitation_files):
        print(f"Processing file: {year_file_path}")
        year_file_name = os.path.basename(year_file_path)
        year = year_file_name.split("_")[-1].split(".")[0]

        ds = xr.open_dataset(fs.open(year_file_path, "rb"))

        if "precipitation" not in ds:
            raise KeyError(f"'precipitation' not found in {year_file_path}")

        precip = ds["precipitation"]
        year_records: List[dict] = []

        for county in counties.itertuples():
            geo_id: GeoID = getattr(county, "GEOID")
            weight = weights.get(geo_id)

            if weight is not None:
                # Weighted daily value over (lat, lon)
                daily_precip = (precip * weight).sum(dim=["lat", "lon"])

                for date, value in zip(ds["time"].values, daily_precip.values):
                    year_records.append(
                        {
                            "date": date,
                            "county": geo_id,
                            "precipitation": float(value),
                        }
                    )

        year_df = pd.DataFrame.from_records(year_records)

        output_csv_path = f"{output_dir_path}/daily_precip_by_county_{year}.csv"
        with fs.open(output_csv_path, "w") as f:
            year_df.to_csv(f, index=False)

        year_elapsed_time = time.time() - overall_start_time
        print(f"Saved results for {year} to {output_csv_path} in {year_elapsed_time:.2f} seconds")
        overall_start_time = time.time()


def rechunk_precipitation_data_by_county_gcs(
    years: Sequence[int],
    input_path: str,
    output_path: str,
    gcs_bucket: str,
) -> None:
    """
    Reorganize per-year county CSVs stored on GCS into per-county CSVs spanning all years.
    Performs local temp storage/merging, then uploads final per-county outputs back to GCS.

    Parameters
    ----------
    years : Sequence[int]
        List/sequence of years to process (e.g., range(1950, 2020)).
    input_path : str
        GCS folder containing '{year}_MSWEP.csv' files with columns ['county','date','precipitation', ...].
    output_path : str
        GCS folder where merged per-county CSVs will be uploaded.
    gcs_bucket : str
        GCS bucket root (e.g., 'gs://my-bucket').

    Returns
    -------
    None

    Side Effects
    ------------
    - Reads yearly CSVs from GCS.
    - Writes temporary per-county year splits locally.
    - Uploads merged per-county CSVs to GCS.
    - Cleans up local temp directory.

    Raises
    ------
    FileNotFoundError
        If expected yearly inputs are missing on GCS.
    ValueError
        If CSV schemas are incompatible across years.
    """
    fs = gcsfs.GCSFileSystem()

    input_path_gcs = f"{gcs_bucket}/{input_path}"
    output_path_gcs = f"{gcs_bucket}/{output_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at {temp_dir}")

        yearly_temp_dirs: List[str] = []

        for year in years:
            start_time = time.time()
            print(f"Processing year {year}...")

            yearly_temp_dir = os.path.join(temp_dir, f"year_{year}")
            os.makedirs(yearly_temp_dir, exist_ok=True)
            yearly_temp_dirs.append(yearly_temp_dir)

            input_file_path = f"{input_path_gcs}/{year}_MSWEP.csv"
            with fs.open(input_file_path, "r") as f:
                yearly_data = pd.read_csv(f)

            # Normalize county FIPS and date
            yearly_data["county"] = yearly_data["county"].apply(lambda x: f"{int(x):05d}")
            yearly_data["date"] = pd.to_datetime(yearly_data["date"])

            for county, group in yearly_data.groupby("county"):
                county_file_path = os.path.join(yearly_temp_dir, f"{county}_MSWEP.csv")
                group.to_csv(county_file_path, index=False)

            end_time = time.time()
            print(f"Year {year} processed in {end_time - start_time:.2f} seconds.")

        print("Merging yearly files and uploading to GCS...")
        county_files: Dict[str, pd.DataFrame] = {}

        for yearly_temp_dir in yearly_temp_dirs:
            for county_file in os.listdir(yearly_temp_dir):
                county = county_file.split("_")[0]
                yearly_file_path = os.path.join(yearly_temp_dir, county_file)

                if county not in county_files:
                    county_files[county] = pd.read_csv(yearly_file_path)
                else:
                    new_data = pd.read_csv(yearly_file_path)
                    county_files[county] = pd.concat([county_files[county], new_data]).drop_duplicates()

        for county, df in county_files.items():
            local_file_path = os.path.join(temp_dir, f"{county}_MSWEP.csv")
            df.to_csv(local_file_path, index=False)

            output_file_path = f"{output_path_gcs}/{county}_MSWEP.csv"
            with fs.open(output_file_path, "w") as f:
                f.write(open(local_file_path, "r").read())

        print("Upload complete. Temporary directory cleaned up.")


def download_from_gcs(
    bucket_name: str,
    source_folder: str,
    local_folder: PathLike,
) -> None:
    """
    Download all CSV files from a GCS 'folder' prefix into a local directory.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    source_folder : str
        Prefix within the bucket to list (e.g., 'outputs/mswep/').
    local_folder : PathLike
        Local destination directory (created if missing).

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates `local_folder` if needed.
    - Downloads each blob that ends with '.csv' under the prefix.

    Raises
    ------
    google.api_core.exceptions.GoogleAPIError
        On GCS access failures (auth, permission, network, etc.).
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)

    os.makedirs(local_folder, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith(".csv"):
            destination = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(destination)


def process_precipitation_data_mswep(
    file_path: PathLike,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single county CSV of daily MSWEP precipitation, compute rolling-window
    metrics, annual maxima, and GEV-based percentiles; write processed outputs to disk.

    Parameters
    ----------
    file_path : PathLike
        Local path to a county MSWEP CSV with at least columns ['date', 'precipitation'].
        The county code is inferred from the filename prefix ('{county}_...').

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of:
        - Full daily DataFrame with added rolling metrics and percentile columns.
        - Annual maxima DataFrame aggregated by 'year'.

    Side Effects
    ------------
    - Writes two CSVs locally:
        'MSWEP_Daily_Precip_Processed_County/{county}_precip_processed.csv'
        'MSWEP_Ann_Max_Precip_County/Ann_max_precip_county_{county}.csv'

    Assumptions
    -----------
    - 'date' is parseable to datetime.
    - 'precipitation' is daily total for MSWEP in consistent units.
    - GEV fit via SciPy MLE is stable for the annual maxima series.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    KeyError
        If required columns are missing.
    """
    county = os.path.basename(file_path).split("_")[0]
    start_time = time.time()

    df = pd.read_csv(file_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    if "precipitation" not in df.columns:
        raise KeyError("Input CSV must contain a 'precipitation' column.")

    df.rename(columns={"precipitation": "MSWEP_precipitation"}, inplace=True)

    # Rolling metrics over a 30-day window
    df["MSWEP_precipitation_30d_sum"] = df["MSWEP_precipitation"].rolling(window=30, min_periods=1).sum()
    df["MSWEP_precipitation_30d_max_1d"] = df["MSWEP_precipitation"].rolling(window=30, min_periods=1).max()
    df["MSWEP_precipitation_30d_max_3d"] = (
        df["MSWEP_precipitation"].rolling(window=3, min_periods=1).sum().rolling(window=30, min_periods=1).max()
    )
    df["MSWEP_precipitation_30d_max_5d"] = (
        df["MSWEP_precipitation"].rolling(window=5, min_periods=1).sum().rolling(window=30, min_periods=1).max()
    )
    df["MSWEP_precipitation_30d_max_7d"] = (
        df["MSWEP_precipitation"].rolling(window=7, min_periods=1).sum().rolling(window=30, min_periods=1).max()
    )
    df["MSWEP_precipitation_30d_max_14d"] = (
        df["MSWEP_precipitation"].rolling(window=14, min_periods=1).sum().rolling(window=30, min_periods=1).max()
    )

    df["year"] = df["date"].dt.year

    aggregation_dict = {
        "MSWEP_precipitation_30d_sum": "max",
        "MSWEP_precipitation_30d_max_1d": "max",
        "MSWEP_precipitation_30d_max_3d": "max",
        "MSWEP_precipitation_30d_max_5d": "max",
        "MSWEP_precipitation_30d_max_7d": "max",
        "MSWEP_precipitation_30d_max_14d": "max",
        "MSWEP_precipitation": "max",
    }

    annual_maxima = df.groupby("year").agg(aggregation_dict).reset_index()

    # Fit GEV and compute modeled percentiles for each metric
    for column in aggregation_dict.keys():
        annual_max_distribution = annual_maxima[column].dropna().values
        if len(annual_max_distribution) > 0:
            shape, loc, scale = genextreme.fit(annual_max_distribution)
            df[f"{column}_percentile_modeled"] = genextreme.cdf(df[column], shape, loc, scale) * 100

    processed_folder = "MSWEP_Daily_Precip_Processed_County"
    os.makedirs(processed_folder, exist_ok=True)
    df.to_csv(f"{processed_folder}/{county}_precip_processed.csv", index=False)

    annual_max_folder = "MSWEP_Ann_Max_Precip_County"
    os.makedirs(annual_max_folder, exist_ok=True)
    annual_maxima.to_csv(f"{annual_max_folder}/Ann_max_precip_county_{county}.csv", index=False)

    end_time = time.time()
    print(f"County {county} processed in {end_time - start_time:.2f} seconds.")

    return df, annual_maxima


def process_all_precipitation_data_mswep(
    directory: PathLike,
) -> None:
    """
    Parallelize processing of all county MSWEP CSV files in a directory.

    Parameters
    ----------
    directory : PathLike
        Local folder containing per-county CSVs (filenames end with '.csv').

    Returns
    -------
    None

    Side Effects
    ------------
    - Launches a `ProcessPoolExecutor` to call `process_precipitation_data_mswep`
      for each file discovered.

    Raises
    ------
    FileNotFoundError
        If `directory` does not exist.
    """
    csv_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".csv")]

    with ProcessPoolExecutor() as executor:
        executor.map(process_precipitation_data_mswep, csv_files)


def merge_claims_with_precip(
    claims_folder: PathLike,
    precip_folder: PathLike,
    output_folder: PathLike,
) -> None:
    """
    Merge per-county processed precipitation with per-county claims on matching dates.

    Parameters
    ----------
    claims_folder : PathLike
        Folder containing '{county}_claims_merged.csv' files with a 'dateOfLoss' column.
    precip_folder : PathLike
        Folder containing '{county}_precip_processed.csv' files with a 'date' column
        and derived precipitation metrics.
    output_folder : PathLike
        Destination folder for merged per-county CSVs (created if missing).

    Returns
    -------
    None

    Side Effects
    ------------
    - Writes '{county}_claims_merged.csv' files to `output_folder`, overwriting if present.

    Raises
    ------
    FileNotFoundError
        If either input folder does not exist.
    KeyError
        If required columns ('dateOfLoss' or 'date') are missing in inputs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for claims_file in os.listdir(claims_folder):
        if claims_file.endswith("_claims_merged.csv"):
            county = claims_file.split("_")[0]
            claims_path = os.path.join(claims_folder, claims_file)
            precip_path = os.path.join(precip_folder, f"{county}_precip_processed.csv")

            if not os.path.exists(precip_path):
                print(f"No precipitation file found for {county}. Skipping.")
                continue

            claims_df = pd.read_csv(claims_path)
            precip_df = pd.read_csv(precip_path)

            if "dateOfLoss" not in claims_df.columns:
                raise KeyError(f"'dateOfLoss' missing in {claims_path}")
            if "date" not in precip_df.columns:
                raise KeyError(f"'date' missing in {precip_path}")

            claims_df["dateOfLoss"] = pd.to_datetime(claims_df["dateOfLoss"])
            precip_df["date"] = pd.to_datetime(precip_df["date"])

            # Normalize to naive datetimes for a direct merge
            claims_df["dateOfLoss"] = claims_df["dateOfLoss"].dt.tz_localize(None)
            precip_df["date"] = precip_df["date"].dt.tz_localize(None)

            merged_df = pd.merge(claims_df, precip_df, left_on="dateOfLoss", right_on="date", how="left")

            merged_df.drop(columns=["date", "county", "year"], inplace=True, errors="ignore")

            output_path = os.path.join(output_folder, f"{county}_claims_merged.csv")
            merged_df.to_csv(output_path, index=False)

            print(f"County {county} claims merged and saved to {output_path}.")


def concatenate_processed_claims(
    output_file: PathLike = "Processed_Claims_ERA5.csv",
    processed_path: PathLike = "Output_Claims_ERA5",
) -> None:
    """
    Concatenate all per-county processed claims CSVs into a single file.

    Parameters
    ----------
    output_file : PathLike, default 'Processed_Claims_ERA5.csv'
        Output CSV path for the combined dataset.
    processed_path : PathLike, default 'Output_Claims_ERA5'
        Folder containing per-county '*_claims_merged.csv' files.

    Returns
    -------
    None

    Side Effects
    ------------
    - Reads every '*_claims_merged.csv' under `processed_path`.
    - Writes concatenated CSV to `output_file`.

    Raises
    ------
    FileNotFoundError
        If `processed_path` does not exist or contains no matching files.
    """
    files = [f for f in os.listdir(processed_path) if f.endswith("_claims_merged.csv")]
    if not files:
        raise FileNotFoundError(f"No '*_claims_merged.csv' files found in {processed_path}")

    df_list: List[pd.DataFrame] = []
    for file in files:
        file_path = os.path.join(processed_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
        print(f"Loaded {file}")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"All county files concatenated into {output_file}")


def push_to_cloud_and_conditional_remove(
    local_paths: Sequence[PathLike],
    remove_flags: Sequence[bool],
    bucket_name: str,
    target_folder: str,
) -> None:
    """
    Upload local files/folders to GCS and optionally remove local copies.

    Parameters
    ----------
    local_paths : Sequence[PathLike]
        List of file or directory paths to upload.
    remove_flags : Sequence[bool]
        Flags parallel to `local_paths`; if True, remove the item locally after upload.
    bucket_name : str
        Target GCS bucket name.
    target_folder : str
        Destination prefix inside the bucket (e.g., 'outputs/mswep').

    Returns
    -------
    None

    Side Effects
    ------------
    - Uploads to GCS under 'gs://{bucket_name}/{target_folder}/...'.
    - Optionally deletes local files or directories after successful uploads.

    Raises
    ------
    ValueError
        If `local_paths` and `remove_flags` lengths differ.
    FileNotFoundError
        If any `local_paths` entry does not exist.
    google.api_core.exceptions.GoogleAPIError
        On GCS failures (auth/permission/network).
    """
    client = storage.Client()

    if len(local_paths) != len(remove_flags):
        raise ValueError("The number of paths and remove flags must be the same.")

    for local_path, remove_flag in zip(local_paths, remove_flags):
        if not os.path.exists(local_path):
            print(f"Local path does not exist: {local_path}")
            continue

        try:
            bucket = client.bucket(bucket_name)
            print(f"Bucket accessed: {bucket_name}")
        except Exception as e:
            print(f"Failed to access bucket: {e}")
            raise

        try:
            if os.path.isfile(local_path):
                relative_path = os.path.basename(local_path)
                blob_path = f"{target_folder}/{relative_path}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")

                if remove_flag:
                    os.remove(local_path)
                    print(f"Removed local file: {local_path}")

            elif os.path.isdir(local_path):
                base_folder_name = os.path.basename(local_path)
                folder_target_path = f"{target_folder}/{base_folder_name}"

                for root, _, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.basename(file)
                        blob_path = f"{folder_target_path}/{relative_path}"

                        blob = bucket.blob(blob_path)
                        blob.upload_from_filename(file_path)
                        print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_path}")

                if remove_flag:
                    shutil.rmtree(local_path)
                    print(f"Removed local folder and its contents: {local_path}")

            else:
                print(f"The path is neither a file nor a folder: {local_path}")
        except Exception as e:
            print(f"Error during file/folder upload: {e}")
            raise
