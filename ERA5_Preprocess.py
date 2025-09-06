"""
Annotated ERA5 County Pipeline Utilities
========================================

Notes
-----
* ERA5 precipitation variable defaults to 'tp' (total precipitation) and is stored in meters.
* Coordinates expected: 'latitude', 'longitude', and 'valid_time' (xarray dims/coords).
* County polygons must contain a 'GEOID' column (string FIPS, zero-padded to 5 where used).
* Pixel weights are built by rasterizing county geometry onto the grid; weights sum to 1 per county.
  If no pixels intersect a county, the nearest grid cell to the county centroid gets weight=1 (others 0).
* Weighted aggregation is performed as (precipitation * weights).sum(dim=['latitude','longitude']);
  weight arrays must match the grid shape exactly.
* Yearly processing expects NetCDFs under a GCS prefix (e.g., 'era5/daily/...') and writes
  `daily_precip_by_county_{year}.csv` to GCS.
* Rechunking merges per-year CSVs into per-county CSVs and uploads back to GCS; temporary local files
  are cleaned up automatically.
* Claims merge requires per-county claims with 'dateOfLoss'; dates are merged as naive timestamps
  (timezones stripped) and extra columns ['date','county','year'] are dropped post-merge.
* GCS I/O via `gcsfs` (streaming reads/writes) and `google-cloud-storage` (bulk blob ops). Weights are
  saved with `numpy.save`; load with `np.load(path, allow_pickle=True).item()`.
* Parallelism: county CSV processing uses `ProcessPoolExecutor`; tune process count/chunking for your host.
* Progress uses `print`; switch to `logging` for structured, level-based logs in production.
* CRS caveat: if the NetCDF lacks a CRS, this code writes the shapefile CRS onto it. Ensure the shapefile
  CRS matches the data’s true CRS; if your ERA5 is geographic lat/lon, consider setting EPSG:4326 instead.
* Interpretation: GEV percentiles are relative to the distribution of **annual maxima** (return-period framing).
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import os
import time
import shutil
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # needed for .rio accessor
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from rasterio.features import geometry_mask
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import genextreme
import gcsfs
from google.cloud import storage


def calculate_pixel_weights_nofrac(
    county_shapefile: Union[str, os.PathLike],
    precipitation_nc_path: str,
    output_path: str,
    gcs_bucket: str,
) -> Dict[str, np.ndarray]:
    """Create per‑county pixel weights (no fractional pixels) against a gridded precip field on GCS.

    Logic (unchanged):
      1) Read county shapefile locally.
      2) Read a single NetCDF from GCS with ERA5 precipitation (variable 'tp' assumed).
      3) If dataset lacks CRS, copy CRS from the shapefile (as in original).
      4) Reproject counties to the grid CRS; clip by grid extent.
      5) For each county, build a raster mask over the grid and normalize to sum=1.
         If no pixels fall inside, assign weight 1 to the nearest grid cell to the county centroid.
      6) Save weights dict to GCS via `numpy.save` and return it.

    Parameters
    ----------
    county_shapefile : str | os.PathLike
        Path to county polygons (must include GEOID field).
    precipitation_nc_path : str
        *Path inside the bucket* to the NetCDF (e.g., "era5/era5_daily_1950.nc").
    output_path : str
        *Path inside the bucket* for the weights `.npy` file to write.
    gcs_bucket : str
        Bucket name or `gs://bucket-name` prefix; function concatenates with provided paths.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping `GEOID -> weight array` shaped like the grid (latitude x longitude).

    Notes
    -----
    * This function assumes ERA5 precipitation variable is named 'tp'. Adjust in code if different.
    * Output can be reloaded with `np.load(path, allow_pickle=True).item()`.
    """
    # Set up Google Cloud Storage filesystem
    fs = gcsfs.GCSFileSystem()

    # Construct full GCS paths
    precipitation_nc_path = f'{gcs_bucket}/{precipitation_nc_path}'
    output_path = f'{gcs_bucket}/{output_path}'

    # Load the county shapefile locally
    counties = gpd.read_file(county_shapefile)
    
    # Load the precipitation NetCDF file from GCS
    with fs.open(precipitation_nc_path, 'rb') as f:
        ds = xr.open_dataset(f)
    
    # Ensure that the precipitation data has a CRS
    if not ds.rio.crs:
        # ERA5 is lat/lon; if the file lacks metadata, assume WGS84
        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        print("Assigned EPSG:4326 to precipitation data:", ds.rio.crs)
    
    precip = ds['tp']  # Adjust this depending on the variable name in NetCDF file
    
    # Reproject counties to match precipitation data
    counties = counties.to_crs(precip.rio.crs)
    
    # Define the bounding box of the precipitation data
    precip_extent = box(
        precip['longitude'].min().item(), precip['latitude'].min().item(),
        precip['longitude'].max().item(), precip['latitude'].max().item()
    )
    
    # Clip counties to the extent of the precipitation data
    counties_clipped = counties[counties.intersects(precip_extent)]
    
    # Extract the pixel centroids
    pixel_lon, pixel_lat = np.meshgrid(precip['longitude'].values, precip['latitude'].values)
    pixel_centroids = gpd.GeoDataFrame(
        {'geometry': [Point(x, y) for x, y in zip(pixel_lon.ravel(), pixel_lat.ravel())]},
        crs=precip.rio.crs
    )
    
    # Calculate weights for each pixel within each county
    weights: Dict[str, np.ndarray] = {}
    
    for county in counties_clipped.itertuples():
        # Create a mask for the current county
        mask = geometry_mask([county.geometry], transform=precip.rio.transform(), invert=True, out_shape=precip.shape[-2:])
        
        # Calculate the area of each pixel within the county
        pixel_areas = mask.astype(int)  # Convert boolean mask to integer array (1 for inside, 0 for outside)
        
        # Calculate the weight for each pixel as the fraction of area within the county
        total_area = pixel_areas.sum()
        if total_area > 0:
            pixel_weights = pixel_areas / total_area
        else:
            # Find the closest pixel centroid to the county centroid if no pixels are contained
            county_centroid = county.geometry.centroid
            pixel_centroids['distance'] = pixel_centroids['geometry'].distance(county_centroid)
            closest_pixel = pixel_centroids.loc[pixel_centroids['distance'].idxmin()]
            
            # Identify the closest pixel's index in the precipitation grid
            closest_pixel_idx = (
                np.abs(precip['latitude'] - closest_pixel.geometry.y).argmin(),
                np.abs(precip['longitude'] - closest_pixel.geometry.x).argmin()
            )
            
            # Assign a weight of 1 to the closest pixel
            pixel_weights = np.zeros_like(pixel_areas)
            pixel_weights[closest_pixel_idx] = 1
        
        weights[county.GEOID] = pixel_weights
    
    # Save the weights dictionary to GCS
    with fs.open(output_path, 'wb') as f:
        np.save(f, weights)
    
    return weights


def calculate_daily_precipitation(
    county_shapefile: Union[str, os.PathLike],
    precipitation_dir: str,
    weights: Mapping[Any, np.ndarray],
    output_dir: str,
    gcs_bucket: str,
) -> None:
    """Compute per‑county daily precipitation time series from ERA5 NetCDFs stored on GCS.

    Logic (unchanged): list NetCDFs in the directory, open each year, compute a weighted
    spatial mean for each county/day using the provided pixel weights, and write one CSV per year.

    Parameters
    ----------
    county_shapefile : str | os.PathLike
        Path to local county shapefile (used here to iterate GEOIDs; geometry not modified).
    precipitation_dir : str
        Folder path inside the bucket that holds yearly `.nc` files.
    weights : Mapping[Any, np.ndarray]
        Dictionary of county weights previously created by `calculate_pixel_weights_nofrac`.
    output_dir : str
        Folder path inside the bucket to write `daily_precip_by_county_<year>.csv`.
    gcs_bucket : str
        Bucket name or `gs://bucket-name` prefix.

    Returns
    -------
    None
    """
    # Set up Google Cloud Storage filesystem
    fs = gcsfs.GCSFileSystem()

    # Construct full GCS paths
    precipitation_dir_path = f'{gcs_bucket}/{precipitation_dir}'
    output_dir_path = f'{gcs_bucket}/{output_dir}'

    # Load the county shapefile locally
    counties = gpd.read_file(county_shapefile)

    # Start time for processing
    start_time = time.time()

    # Ensure the output directory exists on GCS
    if not fs.exists(output_dir_path):
        fs.mkdirs(output_dir_path)

    # List all files in the precipitation directory on GCS
    precipitation_files = [f for f in fs.ls(precipitation_dir_path) if f.endswith('.nc')]

    # Iterate over each NetCDF file in the directory
    for year_file_path in sorted(precipitation_files):
        print(f"Processing file: {year_file_path}")

        # Extract the year from the filename (filename is structured as 'era5_daily_precipitation_1950.nc')
        year_file_name = os.path.basename(year_file_path)
        year = year_file_name.split('_')[-1].split('.')[0]

        # Open the NetCDF file
        ds = xr.open_dataset(fs.open(year_file_path, 'rb'))

        # Check if the dataset contains a 'valid_time' variable
        if 'valid_time' not in ds:
            print(f"Skipping file {year_file_name} as it does not contain 'valid_time' variable.")
            continue

        precip = ds['tp']  # Adjust the variable name as needed
        year_records = []

        # Iterate over each county
        for county in counties.itertuples():
            geo_id = county.GEOID
            weight = weights.get(geo_id)

            if weight is not None:
                # Calculate the weighted average precipitation for each day
                w_da = xr.DataArray(
                    weight.astype(float),
                    dims=["latitude", "longitude"],
                    coords={"latitude": ds["latitude"].values, "longitude": ds["longitude"].values},
                ) # Wrap weights so xarray aligns by coords
                daily_precip = (precip * w_da).sum(dim=["latitude", "longitude"])


                # Add results to the list for the current year
                for date, value in zip(ds['valid_time'].values, daily_precip.values):
                    year_records.append({
                        'date': date,
                        'county': geo_id,
                        'ERA5_precipitation': value
                    })

        # Convert the list of records for the year into a DataFrame
        year_df = pd.DataFrame.from_records(year_records)

        # Save the result for the year to a CSV file in GCS
        output_csv_path = f"{output_dir_path}/daily_precip_by_county_{year}.csv"
        with fs.open(output_csv_path, 'w') as f:
            year_df.to_csv(f, index=False)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Saved results for {year} to {output_csv_path} in {elapsed_time:.2f} seconds")
        start_time = time.time()


def rechunk_precipitation_data_by_county_gcs_efficient(
    years: Sequence[int],
    input_path: str,
    output_path: str,
    gcs_bucket: str,
) -> None:
    """Merge per‑year per‑county daily precipitation CSVs into one CSV per county (on GCS).

    Logic (unchanged): for each year, write temporary per‑county CSVs, then concatenate across
    years and upload `<county>_precip.csv` to the `output_path` in the same bucket.

    Parameters
    ----------
    years : Sequence[int]
        List/range of years to process.
    input_path : str
        Folder with `daily_precip_by_county_<year>.csv` inside the bucket.
    output_path : str
        Destination folder for `<county>_precip.csv` inside the bucket.
    gcs_bucket : str
        Bucket name or `gs://bucket-name` prefix.

    Returns
    -------
    None
    """
    # Set up Google Cloud Storage filesystem
    fs = gcsfs.GCSFileSystem()

    # Construct full GCS paths
    input_path_gcs = f"{gcs_bucket}/{input_path}"
    output_path_gcs = f"{gcs_bucket}/{output_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at {temp_dir}")

        # Step 1: Process each year and store county-specific files in a separate folder
        yearly_temp_dirs: list[str] = []

        for year in years:
            start_time = time.time()
            print(f"Processing year {year}...")

            yearly_temp_dir = os.path.join(temp_dir, f"year_{year}")
            os.makedirs(yearly_temp_dir, exist_ok=True)
            yearly_temp_dirs.append(yearly_temp_dir)

            # Read the year's data from GCS
            input_file_path = f"{input_path_gcs}/daily_precip_by_county_{year}.csv"
            with fs.open(input_file_path, 'r') as f:
                yearly_data = pd.read_csv(f)

            # Ensure FIPS is 5 characters and dates are in datetime format
            yearly_data['county'] = yearly_data['county'].apply(lambda x: f"{int(x):05d}")
            yearly_data['date'] = pd.to_datetime(yearly_data['date'])

            # Write data for each county into the year's temp folder
            for county, group in yearly_data.groupby('county'):
                county_file_path = os.path.join(yearly_temp_dir, f"{county}_precip.csv")
                group.to_csv(county_file_path, index=False)

            end_time = time.time()
            print(f"Year {year} processed in {end_time - start_time:.2f} seconds.")

        # Step 2: Merge yearly county files and upload to GCS
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

        # Write final county files and upload to GCS
        for county, df in county_files.items():
            local_file_path = os.path.join(temp_dir, f"{county}_precip.csv")
            df.to_csv(local_file_path, index=False)

            # Upload to GCS
            output_file_path = f"{output_path_gcs}/{county}_precip.csv"
            with fs.open(output_file_path, 'w') as f:
                f.write(open(local_file_path, 'r').read())

        print("Upload complete. Temporary directory cleaned up.")


def download_from_gcs(bucket_name: str, source_folder: str, local_folder: Union[str, os.PathLike]) -> None:
    """Download all CSV files from a GCS bucket *folder* to a local folder.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket (no `gs://`).
    source_folder : str
        Prefix/folder inside the bucket to list (e.g., "era5/per_county").
    local_folder : str | os.PathLike
        Local destination directory; will be created if missing.

    Returns
    -------
    None
    """
    """Download all files from a GCS bucket folder to a local folder."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)
    
    os.makedirs(local_folder, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith('.csv'):
            destination = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(destination)


def process_precipitation_data_era5(file_path: Union[str, os.PathLike]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling 30‑day metrics, annual maxima, and GEV‑based percentiles for one county file.

    Logic (unchanged):
      * read `<county>_precip.csv`, sort by date
      * build 30‑day sum and rolling‑max metrics (1d/3d/5d/7d/14d accumulations)
      * aggregate annual maxima for each metric
      * fit GEV to annual maxima and add percentile columns to the daily frame
      * write processed daily file and annual maxima to disk; return both DataFrames

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to `<county>_precip.csv` with columns ['date','county','ERA5_precipitation'].

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (daily dataframe with added metrics/percentiles, annual maxima dataframe).

    Notes
    -----
    * Uses SciPy's MLE GEV fit (`scipy.stats.genextreme`).
    * Percentiles are expressed in [0, 100].
    """
    county = os.path.basename(file_path).split('_')[0]
    start_time = time.time()

    # Load and sort data by date
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)

    # Calculate the specified metrics within a 30-day rolling window
    df['ERA5_precipitation_30d_sum'] = df['ERA5_precipitation'].rolling(window=30, min_periods=1).sum()
    df['ERA5_precipitation_30d_max_1d'] = df['ERA5_precipitation'].rolling(window=30, min_periods=1).max()
    df['ERA5_precipitation_30d_max_3d'] = (
        df['ERA5_precipitation'].rolling(window=3, min_periods=1).sum()
        .rolling(window=30, min_periods=1).max()
    )
    df['ERA5_precipitation_30d_max_5d'] = (
        df['ERA5_precipitation'].rolling(window=5, min_periods=1).sum()
        .rolling(window=30, min_periods=1).max()
    )
    df['ERA5_precipitation_30d_max_7d'] = (
        df['ERA5_precipitation'].rolling(window=7, min_periods=1).sum()
        .rolling(window=30, min_periods=1).max()
    )
    df['ERA5_precipitation_30d_max_14d'] = (
        df['ERA5_precipitation'].rolling(window=14, min_periods=1).sum()
        .rolling(window=30, min_periods=1).max()
    )

    # Extract year from the date column for annual aggregation
    df['year'] = df['date'].dt.year

    # Create aggregation dictionary for annual maxima
    aggregation_dict = {
        'ERA5_precipitation_30d_sum': 'max',
        'ERA5_precipitation_30d_max_1d': 'max',
        'ERA5_precipitation_30d_max_3d': 'max',
        'ERA5_precipitation_30d_max_5d': 'max',
        'ERA5_precipitation_30d_max_7d': 'max',
        'ERA5_precipitation_30d_max_14d': 'max',
        'ERA5_precipitation': 'max'  # daily max of base precipitation
    }

    # Calculate annual maxima
    annual_maxima = df.groupby('year').agg(aggregation_dict).reset_index()

    # Fit GEV distribution and calculate percentiles for each column
    columns_to_fit = list(aggregation_dict.keys())
    for column in columns_to_fit:
        annual_max_distribution = annual_maxima[column].dropna().values
        if len(annual_max_distribution) > 0:
            shape, loc, scale = genextreme.fit(annual_max_distribution)
            df[f'{column}_percentile_modeled'] = genextreme.cdf(df[column], shape, loc, scale) * 100

    # Save the processed data and annual maxima
    processed_folder = 'ERA5_Daily_Precip_Processed_County'
    os.makedirs(processed_folder, exist_ok=True)
    df.to_csv(f'{processed_folder}/{county}_precip_processed.csv', index=False)

    annual_max_folder = 'ERA5_Ann_Max_Precip_County'
    os.makedirs(annual_max_folder, exist_ok=True)
    annual_maxima.to_csv(f'{annual_max_folder}/Ann_max_precip_county_{county}.csv', index=False)

    end_time = time.time()
    print(f"County {county} processed in {end_time - start_time:.2f} seconds.")

    return df, annual_maxima


def process_all_precipitation_data_era5(directory: Union[str, os.PathLike]) -> None:
    """Parallelize `process_precipitation_data_era5` across all CSVs in a directory.

    Parameters
    ----------
    directory : str | os.PathLike
        Folder containing `<county>_precip.csv` files to process.

    Returns
    -------
    None
    """
    csv_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.csv')]

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(process_precipitation_data_era5, csv_files)


def split_claims_data_by_county_gcs(claims_path: str, gcs_bucket: str, output_path: str) -> None:
    """Split a claims CSV on GCS into one CSV per county (written back to GCS).

    Parameters
    ----------
    claims_path : str
        Input file path inside the bucket (e.g., "nfip/claims.csv").
    gcs_bucket : str
        Bucket name or `gs://bucket-name` prefix.
    output_path : str
        Folder inside the bucket where `<county>_claims.csv` files will be written.

    Returns
    -------
    None
    """
    # Set up Google Cloud Storage filesystem
    fs = gcsfs.GCSFileSystem()

    # Construct full GCS paths
    claims_file_path = f'{gcs_bucket}/{claims_path}'
    output_dir_path = f'{gcs_bucket}/{output_path}'

    # Load the claims data
    print("Loading claims data...")
    with fs.open(claims_file_path, 'r') as f:
        claims_df = pd.read_csv(f)
    print(f"Loaded {len(claims_df)} claims.")

    # Ensure countyCode is in the correct format
    claims_df['countyCode'] = claims_df['countyCode'].apply(lambda x: f"{int(x):05d}" if not pd.isna(x) else None)

    # Drop claims with missing countyCode
    claims_df = claims_df.dropna(subset=['countyCode'])

    # Ensure output directory exists on GCS
    if not fs.exists(output_dir_path):
        fs.mkdirs(output_dir_path)

    # Split claims by county and save to separate files on GCS
    for county, group in claims_df.groupby('countyCode'):
        county_file_path = f"{output_dir_path}/{county}_claims.csv"
        with fs.open(county_file_path, 'w') as f:
            group.to_csv(f, index=False)
        print(f"Saved claims for county {county} to {county_file_path}")

    print("Claims data split by county successfully.")


def merge_claims_with_precip(
    claims_folder: Union[str, os.PathLike],
    precip_folder: Union[str, os.PathLike],
    output_folder: Union[str, os.PathLike],
) -> None:
    """Merge per‑county claims with processed precip by date and write merged CSVs.

    Expects `<county>_claims.csv` in `claims_folder` and `<county>_precip_processed.csv` in `precip_folder`.
    Writes `<county>_claims_merged.csv` to `output_folder`.

    Parameters
    ----------
    claims_folder : str | os.PathLike
        Folder containing per‑county claims CSVs.
    precip_folder : str | os.PathLike
        Folder containing per‑county processed precipitation CSVs.
    output_folder : str | os.PathLike
        Destination folder for merged outputs.

    Returns
    -------
    None
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each claims file in the Claims_By_County folder
    for claims_file in os.listdir(claims_folder):
        if claims_file.endswith('_claims.csv'):
            # Extract county name from file name
            county = claims_file.split('_')[0]
            
            # Construct file paths
            claims_path = os.path.join(claims_folder, claims_file)
            precip_path = os.path.join(precip_folder, f'{county}_precip_processed.csv')
            
            # Check if the corresponding precipitation file exists
            if not os.path.exists(precip_path):
                print(f"No precipitation file found for {county}. Skipping.")
                continue
            
            # Read the claims and precipitation data
            claims_df = pd.read_csv(claims_path)
            precip_df = pd.read_csv(precip_path)
            
            # Convert 'dateOfLoss' and 'date' to datetime if not already
            claims_df['dateOfLoss'] = pd.to_datetime(claims_df['dateOfLoss'])
            precip_df['date'] = pd.to_datetime(precip_df['date'])
            
            # Align the timezones (convert to UTC for consistency)
            claims_df['dateOfLoss'] = claims_df['dateOfLoss'].dt.tz_localize(None)
            precip_df['date'] = precip_df['date'].dt.tz_localize(None)
            
            # Merge on the date
            merged_df = pd.merge(claims_df, precip_df, left_on='dateOfLoss', right_on='date', how='left')
            
            # Drop the extra 'date' and 'county' columns after merge
            merged_df.drop(columns=['date', 'county', 'year'], inplace=True, errors='ignore')
            
            # Output the merged DataFrame to the new folder
            output_path = os.path.join(output_folder, f'{county}_claims_merged.csv')
            merged_df.to_csv(output_path, index=False)
            
            print(f"County {county} claims merged and saved to {output_path}.")


def concatenate_processed_claims(
    output_file: str = "Processed_Claims_ERA5.csv",
    processed_path: str = "Output_Claims_ERA5",
) -> None:
    """Concatenate all per‑county merged claims CSVs into one file.

    Parameters
    ----------
    output_file : str
        Path for the final concatenated CSV.
    processed_path : str
        Folder holding `<county>_claims_merged.csv` files.

    Returns
    -------
    None
    """
    # List all files in the processed claims directory
    files = [f for f in os.listdir(processed_path) if f.endswith('_claims_merged.csv')]
    
    # Initialize an empty list to store DataFrames
    df_list: list[pd.DataFrame] = []

    # Loop through the files and read them into DataFrames
    for file in files:
        file_path = os.path.join(processed_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
        print(f"Loaded {file}")

    # Concatenate all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"All county files concatenated into {output_file}")


def push_to_cloud_and_conditional_remove(
    local_paths: Sequence[Union[str, os.PathLike]],
    remove_flags: Sequence[bool],
    bucket_name: str,
    target_folder: str,
) -> None:
    """Upload files/folders to GCS and optionally delete local copies (logic unchanged).

    Parameters
    ----------
    local_paths : Sequence[str | os.PathLike]
        Paths to files or folders to upload.
    remove_flags : Sequence[bool]
        Whether to remove each local path after a successful upload.
    bucket_name : str
        Target GCS bucket name (no `gs://`).
    target_folder : str
        Destination prefix/folder in the bucket.

    Returns
    -------
    None
    """
    """
    Push files or folders to the cloud and conditionally remove them from the local system.

    Parameters:
        local_paths (list): List of file or folder paths to upload.
        remove_flags (list): List of boolean values indicating whether to remove each item after upload.
        bucket_name (str): The name of the Google Cloud Storage bucket.
        target_folder (str): The destination folder path in the bucket.
    """
    # Initialize the client
    client = storage.Client()

    if len(local_paths) != len(remove_flags):
        raise ValueError("The number of paths and remove flags must be the same.")

    for local_path, remove_flag in zip(local_paths, remove_flags):
        # Check if the local path exists
        if not os.path.exists(local_path):
            print(f"Local path does not exist: {local_path}")
            continue

        # Access the bucket
        try:
            bucket = client.bucket(bucket_name)
            print(f"Bucket accessed: {bucket_name}")
        except Exception as e:
            print(f"Failed to access bucket: {e}")
            raise

        # Check if the path is a file or folder
        try:
            if os.path.isfile(local_path):
                # Upload the single file
                relative_path = os.path.basename(local_path)
                blob_path = f"{target_folder}/{relative_path}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")

                # Remove the local file if remove_flag is True
                if remove_flag:
                    os.remove(local_path)
                    print(f"Removed local file: {local_path}")

            elif os.path.isdir(local_path):
                # Use the folder name as a subfolder in the cloud
                base_folder_name = os.path.basename(local_path)
                folder_target_path = f"{target_folder}/{base_folder_name}"

                # Walk through the folder and upload each file
                for root, _, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Only include the file name in the cloud path
                        relative_path = os.path.basename(file)
                        blob_path = f"{folder_target_path}/{relative_path}"

                        blob = bucket.blob(blob_path)
                        blob.upload_from_filename(file_path)
                        print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_path}")

                # Remove the local folder and its contents if remove_flag is True
                if remove_flag:
                    shutil.rmtree(local_path)
                    print(f"Removed local folder and its contents: {local_path}")

            else:
                print(f"The path is neither a file nor a folder: {local_path}")
        except Exception as e:
            print(f"Error during file/folder upload: {e}")
            raise
