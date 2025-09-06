import pandas as pd
import numpy as np
from st_dbscan import ST_DBSCAN
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
from itertools import product
from sklearn.preprocessing import MinMaxScaler


def disaster_sensitivity(disasters, space_thres_list, time_thres_list, num_thres_list):
    """
    Runs ST_DBSCAN on the given DataFrame with all combinations of parameters for disaster data.

    Parameters:
    - disasters: DataFrame containing columns 'centroid_lat', 'centroid_lon', 'daysSinceStart'.
    - space_thres_list: List of spatial thresholds.
    - time_thres_list: List of temporal thresholds.
    - num_thres_list: List of minimum number of points to form a cluster.

    Returns:
    - disasters: The updated DataFrame with new cluster fields for each parameter combination.
    """
    # Ensure the data chunk is properly prepared
    st_chunk = disasters[['centroid_lat', 'centroid_lon', 'daysSinceStart']].values
    st_chunk = st_chunk[~np.isnan(st_chunk).any(axis=1)]

    if st_chunk.size == 0:
        raise ValueError("No valid data points for clustering.")

    # Prepare a list to store results for CSV export
    results = []

    # Loop through all combinations of parameters
    for space_thres, time_thres, num_thres in product(space_thres_list, time_thres_list, num_thres_list):
        # Instantiate ST_DBSCAN
        st_dbscan = ST_DBSCAN(eps1=space_thres, eps2=time_thres, min_samples=num_thres)

        # Fit the data
        st_dbscan.fit(st_chunk)

        # Create a new column name for the cluster labels
        cluster_field = f'st_cluster_{space_thres}_{time_thres}_{num_thres}'

        # Add cluster labels to the original DataFrame
        disasters[cluster_field] = st_dbscan.labels  # Use labels_ if applicable

        # Calculate metrics
        total_clusters = disasters[cluster_field].max()
        percent_clustered = 100 * (1 - len(disasters[disasters[cluster_field] == -1]) / len(disasters))

        # Append the results for this parameter combination
        results.append({
            'space_thres': space_thres,
            'time_thres': time_thres,
            'num_thres': num_thres,
            'total_clusters': total_clusters,
            'percent_clustered': percent_clustered
        })

        # Optional: Print summary for each parameter combination
        print(f"Processed: {cluster_field}, Total clusters extracted: {total_clusters}, Percent clustered: {percent_clustered}%")

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    os.makedirs('Clusters', exist_ok=True)  # Ensure the directory exists
    results_df.to_csv('Clusters/cluster_sensitivities_di.csv', index=False)

    return disasters


def sensitivity_analysis(claims, space_thres_list, time_thres_list, num_thres_list):
    """
    Perform sensitivity analysis of spatio-temporal clustering thresholds on insurance claims data.
    
    This function evaluates how varying temporal, spatial, and minimum sample thresholds
    affect the formation of clusters of flood-related claims. It applies a two-stage
    clustering process: first temporal clustering by county, then spatio-temporal clustering
    using an ST-DBSCAN algorithm, and records summary statistics for each parameter combination.
    
    Steps:
        1. Temporal Clustering:
            - For each county, cluster claims based on days since start using DBSCAN.
            - Assign temporal cluster labels within each county.
        2. Spatio-Temporal Clustering:
            - Aggregate temporal clusters by their median latitude, longitude, and time.
            - Apply ST-DBSCAN on the aggregated data with spatial and temporal thresholds.
        3. Mapping Back to Claims:
            - Map spatio-temporal clusters back to the original claims dataset.
            - Handle unclustered temporal clusters by assigning unique new cluster IDs.
        4. Summary Statistics:
            - Compute total number of clusters and the percentage of unclustered claims.
            - Store these metrics for each parameter combination.
    
    Args:
        claims (pd.DataFrame): DataFrame of claims with columns:
            - 'countyCode': County identifier.
            - 'latitude', 'longitude': Claim location.
            - 'daysSinceStart': Time of claim relative to start date.
        space_thres_list (list[float]): List of spatial distance thresholds for ST-DBSCAN.
        time_thres_list (list[float]): List of temporal distance thresholds for DBSCAN/ST-DBSCAN.
        num_thres_list (list[int]): List of minimum sample thresholds for DBSCAN/ST-DBSCAN.
    
    Returns:
        tuple:
            claims (pd.DataFrame): Updated claims DataFrame with new spatio-temporal cluster labels
                added for each parameter combination.
            pd.DataFrame: Summary DataFrame with columns:
                - 'space_thres': Spatial threshold used.
                - 'time_thres': Temporal threshold used.
                - 'num_thres': Minimum samples threshold used.
                - 'total_clusters': Number of resulting clusters.
                - 'unclustered_percentage': Percentage of claims not assigned to a cluster.
    
    Example:
        claims, summary_df = sensitivity_analysis(
            claims_df,
            space_thres_list=[0.1, 0.2],
            time_thres_list=[30, 60],
            num_thres_list=[5, 10]
        )
    
    Notes:
        - The function can be computationally intensive for large datasets since it
          loops through multiple parameter combinations and counties.
        - ST_DBSCAN is assumed to be a custom implementation requiring 'eps1', 'eps2', and 'min_samples'.
    """
    results = []

    for time_thres in time_thres_list:
        for num_thres in num_thres_list:
            # Step 1: Temporal Clustering
            claims['temporal_cluster'] = -1
            start_time_temp = time.time()
            for county in claims['countyCode'].unique():
                county_data = claims[claims['countyCode'] == county]
                db = DBSCAN(eps=time_thres, min_samples=num_thres, metric='euclidean')
                temporal_labels = db.fit_predict(county_data[['daysSinceStart']])
                claims.loc[county_data.index, 'temporal_cluster'] = temporal_labels
            end_time_temp = time.time()
            print(f"Temporal clustering completed in {end_time_temp - start_time_temp:.2f} seconds.")

            # Pre-compute temporal clusters for efficiency
            temporal_clusters = claims[claims['temporal_cluster'] != -1]
            grouped_clusters = temporal_clusters.groupby(
                ['countyCode', 'temporal_cluster']
            ).agg({
                'latitude': 'median',
                'longitude': 'median',
                'daysSinceStart': 'median'
            }).reset_index()

            for space_thres in space_thres_list:
                # Step 2: Spatio-Temporal Clustering
                start_time_st = time.time()
                data = grouped_clusters[['daysSinceStart', 'latitude', 'longitude']].to_numpy()
                st_dbscan = ST_DBSCAN(eps1=space_thres, eps2=time_thres, min_samples=num_thres)
                st_dbscan.fit(data)
                grouped_clusters[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'] = st_dbscan.labels
                end_time_st = time.time()
                print(f"Spatio-temporal clustering completed in {end_time_st - start_time_st:.2f} seconds.")

                # Step 3: Map Spatio-Temporal Clusters to Original Claims
                start_time_merge = time.time()
                
                # Get the maximum spatiotemporal cluster label
                max_st_cluster = grouped_clusters[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'].max()
                
                # Identify temporal clusters that are unclustered in spatiotemporal clustering
                unclustered_temporal_clusters = grouped_clusters[
                    grouped_clusters[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'] == -1
                ]
                
                # Assign new unique cluster IDs to these unclustered temporal clusters
                unclustered_temporal_clusters['new_cluster'] = range(max_st_cluster + 1, max_st_cluster + 1 + len(unclustered_temporal_clusters))
                
                # Create a mapping of (countyCode, temporal_cluster) -> new cluster ID
                cluster_mapping = grouped_clusters.set_index(['countyCode', 'temporal_cluster'])[
                    f'st_cluster_{space_thres}_{time_thres}_{num_thres}'
                ].to_dict()
                
                # Add the new clusters for unclustered temporal clusters
                cluster_mapping.update(unclustered_temporal_clusters.set_index(['countyCode', 'temporal_cluster'])['new_cluster'].to_dict())
                
                # Map clusters back to the claims data
                claims[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'] = claims.apply(
                    lambda row: cluster_mapping.get((row['countyCode'], row['temporal_cluster']), -1), axis=1
                )
                
                end_time_merge = time.time()
                print(f"Merging clusters completed in {end_time_merge - start_time_merge:.2f} seconds.")

                # Step 4: Summary Statistics
                total_clusters = claims[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'].max()
                unclustered_percentage = (
                    len(claims[claims[f'st_cluster_{space_thres}_{time_thres}_{num_thres}'] == -1]) /
                    len(claims) * 100
                )
                print(f"Total clusters: {total_clusters}")
                print(f"Percentage of unclustered claims: {unclustered_percentage:.2f}%")
                results.append({
                    'space_thres': space_thres,
                    'time_thres': time_thres,
                    'num_thres': num_thres,
                    'total_clusters': total_clusters,
                    'unclustered_percentage': unclustered_percentage
                })

    return claims, pd.DataFrame(results)


def plot_cluster_metric(df, cluster_field, value_field, title):
    """
    Calculate the range of a value field for each cluster, plot its KDE, and return the average range.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cluster_field (str): The column name representing clusters.
        value_field (str): The column name for the metric to calculate.
        title (str): Title for the plot.

    Returns:
        float: The average range for the metric across clusters.
    """
    df = df[df[cluster_field] != -1]
    
    # Calculate the range for each cluster
    cluster_ranges = df.groupby(cluster_field)[value_field].agg(lambda x: x.max() - x.min())

    # Calculate the average range
    average_range = cluster_ranges.mean()
    max_range = cluster_ranges.max()
    variance = cluster_ranges.var()

    # Plot the KDE of the ranges
    plt.figure(figsize=(8, 6))
    sns.kdeplot(cluster_ranges, fill=True)
    plt.title(title)
    plt.xlabel(f"Range of {value_field}")
    plt.ylabel("Density")
    plt.show()

    return average_range, max_range, variance


def check_disaster_grouping(df, st_cluster_fields, disaster_field, title_field, print_complex = False):
    """
    Analyze disaster grouping for each st_cluster field and merge results to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        st_cluster_fields (list): List of column names to analyze.
        disaster_field (str): The column representing disasterNumber.
        title_field (str): The column representing declarationTitle.
    
    Returns:
        None
    """
    output_file = 'Clusters/cluster_sensitivities_di.csv'
    results = []

    for st_cluster in st_cluster_fields:
        print(f"\nAnalyzing st_cluster field: {st_cluster}")
        
        # Group by disasterNumber and determine the unique st_cluster values
        disaster_groups = df.groupby(disaster_field)[st_cluster].unique()

        # Identify cases where disasterNumber is in multiple groups
        multi_group_disasters = [
            disaster for disaster, clusters in disaster_groups.items() 
            if len(clusters) > 1
        ]

        # Exclude cases where the groups are only -1 and one other group
        complex_multi_group_disasters = [
            disaster for disaster in multi_group_disasters
            if not (len(disaster_groups[disaster]) == 2 and -1 in disaster_groups[disaster])
        ]

        if print_complex:
            print("Details for complex cases:")
            for disaster in complex_multi_group_disasters:
                title = df[df[disaster_field] == disaster][title_field].iloc[0]
                clusters = disaster_groups[disaster]
                print(f"  {disaster_field}: {disaster}, Title: {title}, Clusters: {clusters}")

        # Count cases
        count_complex_cases = len(complex_multi_group_disasters)

        # Cluster metrics
        avg_duration, max_duration, var_duration = plot_cluster_metric(
            df, st_cluster, "daysSinceStart", 
            f"Distribution of Cluster Durations for {st_cluster}"
        )
        avg_lat_span, max_lat_span, var_lat_span = plot_cluster_metric(
            df, st_cluster, "centroid_lat", 
            f"Distribution of Event Latitude Span for {st_cluster}"
        )
        avg_lon_span, max_lon_span, var_lon_span = plot_cluster_metric(
            df, st_cluster, "centroid_lon", 
            f"Distribution of Event Longitude Span for {st_cluster}"
        )

        # Prepare results for this st_cluster
        results.append({
            'space_thres': st_cluster.split('_')[2],
            'time_thres': st_cluster.split('_')[3],
            'num_thres': st_cluster.split('_')[4],
            'avg_duration': avg_duration,
            'avg_lat_span': avg_lat_span,
            'avg_lon_span': avg_lon_span,
            'max_duration': max_duration,
            'max_lat_span': max_lat_span,
            'max_lon_span': max_lon_span,
            'var_duration': var_duration,
            'var_lat_span': var_lat_span,
            'var_lon_span': var_lon_span,
            'count_complex_cases': count_complex_cases
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Ensure column types match before merging
    results_df['space_thres'] = results_df['space_thres'].astype(float).astype(str)
    results_df['time_thres'] = results_df['time_thres'].astype(float).astype(str)
    results_df['num_thres'] = results_df['num_thres'].astype(float).astype(str)

    try:
        existing_df = pd.read_csv(output_file)

        # Ensure matching types for the existing DataFrame
        existing_df['space_thres'] = existing_df['space_thres'].astype(float).astype(str)
        existing_df['time_thres'] = existing_df['time_thres'].astype(float).astype(str)
        existing_df['num_thres'] = existing_df['num_thres'].astype(float).astype(str)

        # Merge the DataFrames
        merged_df = pd.merge(existing_df, results_df, on=['space_thres', 'time_thres', 'num_thres'], how='outer')
    except FileNotFoundError:
        # If the file does not exist, use the new DataFrame directly
        merged_df = results_df

    # Save the merged DataFrame to the CSV file
    merged_df.to_csv(output_file, index=False)


def check_claims_grouping(df, st_cluster_fields):
    """
    Analyze disaster grouping for each st_cluster field and merge results to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        st_cluster_fields (list): List of column names to analyze.
    
    Returns:
        None
    """
    output_file = 'Clusters/cluster_sensitivities_cl.csv'
    results = []

    for st_cluster in st_cluster_fields:
        print(f"\nAnalyzing st_cluster field: {st_cluster}")

        # Cluster metrics
        avg_duration, max_duration, var_duration = plot_cluster_metric(
            df, st_cluster, "daysSinceStart", 
            f"Distribution of Cluster Durations for {st_cluster}"
        )
        avg_lat_span, max_lat_span, var_lat_span = plot_cluster_metric(
            df, st_cluster, "latitude", 
            f"Distribution of Event Latitude Span for {st_cluster}"
        )
        avg_lon_span, max_lon_span, var_lon_span = plot_cluster_metric(
            df, st_cluster, "longitude", 
            f"Distribution of Event Longitude Span for {st_cluster}"
        )

        # Prepare results for this st_cluster
        results.append({
            'space_thres': st_cluster.split('_')[2],
            'time_thres': st_cluster.split('_')[3],
            'num_thres': st_cluster.split('_')[4],
            'avg_duration': avg_duration,
            'avg_lat_span': avg_lat_span,
            'avg_lon_span': avg_lon_span,
            'max_duration': max_duration,
            'max_lat_span': max_lat_span,
            'max_lon_span': max_lon_span,
            'var_duration': var_duration,
            'var_lat_span': var_lat_span,
            'var_lon_span': var_lon_span,
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Ensure column types match before merging
    results_df['space_thres'] = results_df['space_thres'].astype(float).astype(str)
    results_df['time_thres'] = results_df['time_thres'].astype(float).astype(str)
    results_df['num_thres'] = results_df['num_thres'].astype(float).astype(str)

    try:
        existing_df = pd.read_csv(output_file)

        # Ensure matching types for the existing DataFrame
        existing_df['space_thres'] = existing_df['space_thres'].astype(float).astype(str)
        existing_df['time_thres'] = existing_df['time_thres'].astype(float).astype(str)
        existing_df['num_thres'] = existing_df['num_thres'].astype(float).astype(str)

        # Merge the DataFrames
        merged_df = pd.merge(existing_df, results_df, on=['space_thres', 'time_thres', 'num_thres'], how='outer')
    except FileNotFoundError:
        # If the file does not exist, use the new DataFrame directly
        merged_df = results_df

    # Save the merged DataFrame to the CSV file
    merged_df.to_csv(output_file, index=False)


# Create heatmaps for combinations of two parameters vs one result
def plot_heatmaps(data, param1, param2, results, figsize=(12, 8)):
    for result in results:
        pivot_table = data.pivot_table(index=param1, columns=param2, values=result, aggfunc='mean')
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Heatmap of {result} vs {param1} and {param2}")
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.show()


# Create scatter/line plots for individual parameter effects
def plot_scatter_lines(data, params, results):
    for param in params:
        for result in results:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x=param, y=result, marker="o", ci=None)
            plt.title(f"{result} vs {param}")
            plt.xlabel(param)
            plt.ylabel(result)
            plt.grid(True)
            plt.show()


# Function to plot 3D surface plots
def plot_3d_surfaces(data, param1, param2, result_columns, figsize=(10, 8)):
    for result in result_columns:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for parameters
        x = data[param1]
        y = data[param2]
        z = data[result]
        xi, yi = np.meshgrid(sorted(x.unique()), sorted(y.unique()))
        zi = data.pivot_table(index=param1, columns=param2, values=result).values

        # Surface plot
        surf = ax.plot_surface(xi, yi, zi, cmap="viridis", edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        ax.set_title(f"3D Surface Plot of {result} vs {param1} and {param2}")
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(result)
        plt.show()
