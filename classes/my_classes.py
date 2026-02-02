import pandas as pd
import numpy as np
from numpy import nan
import geopandas as gpd

from shapely.geometry import Point, box
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import time
import re


import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

import joblib

import warnings

class Format_Flood_Data:
    """
    A class to process flood risk data for geographic points.
    
    Parameters:
    -----------
    flood_gdf : GeoDataFrame
        GeoDataFrame containing flood zone geometries with 'scenario', 'ht_min', 'ht_max' columns
    
    Attributes:
    -----------
    flood_gdf : GeoDataFrame
        The filtered flood data
    buffer_distance_meters : float
        Buffer distance in meters (default: 2500m)
    buffer_distance_degrees : float
        Buffer distance converted to degrees
    sindex : spatial index
        R-tree spatial index for efficient querying
    """
    
    def __init__(self, flood_gdf, data_drias, buffer_distance_meters=2500):
        """
        Initialize the flood data processor.
        
        Parameters:
        -----------
        flood_gdf : GeoDataFrame
            GeoDataFrame containing flood zone geometries
        buffer_distance_meters : float, optional
            Buffer distance in meters for flood risk assessment (default: 2500)
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.original_flood_gdf = flood_gdf
        self.data_drias = data_drias
        self.flood_gdf = None
        self.buffer_distance_meters = buffer_distance_meters
        # Convert meters to degrees (rough conversion at ~45° latitude)
        self.buffer_distance_degrees = buffer_distance_meters / 111000
        self.sindex = None
        
        print(f"FloodDataProcessor initialized with buffer: {buffer_distance_meters}m "
              f"({self.buffer_distance_degrees:.6f} degrees)")
    
    def _create_points_gdf(self, data_drias):
        """
        Create GeoDataFrame with point geometries from longitude/latitude data.
        
        Parameters:
        -----------
        data_drias : DataFrame
            DataFrame containing 'Longitude' and 'Latitude' columns
            
        Returns:
        --------
        GeoDataFrame
            GeoDataFrame with point geometries
        """
        print("\nCreating point geometries...")
        points_gdf = gpd.GeoDataFrame(
            data_drias,
            geometry=[Point(lon, lat) for lon, lat in zip(data_drias['Longitude'], data_drias['Latitude'])],
            crs=self.original_flood_gdf.crs
        )
        print(f"Points created: {len(points_gdf)}")
        return points_gdf
    
    def _filter_flood_data(self, points_gdf, buffer_margin=0.5):
        """
        Filter flood data to the region around the input points.
        
        Parameters:
        -----------
        points_gdf : GeoDataFrame
            GeoDataFrame containing point locations
        buffer_margin : float, optional
            Buffer margin in degrees (default: 0.5, ~50km)
        """
        print("\nFiltering flood data to region...")
        
        # Get bounds of points with buffer
        point_bounds = points_gdf.total_bounds
        min_x = point_bounds[0] - buffer_margin
        min_y = point_bounds[1] - buffer_margin
        max_x = point_bounds[2] + buffer_margin
        max_y = point_bounds[3] + buffer_margin
        
        print(f"Point extent: X=[{point_bounds[0]:.2f}, {point_bounds[2]:.2f}], "
              f"Y=[{point_bounds[1]:.2f}, {point_bounds[3]:.2f}]")
        print(f"Filtering to: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
        
        # Filter flood data
        original_count = len(self.original_flood_gdf)
        self.flood_gdf = self.original_flood_gdf.cx[min_x:max_x, min_y:max_y]
        
        reduction_pct = (1 - len(self.flood_gdf) / original_count) * 100
        print(f"Filtered flood data: {len(self.flood_gdf)} features "
              f"(reduced by {reduction_pct:.1f}%)")
    
    def _create_spatial_index(self):
        """Create spatial index on filtered flood data."""
        print("\nCreating spatial index on filtered data...")
        self.sindex = self.flood_gdf.sindex
        print("Spatial index created!")
    
    def _process_points(self, points_gdf):
        """
        Process each point to determine flood risk.
        
        Parameters:
        -----------
        points_gdf : GeoDataFrame
            GeoDataFrame containing point locations
            
        Returns:
        --------
        DataFrame
            DataFrame with flood risk results for each point
        """
        results = []
        buffer_distance = self.buffer_distance_degrees
        
        start_time = time.time()
        print(f"\nProcessing {len(points_gdf)} points...")
        
        for idx_num, (idx, point_row) in enumerate(points_gdf.iterrows()):
            
            # Progress update every 100 points
            if idx_num % 100 == 0:
                elapsed = time.time() - start_time
                if idx_num > 0:
                    rate = idx_num / elapsed
                    remaining = len(points_gdf) - idx_num
                    est_remaining = remaining / rate / 60
                    
                    print(f"Point {idx_num}/{len(points_gdf)} ({idx_num/len(points_gdf)*100:.1f}%) | "
                          f"Rate: {rate:.2f} pts/sec | Elapsed: {elapsed:.1f}s | "
                          f"Est. remaining: {est_remaining:.1f}min")
            
            point = point_row.geometry
            lon = point_row['Longitude']
            lat = point_row['Latitude']
            
            # Create bounding box
            x, y = point.x, point.y
            bbox = box(x - buffer_distance, y - buffer_distance, 
                      x + buffer_distance, y + buffer_distance)
            
            # Query spatial index
            possible_matches_idx = list(self.sindex.query(bbox))
            
            # Debug first point
            if idx_num == 0:
                print(f"\nFirst point found {len(possible_matches_idx)} candidates")
            
            if not possible_matches_idx:
                results.append({
                    'longitude': lon,
                    'latitude': lat,
                    'scenario': [],
                    'ht': {'high': [], 'mid': [], 'low': []}
                })
                continue
            
            # Get candidates
            possible_matches = self.flood_gdf.iloc[possible_matches_idx]
            
            # Check distances
            point_data = {'scenarios': set(), 'ht_data': defaultdict(list)}
            
            for _, flood_row in possible_matches.iterrows():
                if point.distance(flood_row.geometry) <= buffer_distance:
                    scenario = flood_row['scenario']
                    point_data['scenarios'].add(scenario)
                    point_data['ht_data'][scenario].append([flood_row['ht_min'], flood_row['ht_max']])
            
            # Build result
            if point_data['scenarios']:
                scenarios = list(point_data['scenarios'])
                ht_dict = {
                    'high': point_data['ht_data'].get('high', []),
                    'mid': point_data['ht_data'].get('mid', []),
                    'low': point_data['ht_data'].get('low', [])
                }
            else:
                scenarios = []
                ht_dict = {'high': [], 'mid': [], 'low': []}
            
            results.append({
                'longitude': lon,
                'latitude': lat,
                'scenario': scenarios,
                'ht': ht_dict
            })
        
        # Print summary
        total_time = (time.time() - start_time) / 60
        result_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Total processing time: {total_time:.1f} minutes")
        print(f"Total points processed: {len(result_df)}")
        print(f"Points with flood risk: {len(result_df[result_df['scenario'].apply(len) > 0])}")
        print(f"Points with no flood risk: {len(result_df[result_df['scenario'].apply(len) == 0])}")
        
        print("\nFirst few rows:")
        print(result_df.head())
        
        print("\nExample of point WITH flood risk:")
        with_risk = result_df[result_df['scenario'].apply(len) > 0]
        if len(with_risk) > 0:
            print(with_risk.head(1))
        else:
            print("No points found with flood risk")
        
        return result_df
    
    def process(self, buffer_margin=0.5):
        """
        Main processing method to analyze flood risk for points.
        
        Parameters:
        -----------
        data_drias : DataFrame
            DataFrame containing 'Longitude' and 'Latitude' columns
        buffer_margin : float, optional
            Buffer margin in degrees for filtering flood data (default: 0.5)
            
        Returns:
        --------
        DataFrame
            DataFrame with columns: longitude, latitude, scenario, ht
        """
        # Create point geometries
        points_gdf = self._create_points_gdf(self.data_drias)
        
        # Filter flood data to region
        self._filter_flood_data(points_gdf, buffer_margin)
        
        # Create spatial index
        self._create_spatial_index()
        
        # Process points and return results
        result_df = self._process_points(points_gdf)
        
        return result_df
    
class Format_Clay_Data:
    """
    A class to process clay risk data for geographic points.
    
    Parameters:
    -----------
    data_drias : DataFrame
        DataFrame containing 'Longitude' and 'Latitude' columns
    data_clay : GeoDataFrame
        GeoDataFrame containing clay zone geometries with 'ALEA' and 'NIVEAU' columns
    
    Attributes:
    -----------
    data_drias : DataFrame
        The input point data
    data_clay : GeoDataFrame
        The filtered clay data
    buffer_distance_meters : float
        Buffer distance in meters (default: 1m)
    sindex : spatial index
        R-tree spatial index for efficient querying
    """
    
    def __init__(self, data_drias, data_clay, buffer_distance_meters=1):
        """
        Initialize the clay data processor.
        
        Parameters:
        -----------
        data_drias : DataFrame
            DataFrame containing 'Longitude' and 'Latitude' columns
        data_clay : GeoDataFrame
            GeoDataFrame containing clay zone geometries
        buffer_distance_meters : float, optional
            Buffer distance in meters for clay risk assessment (default: 1)
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.data_drias = data_drias
        self.original_data_clay = data_clay
        self.data_clay = None
        self.buffer_distance_meters = buffer_distance_meters
        self.sindex = None
        self.points_gdf = None
        
        print(f"Format_Clay_Data initialized with buffer: {buffer_distance_meters}m")
    
    def _create_points_gdf(self):
        """
        Create GeoDataFrame with point geometries from longitude/latitude data.
        
        Returns:
        --------
        GeoDataFrame
            GeoDataFrame with point geometries in WGS84 (EPSG:4326)
        """
        print("\nCreating point geometries...")
        points_gdf = gpd.GeoDataFrame(
            self.data_drias,
            geometry=[Point(lon, lat) for lon, lat in zip(
                self.data_drias['Longitude'], 
                self.data_drias['Latitude']
            )],
            crs='EPSG:4326'  # Points are in WGS84
        )
        print(f"Points created: {len(points_gdf)}")
        return points_gdf
    
    def _transform_points(self, points_gdf):
        """
        Transform points to match clay data CRS.
        
        Parameters:
        -----------
        points_gdf : GeoDataFrame
            GeoDataFrame with point geometries in EPSG:4326
            
        Returns:
        --------
        GeoDataFrame
            GeoDataFrame with point geometries transformed to clay data CRS
        """
        print(f"\nTransforming points from EPSG:4326 to {self.original_data_clay.crs}...")
        points_gdf = points_gdf.to_crs(self.original_data_clay.crs)
        return points_gdf
    
    def _filter_clay_data(self, points_gdf, buffer_margin=1000):
        """
        Filter clay data to the region around the input points i.e France Metropolitan.
        
        Parameters:
        -----------
        points_gdf : GeoDataFrame
            GeoDataFrame containing point locations
        buffer_margin : float, optional
            Buffer margin in meters (default: 1000)
        """
        print("\nFiltering clay data to points region...")
        
        # Get bounds of points with buffer
        point_bounds = points_gdf.total_bounds
        min_x = point_bounds[0] - buffer_margin
        min_y = point_bounds[1] - buffer_margin
        max_x = point_bounds[2] + buffer_margin
        max_y = point_bounds[3] + buffer_margin
        
        # Filter clay data
        self.data_clay = self.original_data_clay.cx[min_x:max_x, min_y:max_y]
        
        print(f"Filtered clay data: {len(self.data_clay)} features")
    
    def _create_spatial_index(self):
        """Create spatial index on filtered clay data."""
        print("Creating spatial index...")
        self.sindex = self.data_clay.sindex
        print("Spatial index created!")
    
    def _process_points(self):
        """
        Process each point to determine clay risk.
        
        Returns:
        --------
        DataFrame
            DataFrame with clay risk results for each point
        """
        results = []
        buffer_distance = self.buffer_distance_meters
        
        start_time = time.time()
        print(f"\nProcessing {len(self.points_gdf)} points with buffer_distance={buffer_distance}m...")
        
        for idx_num, (idx, point_row) in enumerate(self.points_gdf.iterrows()):
            
            # Progress update every 100 points
            if idx_num % 100 == 0:
                elapsed = time.time() - start_time
                if idx_num > 0:
                    rate = idx_num / elapsed
                    remaining = len(self.points_gdf) - idx_num
                    est_remaining = remaining / rate / 60
                    
                    print(f"Point {idx_num}/{len(self.points_gdf)} "
                          f"({idx_num/len(self.points_gdf)*100:.1f}%) | "
                          f"Rate: {rate:.2f} pts/sec | Elapsed: {elapsed:.1f}s | "
                          f"Est. remaining: {est_remaining:.1f}min")
            
            point = point_row.geometry
            lon = self.data_drias.loc[idx, 'Longitude']
            lat = self.data_drias.loc[idx, 'Latitude']
            
            # Create bounding box
            x, y = point.x, point.y
            bbox = box(x - buffer_distance, y - buffer_distance, 
                      x + buffer_distance, y + buffer_distance)
            
            # Query spatial index
            possible_matches_idx = list(self.sindex.query(bbox))
            
            # Debug first point
            if idx_num == 0:
                print(f"\nFirst point found {len(possible_matches_idx)} candidates")
            
            if not possible_matches_idx:
                results.append({
                    'longitude': lon,
                    'latitude': lat,
                    'alea': None,
                    'niveau': None
                })
                continue
            
            # Get candidates
            possible_matches = self.data_clay.iloc[possible_matches_idx]
            
            # Collect all matching ALEA and NIVEAU values
            alea_list = []
            niveau_list = []
            
            for _, clay_row in possible_matches.iterrows():
                if point.distance(clay_row.geometry) <= buffer_distance:
                    alea_list.append(clay_row['ALEA'])
                    niveau_list.append(clay_row['NIVEAU'])
            
            results.append({
                'longitude': lon,
                'latitude': lat,
                'alea': max(set(alea_list), key=alea_list.count, default=None),
                'niveau': max(set(niveau_list), key=niveau_list.count, default=None)
            })
        
        # Print summary
        total_time = (time.time() - start_time) / 60
        result_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Total processing time: {total_time:.1f} minutes")
        print(f"Total points processed: {len(result_df)}")
        
        print("\nFirst few rows:")
        print(result_df.head())
        
        return result_df
    
    def process(self, buffer_margin=1000):
        """
        Main processing method to analyze clay risk for points.
        
        Parameters:
        -----------
        buffer_margin : float, optional
            Buffer margin in meters for filtering clay data (default: 1000)
            
        Returns:
        --------
        DataFrame
            DataFrame with columns: longitude, latitude, alea, niveau
        """
        # Create point geometries
        self.points_gdf = self._create_points_gdf()
        
        # Transform points to clay data CRS
        self.points_gdf = self._transform_points(self.points_gdf)
        
        # Filter clay data to region
        self._filter_clay_data(self.points_gdf, buffer_margin)
        
        # Create spatial index
        self._create_spatial_index()
        
        # Process points and return results
        result_df = self._process_points()
        
        return result_df

class Format_Hydrography_Data:
    """
    A class to process hydrography data and calculate distances from points to water bodies.
    
    Parameters:
    -----------
    data_drias : DataFrame
        DataFrame containing 'Longitude' and 'Latitude' columns
    data_hydro : GeoDataFrame
        GeoDataFrame containing hydrography features with 'NomEntiteH' column
    data_ocean : GeoDataFrame
        GeoDataFrame containing ocean/sea geometries
    
    Attributes:
    -----------
    data_drias : DataFrame
        The input point data
    data_hydro : GeoDataFrame
        The hydrography data
    data_ocean : GeoDataFrame
        The ocean/sea data
    gdf_drias : GeoDataFrame
        Points as GeoDataFrame
    """
    
    def __init__(self, data_drias, data_hydro, data_ocean):
        """
        Initialize the hydrography data processor.
        
        Parameters:
        -----------
        data_drias : DataFrame
            DataFrame containing 'Longitude' and 'Latitude' columns
        data_hydro : GeoDataFrame
            GeoDataFrame containing hydrography features with 'NomEntiteH' column
        data_ocean : GeoDataFrame
            GeoDataFrame containing ocean/sea geometries
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.data_drias = data_drias.copy()
        self.data_hydro = data_hydro.copy()
        self.data_ocean = data_ocean.copy()
        self.gdf_drias = None
        self.gdf_drias_proj = None
        self.fleuves = None
        self.rivieres = None
        
        print("Format_Hydrography_Data initialized")
    
    @staticmethod
    def classify_water(name):
        """
        Classify water bodies based on their name.
        
        Parameters:
        -----------
        name : str
            Name of the water body
            
        Returns:
        --------
        str
            Classification of the water body
        """
        if pd.isna(name):
            return "inconnu"
        
        if re.search(r"\bfleuve\b", name):
            return "fleuve"
        
        if re.search(r"\briviere\b", name):
            return "riviere"
        
        if re.search(r"\bcanal\b|\bchenal\b", name):
            return "canal"
        
        if re.search(r"\bestuaire\b", name):
            return "estuaire"
        
        # petits cours d'eau typiques FR
        if re.search(r"\b(rec|ruisseau|ru|vallat|torrent|ravin)\b", name):
            return "petit_cours_eau"
        
        return "autre"
    
    def _classify_hydro_features(self):
        """Classify hydrography features by type."""
        print("\nClassifying water bodies...")
        
        # Clean names
        self.data_hydro["NomEntiteH_clean"] = (
            self.data_hydro["NomEntiteH"]
            .str.lower()
            .str.normalize("NFKD")              # enlève accents
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        
        # Classify
        self.data_hydro["type_cours_eau"] = self.data_hydro["NomEntiteH_clean"].apply(
            self.classify_water
        )
        
        print("Water body classification:")
        print(self.data_hydro["type_cours_eau"].value_counts())
        
        # Separate by type
        self.fleuves = self.data_hydro[self.data_hydro["type_cours_eau"] == "fleuve"]
        self.rivieres = self.data_hydro[
            self.data_hydro["type_cours_eau"].isin(["riviere", "petit_cours_eau"])
        ]
        
        print(f"\nFleuves: {len(self.fleuves)}")
        print(f"Rivières/petits cours d'eau: {len(self.rivieres)}")
    
    def _create_points_gdf(self):
        """Create GeoDataFrame from point data."""
        print("\nCreating point geometries...")
        
        self.gdf_drias = gpd.GeoDataFrame(
            self.data_drias,
            geometry=gpd.points_from_xy(
                self.data_drias["Longitude"], 
                self.data_drias["Latitude"]
            ),
            crs="EPSG:4326"
        )
        
        print(f"Points created: {len(self.gdf_drias)}")
    
    def _project_data(self):
        """Project all data to EPSG:2154 (Lambert 93)."""
        print("\nProjecting data to EPSG:2154...")
        
        self.gdf_drias_proj = self.gdf_drias.to_crs("EPSG:2154")
        self.fleuves_proj = self.fleuves.to_crs("EPSG:2154")
        self.rivieres_proj = self.rivieres.to_crs("EPSG:2154")
        self.ocean_proj = self.data_ocean.to_crs("EPSG:2154")
        
        # Add temporary ID for grouping
        self.gdf_drias_proj = self.gdf_drias_proj.reset_index(drop=True)
        self.gdf_drias_proj['temp_id'] = self.gdf_drias_proj.index
        
        print("Projection complete")
    
    def _calculate_distances(self):
        """Calculate distances to different water body types."""
        print("\nCalculating distances...")
        
        # Distance to fleuves (rivers)
        print("Calculating distance to fleuves...")
        nearest_fleuve = self.gdf_drias_proj.sjoin_nearest(
            self.fleuves_proj, 
            distance_col="dist_fleuve_m", 
            how="left"
        )
        nearest_fleuve = nearest_fleuve.groupby('temp_id').first().reset_index()
        self.data_drias['dist_fleuve_m'] = nearest_fleuve['dist_fleuve_m'].values
        
        # Distance to rivières (streams)
        print("Calculating distance to rivières...")
        nearest_riviere = self.gdf_drias_proj.sjoin_nearest(
            self.rivieres_proj, 
            distance_col="dist_riviere_m", 
            how="left"
        )
        nearest_riviere = nearest_riviere.groupby('temp_id').first().reset_index()
        self.data_drias['dist_riviere_m'] = nearest_riviere['dist_riviere_m'].values
        
        # Distance to ocean/sea
        print("Calculating distance to ocean/sea...")
        nearest_ocean = self.gdf_drias_proj.sjoin_nearest(
            self.ocean_proj, 
            distance_col="dist_cote_m", 
            how="left"
        )
        nearest_ocean = nearest_ocean.groupby('temp_id').first().reset_index()
        self.data_drias['dist_cote_m'] = nearest_ocean['dist_cote_m'].values
        
        print("Distance calculations complete")
    
    def _convert_to_kilometers(self):
        """Convert distances from meters to kilometers."""
        print("\nConverting distances to kilometers...")
        
        self.data_drias['dist_fleuve_km'] = self.data_drias['dist_fleuve_m'] / 1000
        self.data_drias['dist_riviere_km'] = self.data_drias['dist_riviere_m'] / 1000
        self.data_drias['dist_cote_km'] = self.data_drias['dist_cote_m'] / 1000
        
        # Drop meter columns
        self.data_drias.drop(
            columns=['dist_fleuve_m', 'dist_riviere_m', 'dist_cote_m'], 
            inplace=True
        )
        
        print("Conversion complete")
    
    def process(self):
        """
        Main processing method to calculate distances to water bodies.
        
        Returns:
        --------
        DataFrame
            Modified data_drias with added columns: dist_fleuve_km, dist_riviere_km, dist_cote_km
        """
        # Classify hydrography features
        self._classify_hydro_features()
        
        # Create point geometries
        self._create_points_gdf()
        
        # Project all data to Lambert 93
        self._project_data()
        
        # Calculate distances
        self._calculate_distances()
        
        # Convert to kilometers
        self._convert_to_kilometers()
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Added columns: dist_fleuve_km, dist_riviere_km, dist_cote_km")
        print("\nFirst few rows:")
        print(self.data_drias[['Longitude', 'Latitude', 'dist_fleuve_km', 
                                'dist_riviere_km', 'dist_cote_km']].head())
        
        return self.data_drias
    

class Prepare_Data:
    """
    Classe pour préparer les données de risque d'inondation en fusionnant
    les données Flood, Drias et Clay.
    """
    
    def __init__(self, data_flood_path, data_drias_path, data_clay_path):
        """
        Initialise la classe avec les chemins des fichiers de données.
        
        Parameters:
        -----------
        data_flood_path : str
            Chemin vers le fichier flood_risk_results.csv
        data_drias_path : str
            Chemin vers le fichier RCP_4.5_with_distance.csv
        data_clay_path : str
            Chemin vers le fichier clay_risk_results.csv
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.data_flood_path = data_flood_path
        self.data_drias_path = data_drias_path
        self.data_clay_path = data_clay_path
        
        # Chargement des données brutes
        self.data_flood = None
        self.data_drias = None
        self.data_clay = None
        
        # Données préparées
        self.data_flood_high = None
        self.data_flood_mid = None
        self.data_flood_low = None
        self.data_merged = None
        
    def load_data(self):
        """Charge les données depuis les fichiers CSV."""
        print("Chargement des données...")
        # Charger les données Flood
        self.data_flood = pd.read_csv(self.data_flood_path)
        
        # Charger les données Drias
        self.data_drias = pd.read_csv(self.data_drias_path, sep=";")
        
        # Charger les données Clay
        self.data_clay = gpd.read_file(self.data_clay_path)
        self.data_clay = self.data_clay.astype({'longitude': 'Float64', 'latitude': 'Float64'})
        
        print("Données chargées avec succès!")
        print("")

    def prepare_clay_data(self):
        """Prépare les données sur les sols argileux."""
        print("Préparation des données clay...")
        
        # Remplacer les valeurs textuelles par des valeurs numériques
        self.data_clay["alea"] = self.data_clay["alea"].replace({
            "Faible": 1,
            "Moyen": 2,
            "Fort": 3
        })
        
        # Supprimer la colonne 'niveau'
        self.data_clay.drop(columns=['niveau'], inplace=True)
        
        # Remplacer les chaînes vides par NaN puis remplir avec 0
        self.data_clay["alea"] = self.data_clay["alea"].replace("", pd.NA)
        self.data_clay["alea"] = self.data_clay["alea"].fillna(0)
        self.data_clay = self.data_clay.astype({'alea': 'int32'})
        
        print("Données clay préparées!")
        print("")

    def prepare_flood_data(self):
        """Prépare les données de risque d'inondation pour les 3 scénarios."""
        print("Préparation des données Flood...")
        
        # Scénario High
        self.data_flood_high = self.data_flood.copy()
        self.data_flood_high["scenario"] = self.data_flood_high["scenario"].apply(
            lambda x: "High" if "high" in x else None
        )
        self.data_flood_high["ht"] = self.data_flood_high["ht"].apply(lambda x: eval(x)['high'])
        self.data_flood_high["ht"] = self.data_flood_high["ht"].apply(
            lambda x: max(set([tuple(l) for l in x]), key=x.count, default=None)
        )
        self.data_flood_high["ht_min"] = self.data_flood_high["ht"].str[0]
        self.data_flood_high["ht_max"] = self.data_flood_high["ht"].str[-1]
        self.data_flood_high.drop(columns=["ht"], inplace=True)
        
        # Scénario Mid
        self.data_flood_mid = self.data_flood.copy()
        self.data_flood_mid["scenario"] = self.data_flood_mid["scenario"].apply(
            lambda x: "Mid" if "mid" in x else None
        )
        self.data_flood_mid["ht"] = self.data_flood_mid["ht"].apply(lambda x: eval(x)['mid'])
        self.data_flood_mid["ht"] = self.data_flood_mid["ht"].apply(
            lambda x: max(set([tuple(l) for l in x]), key=x.count, default=None)
        )
        self.data_flood_mid["ht_min"] = self.data_flood_mid["ht"].str[0]
        self.data_flood_mid["ht_max"] = self.data_flood_mid["ht"].str[-1]
        self.data_flood_mid.drop(columns=["ht"], inplace=True)
        
        # Scénario Low
        self.data_flood_low = self.data_flood.copy()
        self.data_flood_low["scenario"] = self.data_flood_low["scenario"].apply(
            lambda x: "Low" if "low" in x else None
        )
        self.data_flood_low["ht"] = self.data_flood_low["ht"].apply(lambda x: eval(x)['low'])
        self.data_flood_low["ht"] = self.data_flood_low["ht"].apply(
            lambda x: max(set([tuple(l) for l in x]), key=x.count, default=None)
        )
        self.data_flood_low["ht_min"] = self.data_flood_low["ht"].str[0]
        self.data_flood_low["ht_max"] = self.data_flood_low["ht"].str[-1]
        self.data_flood_low.drop(columns=["ht"], inplace=True)
        
        print("Données Flood préparées pour les 3 scénarios!")
        print("")

    def join_datasets(self):
        """Fusionne tous les datasets (Flood, Drias, Clay)."""
        print("Fusion des datasets...")
        
        # 1. Concaténer les 3 scénarios Flood
        flood_list = [self.data_flood_high, self.data_flood_mid, self.data_flood_low]
        data_flood_concat = pd.concat(flood_list)
        
        # 2. Joindre avec les données Drias (période H1)
        self.data_merged = data_flood_concat.merge(
            self.data_drias[self.data_drias['Période'] == 'H1'],
            left_on=["latitude", "longitude"],
            right_on=["Latitude", "Longitude"],
            how="inner"
        )
        
        # 3. Joindre avec les données Clay (spatial join)
        gdf1 = gpd.GeoDataFrame(
            self.data_merged,
            geometry=gpd.points_from_xy(self.data_merged.longitude, self.data_merged.latitude),
            crs="EPSG:4326"
        )
        gdf2 = gpd.GeoDataFrame(
            self.data_clay,
            geometry=gpd.points_from_xy(self.data_clay.longitude, self.data_clay.latitude),
            crs="EPSG:4326"
        )
        
        # Reprojection en EPSG:3857 pour mesures en mètres
        gdf1 = gdf1.to_crs(3857)
        gdf2 = gdf2.to_crs(3857)
        
        # Spatial join nearest avec distance max de 50m
        self.data_merged = gpd.sjoin_nearest(
            gdf1,
            gdf2.drop(columns=['latitude', 'longitude']),
            max_distance=50,
            distance_col="distance_m"
        )
        
        print("Datasets fusionnés!")
        print("")

    def clean_merged_data(self):
        """Nettoie les données fusionnées."""
        print("Nettoyage des données fusionnées...")
        
        # Supprimer les colonnes inutiles
        columns_to_drop = [
            'Unnamed: 0', 'Unnamed: 17', 'Point', 'longitude', 'latitude',
            'Longitude', 'Latitude', 'Contexte', 'Période', 'field_1',
            'distance_m', 'index_right', 'geometry'
        ]
        
        # Ne supprimer que les colonnes qui existent
        existing_columns = [col for col in columns_to_drop if col in self.data_merged.columns]
        self.data_merged.drop(columns=existing_columns, inplace=True)
        
        # Encoder les scénarios
        self.data_merged["scenario"] = self.data_merged["scenario"].replace({
            "Low": 3,
            "Mid": 2,
            "High": 1
        })
        
        # Remplir les NaN
        self.data_merged["ht_min"] = self.data_merged["ht_min"].fillna(0)
        self.data_merged["ht_max"] = self.data_merged["ht_max"].fillna(0)
        self.data_merged["scenario"] = self.data_merged["scenario"].fillna(0)
        
        print("Données nettoyées!")
        print("")

    def prepare_all(self):
        """
        Exécute toutes les étapes de préparation des données.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame contenant toutes les données préparées et fusionnées
        """
        self.load_data()
        self.prepare_clay_data()
        self.prepare_flood_data()
        self.join_datasets()
        self.clean_merged_data()
        
        print("=" * 50)
        print("Préparation des données terminée!")
        print(f"Shape du dataset final: {self.data_merged.shape}")
        print("=" * 50)
        
        return self.data_merged
    
    def get_merged_data(self):
        """Retourne les données fusionnées."""
        if self.data_merged is None:
            raise ValueError("Les données n'ont pas encore été préparées. Appelez prepare_all() d'abord.")
        return self.data_merged


class DenseNetClassifier(nn.Module):
    """Architecture de réseau de neurones pour la classification"""
    def __init__(self, input_dim: int, num_classes_min: int, num_classes_max: int):
        super().__init__()
        self.ht_min = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes_min)
        )
        self.ht_max = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes_max)
        )

    def forward(self, x):
        return self.ht_min(x), self.ht_max(x)


class FloodScoring:
    """
    Classe pour calculer les prédictions de risque d'inondation
    
    Parameters
    ----------
    scaler : sklearn scaler
        Scaler pour normaliser les données
    model_height : DenseNetClassifier
        Modèle pour prédire les hauteurs d'eau
    model_class : dict
        Dictionnaire contenant les modèles de classification par risque
    data_drias : pd.DataFrame
        Données climatiques DRIAS
    data_clay : gpd.GeoDataFrame
        Données sur le risque d'argile
    """
    
    def __init__(self, scaler, model_height, model_class: Dict, 
                 data_drias: pd.DataFrame, data_clay: gpd.GeoDataFrame):
        
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.scaler = scaler
        self.model_height = model_height
        self.model_class = model_class
        self.data_drias = data_drias
        self.data_clay = data_clay
        
        # Colonnes à scaler
        self.scale_col = ['NORPAV', 'NORRR', 'NORRR1MM', 'NORPN20MM', 'NORPFL90',
                          'NORPXCDD', 'NORPINT', 'NORPQ90', 'NORPQ99', 'NORRR99', 
                          'NORHUSAV', 'NORETPC', 'dist_fleuve_km', 'dist_riviere_km', 
                          'dist_cote_km']
        
        # Ordre des features
        self.feature_order = ['Flood_risk', 'NORPAV', 'NORRR', 'NORRR1MM',
                              'NORPN20MM', 'NORPFL90', 'NORPXCDD', 'NORPINT', 'NORPQ90',
                              'NORPQ99', 'NORRR99', 'NORHUSAV', 'NORETPC', 'dist_fleuve_km',
                              'dist_riviere_km', 'dist_cote_km', 'alea']
        
        # Classes de risque
        self.risk_class = {'low': 3.0, 'mid': 2.0, 'high': 1.0}
        
        # Préparer les données
        self.df_merged = self._prepare_data()
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prépare et fusionne les données DRIAS et clay"""
        # Traiter les données clay
        data_clay = self.data_clay.copy()
        data_clay["alea"] = data_clay["alea"].replace({"Faible": 1, "Moyen": 2, "Fort": 3})
        data_clay.drop(columns=['niveau'], inplace=True)
        data_clay["alea"] = data_clay["alea"].replace("", pd.NA)
        data_clay["alea"] = data_clay["alea"].fillna(0)
        data_clay = data_clay.astype({'alea': 'int32'})
                
        # Créer les GeoDataFrames
        gdf1 = gpd.GeoDataFrame(
            self.data_drias, 
            geometry=gpd.points_from_xy(self.data_drias.Longitude, self.data_drias.Latitude),
            crs="EPSG:4326"
        )
        gdf2 = gpd.GeoDataFrame(
            data_clay,
            geometry=gpd.points_from_xy(data_clay.longitude, data_clay.latitude),
            crs="EPSG:4326"
        )
        
        # Reprojeter
        gdf1 = gdf1.to_crs(3857)
        gdf2 = gdf2.to_crs(3857)
        
        # Joindre
        data_merged = gpd.sjoin_nearest(
            gdf1, 
            gdf2.drop(columns=['latitude', 'longitude']),
            max_distance=50,
            distance_col="distance_m"
        )
        
        # Nettoyer
        cols_to_drop = [c for c in ['Unnamed: 17', 'Point', 'Contexte', 'field_1', 
                                     'distance_m', 'index_right', 'geometry'] 
                       if c in data_merged.columns]
        data_merged.drop(columns=cols_to_drop, inplace=True)
        
        return data_merged
    
    def _prep_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prépare les données pour la prédiction"""
        df = data.copy()
        periode = df["Période"]
        latitude, longitude = df['Latitude'], df['Longitude']
        X = df.drop(columns=['Période', 'Longitude', 'Latitude'])
        X = X.select_dtypes(include=["int32", "int64", "float64"])
        return X, periode, latitude, longitude
    
    def _create_results(self, data: pd.DataFrame, risk_value: float, 
                       model_class, model_ht) -> pd.DataFrame:
        """Crée les résultats de prédiction pour une classe de risque"""
        # Préparer les données
        df, periode, latitude, longitude = self._prep_data(data)
        
        # Prédire la classe de risque
        pred_class = model_class.predict(df)
        df['Flood_risk'] = pred_class
        df['Flood_risk'].replace(1, risk_value)
        
        # Scaler
        df[self.scale_col] = self.scaler.transform(df[self.scale_col])
        
        # Ordonner les colonnes
        df = df[self.feature_order]
        
        # Prédire les hauteurs
        df_for_ht = torch.tensor(df.to_numpy(), dtype=torch.float32)
        model_ht.eval()
        with torch.no_grad():
            ht_min_prob, ht_max_prob = model_ht(df_for_ht)
            ht_min_pred = torch.argmax(ht_min_prob, dim=1)
            ht_max_pred = torch.argmax(ht_max_prob, dim=1)
        
        # Créer le dataframe de résultats
        df['ht_min'] = ht_min_pred.numpy()
        df['ht_max'] = ht_max_pred.numpy()
        df['Période'] = periode
        df['longitude'] = longitude.values
        df['latitude'] = latitude.values
        
        return df
    
    def run(self) -> pd.DataFrame:
        """
        Execute le scoring et retourne df_results
        
        Returns
        -------
        pd.DataFrame
            DataFrame avec les colonnes: Flood_risk, ht_min, ht_max, Période, 
            longitude, latitude, et toutes les features
        """
        results = {}
        
        for risk, model in self.model_class.items():
            result = self._create_results(
                self.df_merged, 
                self.risk_class[risk], 
                model, 
                self.model_height
            )
            
            # Ajuster les valeurs de Flood_risk
            if risk == 'mid':
                result['Flood_risk'] = result['Flood_risk'].replace(1, 2)
            elif risk == 'low':
                result['Flood_risk'] = result['Flood_risk'].replace(1, 3)
            
            results[risk] = result
        
        # Combiner tous les résultats
        df_results = pd.DataFrame()
        for risk, df in results.items():
            if df_results.empty:
                df_results = df.copy()
            else:
                df_results = pd.concat([df_results, df])
        
        return df_results


class ScoreCreation:
    """
    Classe pour créer des scores de risque à partir d'un portfolio
    
    Parameters
    ----------
    localisation : gpd.GeoDataFrame
        GeoDataFrame contenant les communes avec leur géométrie
    current_year : int, optional
        Année de référence pour le calcul, default=2025
    buffer_m : float, optional
        Rayon de buffer autour des communes en mètres, default=1000
    """
    
    def __init__(self, localisation: gpd.GeoDataFrame, 
                 current_year: int = 2025, buffer_m: float = 1000):
        
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.localisation = localisation
        self.current_year = current_year
        self.buffer_m = buffer_m
        
        # Mapping Flood_risk → période de retour (en années)
        self.T_map = {1: 30, 2: 200, 3: 1000, 0: 0}
    
    def read_portfolio(self, path: str) -> pd.DataFrame:
        """
        Lit un fichier portfolio Excel
        
        Parameters
        ----------
        path : str
            Chemin vers le fichier Excel
            
        Returns
        -------
        pd.DataFrame
            DataFrame du portfolio
        """
        try:
            # Essayer de lire avec header
            df = pd.read_excel(path)
            df.drop(df.columns[0], axis=1, inplace=True)
            expected_cols = ['id', 'commune', 'INSEE_COM', 'sector', 'maturite_pret', 'encours']
            if sum(df.columns[:6] != expected_cols):
                # Renommer si les colonnes sont différentes
                df = df.iloc[:, :6]
                df = df.rename(columns=dict(zip(df.columns, expected_cols)))
        except:
            # Fallback: pas de header
            df = pd.read_excel(
                path, 
                header=None, 
                names=['id', 'commune', 'INSEE_COM', 'sector', 'maturite_pret', 'encours']
            )
        return df
    
    def prep_data(self, portfolio_file: str, scenario: str = 'RCP4.5') -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Prépare les données de prédiction et de portfolio
        
        Parameters
        ----------
        portfolio_file : str
            Nom du fichier portfolio
        scenario : str, optional
            Scénario climatique ('RCP2.6', 'RCP4.5', 'RCP8.5'), default='RCP4.5'
            
        Returns
        -------
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            (prediction, portfolio) en tant que GeoDataFrames
        """
        pred_url = f'prediction_data/{scenario}_prediction.csv'
        portfolio_url = rf'portefeuille\{portfolio_file}'

        # Load Data
        prediction = pd.read_csv(pred_url)
        portfolio = self.read_portfolio(portfolio_url)
        
        # Add geopandas geometry to portfolio
        portfolio["INSEE_COM"] = portfolio["INSEE_COM"].astype("category")
        portfolio["INSEE_COM"] = portfolio["INSEE_COM"].astype(str).str.strip().str.zfill(5)
        
        localisation_copy = self.localisation.copy()
        localisation_copy["INSEE_COM"] = localisation_copy["INSEE_COM"].astype(str).str.strip().str.zfill(5)
        
        portfolio = portfolio.merge(localisation_copy, on='INSEE_COM')
        portfolio = gpd.GeoDataFrame(portfolio, geometry="geometry", crs="EPSG:2154")
        
        # Clean prediction from no data (denoted by 1000)
        prediction['ht_max'] = np.where(
            prediction['ht_max'] == 1000, 
            prediction['ht_min'], 
            prediction['ht_max']
        )
        
        # Add geopandas geometry to prediction
        prediction["geometry"] = gpd.points_from_xy(prediction["longitude"], prediction["latitude"])
        prediction = gpd.GeoDataFrame(prediction, geometry="geometry", crs="EPSG:4326")
        prediction = prediction.to_crs(epsg=2154)
        
        return prediction, portfolio
    
    def _compute_risk_multi_period(self, join: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Calcule le risque en tenant compte des horizons climatiques et des différents types de risque.
        
        Pour chaque type de risque (High, Mid, Low) :
            - Calcule P (probabilité d'inondation) sur tous les horizons H1, H2, H3
            - Calcule damage_factor (moyenne pondérée sur les horizons)
            - Risk = P × damage_factor
        
        Le risque total est la SOMME des risques pour chaque type.

        Parameters
        ----------
        join : GeoDataFrame
            Résultat du sjoin, contenant les colonnes :
            INSEE_COM, geometry, Période, Flood_risk, T, ht_min, ht_max, n_years, maturite_pret, encours

        Returns
        -------
        pd.DataFrame
            Une ligne par point géographique unique (INSEE_COM + geometry)
            avec la colonne : risk (somme des risques de tous les types)
        """
        H1_END = 2050
        H2_END = 2070

        # --- Extract duration per horizon per asset ---
        base = (
            join[["INSEE_COM", "geometry", "n_years"]]
            .drop_duplicates()
            .copy()
        )
        n = base["n_years"]
        base["n_H1"] = (H1_END - self.current_year)  # 25 everywhere
        base["n_H1"] = base["n_H1"].clip(lower=0, upper=n)

        base["n_H2"] = (H2_END - H1_END)  # 20 everywhere
        base["n_H2"] = base["n_H2"].clip(lower=0, upper=(n - base["n_H1"]).clip(lower=0))

        base["n_H3"] = (n - base["n_H1"] - base["n_H2"]).clip(lower=0)

        # --- Process each flood risk type separately ---
        risk_types = join["Flood_risk"].dropna().unique()
        
        all_risks = []
        
        for risk_type in risk_types:
            # Filter data for this risk type
            risk_data = join[join["Flood_risk"] == risk_type].copy()
            
            # --- Get data per period for this risk type ---
            def pivot_period_for_risk(df, period_label):
                sub = (
                    df[df["Période"] == period_label][["INSEE_COM", "geometry", "T", "ht_min", "ht_max"]]
                    .drop_duplicates()
                    .rename(columns={
                        "T": f"T_{period_label}",
                        "ht_min": f"ht_min_{period_label}",
                        "ht_max": f"ht_max_{period_label}",
                    })
                )
                return sub

            h1 = pivot_period_for_risk(risk_data, "H1")
            h2 = pivot_period_for_risk(risk_data, "H2")
            h3 = pivot_period_for_risk(risk_data, "H3")

            # Merge on (INSEE_COM, geometry)
            merged = (
                base.copy()
                .merge(h1, on=["INSEE_COM", "geometry"], how="left")
                .merge(h2, on=["INSEE_COM", "geometry"], how="left")
                .merge(h3, on=["INSEE_COM", "geometry"], how="left")
            )

            # --- Formula for P for this risk type ---
            # P_Hi = (1 - 1/T_Hi) ^ n_Hi
            # if T is NaN or <= 0 on a horizon then survival = 1 (no flood of this type)
            # P = 1 - survival_H1 * survival_H2 * survival_H3
            for h in ["H1", "H2", "H3"]:
                T_col = f"T_{h}"
                n_col = f"n_{h}"
                merged[f"surv_{h}"] = np.where(
                    merged[T_col] > 0,
                    (1 - 1 / merged[T_col]) ** merged[n_col],
                    1.0
                )

            merged["P"] = 1 - (merged["surv_H1"] * merged["surv_H2"] * merged["surv_H3"])

            # --- Formula for Damage Factor for this risk type ---
            # df_Hi = 1 / (1 + exp(1 - (ht_min_Hi + ht_max_Hi) / 2))
            # damage_factor = (n_H1 * df_H1 + n_H2 * df_H2 + n_H3 * df_H3) / n_total
            for h in ["H1", "H2", "H3"]:
                merged[f"df_{h}"] = 1 / (
                    1 + np.exp(1 - (merged[f"ht_min_{h}"].fillna(0) + merged[f"ht_max_{h}"].fillna(0)) / 2)
                )
                # Set damage to 0 if ht_min and ht_max are NaN
                mask_missing = merged[f"ht_min_{h}"].isna() & merged[f"ht_max_{h}"].isna()
                merged.loc[mask_missing, f"df_{h}"] = 0

            n_total = merged["n_years"].clip(lower=1)  # avoid division by 0
            merged["damage_factor"] = (
                merged["n_H1"] * merged["df_H1"]
                + merged["n_H2"] * merged["df_H2"]
                + merged["n_H3"] * merged["df_H3"]
            ) / n_total

            # --- Risk for this specific risk type ---
            merged[f"risk_{risk_type}"] = merged["P"] * merged["damage_factor"]
            
            all_risks.append(merged[["INSEE_COM", "geometry", f"risk_{risk_type}"]])

        # --- Sum all risks together ---
        # Start with base locations
        result = base[["INSEE_COM", "geometry"]].copy()
        result["risk"] = 0.0
        
        # Add each risk type's contribution
        for risk_df in all_risks:
            result = result.merge(risk_df, on=["INSEE_COM", "geometry"], how="left")
            risk_col = [col for col in risk_df.columns if col.startswith("risk_")][0]
            result["risk"] = result["risk"] + result[risk_col].fillna(0)
            result = result.drop(columns=[risk_col])

        return result[["INSEE_COM", "geometry", "risk"]]


    def get_score(self, portfolio_file: str, scenario: str = 'RCP4.5') -> gpd.GeoDataFrame:
        """
        Calcule le score de risque financier pour un portfolio.

        Le risque est calculé séparément pour chaque type de risque d'inondation 
        (High, Mid, Low), puis sommé pour obtenir le risque total.
        
        Pour chaque type de risque, on calcule :
            - P : probabilité d'inondation sur les horizons H1, H2, H3
            - damage_factor : facteur de dommage moyen pondéré
            - risk = P × damage_factor
        
        Risk_total = risk_High + risk_Mid + risk_Low

        Parameters
        ----------
        portfolio_file : str
            Nom du fichier portfolio
        scenario : str, optional
            Scénario climatique, default='RCP4.5'

        Returns
        -------
        gpd.GeoDataFrame
            Portfolio avec colonnes de risque ajoutées
        """
        # Prepare Data
        prediction, portfolio = self.prep_data(portfolio_file, scenario=scenario)

        # Map Flood_risk to Return Period T
        prediction["T"] = prediction["Flood_risk"].map(self.T_map)

        # Buffer on municipality
        portfolio_buffered = portfolio.copy()
        if self.buffer_m > 0:
            portfolio_buffered["geometry"] = portfolio_buffered.geometry.buffer(self.buffer_m)

        # Spatial Join
        join = gpd.sjoin(
            prediction,
            portfolio_buffered[["INSEE_COM", "geometry", "maturite_pret", "encours"]],
            how="inner",
            predicate="intersects"
        )

        # Get years number before maturity
        join["n_years"] = (join["maturite_pret"] - self.current_year).clip(lower=0)

        # Drop duplicates
        join.drop_duplicates(inplace=True)

        # Get risk accounting for periods H1/H2/H3 and all risk types
        risk_per_point = self._compute_risk_multi_period(join)

        # Get mean per commune
        risk_commune = (
            risk_per_point
            .groupby("INSEE_COM")["risk"]
            .mean()
            .reset_index()
            .rename(columns={"risk": "risk_mean"})
        )

        # Merge with portfolio
        portfolio = portfolio.merge(risk_commune, on="INSEE_COM", how="left")
        portfolio["risk_mean"] = portfolio["risk_mean"].fillna(0)

        # Calculate Financial Risk
        portfolio["Risk_financier"] = portfolio["risk_mean"] * portfolio["encours"]

        return portfolio
    
    def summarize_prediction(self, portfolio_file: str, 
                           scenarios: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare predictions between different scenarios
        
        Parameters
        ----------
        portfolio_file : str
            Name of the portfolio file
        scenarios : List[str]
            List of scenarios to compare (e.g., ['RCP2.6', 'RCP4.5', 'RCP8.5'])
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (comparison_classes, comparison_heights)
        """
        results = []
        height_results = []
        
        for scenario in scenarios:
            portfolio = self.get_score(portfolio_file, scenario=scenario)
            
            # Statistics by risk class
            result = {
                'scenario': scenario,
                'encours_total': portfolio['encours'].sum(),
                'risk_mean_total': portfolio['risk_mean'].sum(),
                'Risk_financier_total': portfolio['Risk_financier'].sum()
            }
            results.append(result)
            
            # Prepare height data
            prediction, _ = self.prep_data(portfolio_file, scenario=scenario)
            
            # Count by (ht_min, ht_max) pair
            height_counts = (
                prediction.groupby(['ht_min', 'ht_max'])
                .size()
                .reset_index(name='count')
            )
            height_counts['scenario'] = scenario
            height_results.append(height_counts)
        
        # Create comparison DataFrames
        df_classes = pd.DataFrame(results)
        df_heights = pd.concat(height_results, ignore_index=True)
        
        return df_classes, df_heights
    
    def summarize_scenario(self, scenarios: List[str], 
                      folder: str = "prediction_data",
                      scaler_path: str = "model/ht_scaler.pkl") -> pd.DataFrame:
        """
        Descale and compare descriptive statistics between scenarios AND periods
        
        Returns
        -------
        pd.DataFrame
            Descriptive statistics by scenario and period
        """
        scale_col = ['NORPAV', 'NORRR', 'NORRR1MM', 'NORPN20MM', 'NORPFL90',
                    'NORPXCDD', 'NORPINT', 'NORPQ90', 'NORPQ99', 'NORRR99', 'NORHUSAV',
                    'NORETPC', 'dist_fleuve_km', 'dist_riviere_km', 'dist_cote_km']
        
        scaler = joblib.load(scaler_path)
        
        cols_to_summarize = ["Flood_risk", "ht_min", "ht_max", "alea", "Période"] + scale_col
        
        all_stats = []
        
        for scenario in scenarios:
            pred_file = f"{folder}/{scenario}_prediction.csv"
            df = pd.read_csv(pred_file)
            
            df = df[cols_to_summarize].copy()
            
            df['ht_max'] = np.where(df['ht_max'] == 1000, df['ht_min'], 0)
            
            df_scaled = df[scale_col].values
            df[scale_col] = scaler.inverse_transform(df_scaled)
            
            # group by period
            for periode, g in df.groupby("Période"):
                g = g.drop(columns='Période')

                stats = g.describe().T[['mean', 'std', 'min', 'max']].copy()
                stats['scenario'] = scenario
                stats['periode'] = periode
                
                quantiles = g.quantile([0.25, 0.5, 0.75]).T
                quantiles.columns = ['q1', 'q2', 'q3']
                
                stats = stats.merge(quantiles, left_index=True, right_index=True)
                stats.reset_index(inplace=True)
                stats.rename(columns={'index': 'variable'}, inplace=True)
                
                all_stats.append(stats)
        
        summary_df = pd.concat(all_stats, ignore_index=True)
        
        return summary_df


class VisualizeResults:
    """
    Class to visualize prediction results
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of results (normally df_results)
    periode : str, optional
        Period to visualize ('H1', 'H2', 'H3'), default='H1'
    show_table : bool, optional
        Show statistics tables, default=False
    """
    
    def __init__(self, df: pd.DataFrame, periode: str = 'H1', show_table: bool = False):

        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.df = df
        self.periode = periode
        self.show_table = show_table
    
    def plot(self):
        """Generate the visualization plot"""
        # Filter by period
        df_filtered = self.df[self.df['Période'] == self.periode]
        
        # Count occurrences of each ht_min / ht_max combination
        counts = df_filtered.groupby(['ht_min', 'ht_max']).size().reset_index(name='count')
        
        # Remove extreme values
        counts_filtered = counts[counts['ht_max'] < 6]
        
        # Create the bubble plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            counts_filtered['ht_min'],
            counts_filtered['ht_max'],
            s=counts_filtered['count'] * 10,  # size proportional to count
            alpha=0.6,
            color='teal',
            edgecolors='w'
        )
        plt.xlabel('ht_min')
        plt.ylabel('ht_max')
        plt.title(f'ht_min vs ht_max for period {self.periode} (size = occurrences)')
        plt.grid(True)
        plt.show()
        
        # Show tables if requested
        if self.show_table:
            print(f"\nStatistics for period {self.periode}:")
            print("\nFlood_risk distribution:")
            print(df_filtered['Flood_risk'].value_counts())
            print("\n(ht_max, ht_min) distribution:")
            print(df_filtered[['ht_max', 'ht_min']].value_counts())

    def plot_mean_evolution_by_period(self):
        """
        Plot the evolution of mean values of given variables over periods (H1, H2, H3)
        
        Parameters
        ----------
        variables : List[str]
            List of variable names to visualize
        """
        import matplotlib.pyplot as plt
        
        # Filter relevant periods
        df = self.df
        
        for var, g in df.groupby('variable'):
            plt.figure(figsize=(5, 3))
            #plt.plot(g['periode'], g['mean'], marker='o', color=g['scenario'], label=var)
            sns.lineplot(data=g, x="periode", y="mean", hue="scenario")
            plt.title(f"Évolution de la moyenne de {var} par période")
            plt.xlabel("Période")
            plt.ylabel("Valeur moyenne")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        

