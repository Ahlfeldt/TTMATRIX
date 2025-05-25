import os
import subprocess
import sys

# === USER SETTINGS ===
working_dir = r"A:\Research\TTMATRIX-toolkit"  # Select your working directory here
points_file = "B4m_com_ll.shp"                 # Select the point shapefile containing locations here (origins/destinations)
stations_file = "UBahn2020_stops_ll.shp"       # Select the point shapefile containing stations to enter and exit the network here
network_file = "UBahn2020_lines_ll.shp"        # Select the polyline shapefile containing network here
point_id_field = "STAT_BLOCK"                  # Select the identifier variable in your location shapefile here
walking_speed_kmh = 4                          # Select the speed off the network (e.g. walking) here
network_speed_kmh = 35                         # Select the speed on the network (e.g. subways) here
output_matrix_file = "TTMATRIX-final.csv"      # Select the name of the outcome travel time matrix here
output_shapefile = "ATT-final.shp"             # Select the name of the outcome shapefile with average travel times here

# === PACKAGE INSTALLATION ===
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["geopandas", "shapely", "tqdm", "matplotlib", "networkx", "pandas", "numpy", "pyproj"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# === IMPORTS ===
import geopandas as gpd
from shapely.geometry import LineString, Point
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from pyproj import CRS

# === SET PATHS ===
input_dir = os.path.join(working_dir, "input")
output_dir = os.path.join(working_dir, "output")
os.makedirs(output_dir, exist_ok=True)

points_path = os.path.join(input_dir, points_file)
stations_path = os.path.join(input_dir, stations_file)
network_path = os.path.join(input_dir, network_file)

# === LOAD DATA ===
points = gpd.read_file(points_path)

# === LIMIT TO FIRST 500 POINTS FOR DEBUGGING ===
# max_points = 500
# points = points.iloc[:max_points].copy()

stations = gpd.read_file(stations_path)
network = gpd.read_file(network_path)

print(f"Loaded {len(points)} points")
print(f"Loaded {len(stations)} stations")
print(f"Loaded {len(network)} network elements")

# === CHECK & ALIGN CRS ===
if not points.crs.is_projected:
    print(f"Input CRS: {points.crs}")
    print("Points file is in geographic coordinates. Reprojecting to a local UTM CRS...")

    centroid = points.geometry.union_all().centroid

    # Compute UTM zone manually
    zone_number = int((centroid.x + 180) / 6) + 1
    is_northern = centroid.y >= 0
    epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number
    best_utm_crs = CRS.from_epsg(epsg_code)

    print(f"Reprojecting to UTM zone {zone_number}, EPSG:{epsg_code}")

    points = points.to_crs(best_utm_crs)
    stations = stations.to_crs(best_utm_crs)
    network = network.to_crs(best_utm_crs)
else:
    stations = stations.to_crs(points.crs)
    network = network.to_crs(points.crs)

# === CENTROID CONVERSION IF NEEDED ===
if points.geom_type.isin(["Polygon", "MultiPolygon"]).any():
    print("Converting polygons to centroids...")
    points["geometry"] = points.centroid

# === BUILD AUGMENTED GRAPH ===
print("Building augmented graph with transit + walking...")

G_aug = nx.Graph()

# Add transit network edges
for idx, row in network.iterrows():
    coords = list(row.geometry.coords)
    for i in range(len(coords) - 1):
        u, v = coords[i], coords[i + 1]
        segment = LineString([u, v]).length
        time = (segment / 1000) / network_speed_kmh * 60
        G_aug.add_edge(u, v, weight=time)

# Add station nodes
for idx, row in stations.iterrows():
    G_aug.add_node(f"station_{idx}", geometry=row.geometry)

# === CONNECT STATIONS TO TRANSIT NETWORK ===
print("Connecting stations to nearest transit network node...")

network_nodes = [n for n in G_aug.nodes if isinstance(n, tuple)]
network_node_points = [Point(n) for n in network_nodes]

def get_nearest_network_node(station_geom):
    distances = [station_geom.distance(pt) for pt in network_node_points]
    return network_nodes[np.argmin(distances)]

for idx, row in tqdm(stations.iterrows(), total=len(stations), desc="Snapping stations"):
    station_name = f"station_{idx}"
    nearest_node = get_nearest_network_node(row.geometry)
    G_aug.add_edge(station_name, nearest_node, weight=0.0001)

# Add point nodes
for idx, row in points.iterrows():
    G_aug.add_node(f"point_{idx}", geometry=row.geometry)

# Add access edges from each point to each station
for i, point in tqdm(points.iterrows(), total=len(points), desc="Access edges → graph"):
    p_node = f"point_{i}"
    for j, station in stations.iterrows():
        s_node = f"station_{j}"
        dist = point.geometry.distance(station.geometry)
        time = (dist / 1000) / walking_speed_kmh * 60
        G_aug.add_edge(p_node, s_node, weight=time)

# === ADD WALKING EDGES BETWEEN ALL POINTS ===
print("Adding walking edges between all points...")

for i, point_i in tqdm(points.iterrows(), total=len(points), desc="Point-to-point edges"):
    p_node_i = f"point_{i}"
    geom_i = point_i.geometry
    for j, point_j in points.iterrows():
        if i >= j:
            continue
        p_node_j = f"point_{j}"
        geom_j = point_j.geometry
        distance_m = geom_i.distance(geom_j)
        time_min = (distance_m / 1000) / walking_speed_kmh * 60
        G_aug.add_edge(p_node_i, p_node_j, weight=time_min)

# === COMPUTE TRAVEL TIME MATRIX (SERIAL) ===
print("Computing travel time matrix (serial)...")
matrix = pd.DataFrame(index=points.index, columns=points.index)

for i in tqdm(points.index, desc="Dijkstra"):
    source = f"point_{i}"
    lengths = nx.single_source_dijkstra_path_length(G_aug, source, weight='weight')
    for j in points.index:
        target = f"point_{j}"
        matrix.at[i, j] = lengths.get(target, np.nan)

# === ASSIGN ID LABELS WITH PREFIX TO MATRIX ===
if point_id_field not in points.columns:
    raise ValueError(f"ID field '{point_id_field}' not found in points file.")

id_labels = points[point_id_field].astype(str).values
row_labels = [point_id_field + str(val) for val in id_labels]
matrix.index = row_labels
matrix.columns = row_labels

# === SAVE MATRIX TO CSV ===
output_csv = os.path.join(output_dir, output_matrix_file)
matrix.to_csv(output_csv, index_label=point_id_field)
print(f"Saved matrix to: {output_csv}")

# === COMPUTE MEAN TRAVEL TIME ===
mean_travel_times = matrix.mean(axis=1, skipna=True)
points["mean_time_min"] = pd.to_numeric(mean_travel_times.values, errors='coerce').astype("float64")

# === SAVE POINTS WITH MEAN TIME ===
points_out_path = os.path.join(output_dir, output_shapefile)
points.to_file(points_out_path)
print(f"Saved enriched points with mean travel times to: {points_out_path}")

# === PLOT MEAN TRAVEL TIME MAP ===
fig, ax = plt.subplots(figsize=(10, 10))
points.plot(
    column="mean_time_min",
    ax=ax,
    legend=True,
    cmap="viridis",
    markersize=60,
    edgecolor="black",
    linewidth=0.2
)
plt.title("Mean Travel Time from Each Origin (minutes)")
plt.tight_layout()
plt.show()

# === STATISTICS FOR MEAN TRAVEL TIMES ===
print("\nMean travel time statistics (in minutes):")
print(f"Mean: {points['mean_time_min'].mean():.2f}")
print(f"Min:  {points['mean_time_min'].min():.2f}")
print(f"Max:  {points['mean_time_min'].max():.2f}")
