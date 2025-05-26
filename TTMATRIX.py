import os
import subprocess
import sys

# === USER SETTINGS ===
working_dir = r"H:\Research\TTMATRIX-toolkit"       # Set your working directory
points_file = "B4m_com_ll.shp"                      # Point shapefile (origins/destinations)
stations_file = "UBahn2020noU5ext_stops_ll.shp"            # Station shapefile (entry/exit points)
network_file = "UBahn2020noU5ext_lines_ll.shp"             # Network polyline shapefile
point_id_field = "STAT_BLOCK"                       # Identifier field in point shapefile
walking_speed_kmh = 4                               # Walking speed (km/h)
network_speed_kmh = 35                              # Network speed (km/h)
snap_tolerance_m = 1.0                              # Tolerance for snapping network segment endpoints (meters)
output_matrix_file = "TTMATRIX-noU5ext.csv"           # Output travel time matrix CSV
output_shapefile = "ATT-noU5ext.shp"                  # Output shapefile with average travel times
output_edges_shapefile = "graph_edges-noU5ext.shp"    # Output shapefile showing the graph (network + walking) used in Dijkstra

# === PACKAGE INSTALLATION ===
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["geopandas", "shapely", "tqdm", "matplotlib", "networkx", "pandas", "numpy", "pyproj", "scipy"]:
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
from scipy.spatial import cKDTree

# === SET PATHS ===
input_dir = os.path.join(working_dir, "input")
output_dir = os.path.join(working_dir, "output")
os.makedirs(output_dir, exist_ok=True)

points_path = os.path.join(input_dir, points_file)
stations_path = os.path.join(input_dir, stations_file)
network_path = os.path.join(input_dir, network_file)

# === LOAD DATA ===
points = gpd.read_file(points_path)
# === OPTIONAL: LIMIT TO FIRST 1000 POINTS FOR DEBUGGING ===
# points = points.iloc[:1000].copy()
stations = gpd.read_file(stations_path)
network = gpd.read_file(network_path)

print(f"Loaded {len(points)} points")
print(f"Loaded {len(stations)} stations")
print(f"Loaded {len(network)} network elements")

# === CHECK & ALIGN CRS ===
if not points.crs.is_projected:
    print(f"Input CRS: {points.crs}")
    print("Points file is in geographic coordinates. Reprojecting to a local UTM CRS...")

    centroid = points.geometry.unary_union.centroid
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

# === SNAP NEARBY ENDPOINTS IN NETWORK ===
if snap_tolerance_m > 0:
    print(f"Snapping nearby network segment endpoints within {snap_tolerance_m} meter(s)...")

    endpoints = []
    for geom in network.geometry:
        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints.append(Point(coords[0]))
            endpoints.append(Point(coords[-1]))

    endpoint_coords = np.array([[pt.x, pt.y] for pt in endpoints])
    endpoint_kdtree = cKDTree(endpoint_coords)
    snapped_coords = endpoint_coords.copy()
    visited = set()

    for i in range(len(endpoint_coords)):
        if i in visited:
            continue
        idxs = endpoint_kdtree.query_ball_point(endpoint_coords[i], r=snap_tolerance_m)
        if len(idxs) > 1:
            visited.update(idxs)
            cluster_pts = endpoint_coords[idxs]
            centroid = cluster_pts.mean(axis=0)
            for idx in idxs:
                snapped_coords[idx] = centroid

    coord_map = {tuple(pt): tuple(snapped_coords[i]) for i, pt in enumerate(endpoint_coords)}

    def snap_coords(coords):
        return [coord_map.get(tuple(c), c) for c in coords]

    snapped_geoms = []
    for geom in network.geometry:
        coords = list(geom.coords)
        new_coords = snap_coords(coords)
        snapped_geoms.append(LineString(new_coords))

    network["geometry"] = snapped_geoms
    print("Finished snapping network endpoints.")

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
network_kdtree = cKDTree(np.array([[pt.x, pt.y] for pt in network_node_points]))

for idx, row in tqdm(stations.iterrows(), total=len(stations), desc="Snapping stations"):
    station_name = f"station_{idx}"
    station_coord = np.array([row.geometry.x, row.geometry.y])
    _, nearest_idx = network_kdtree.query(station_coord, k=1)
    nearest_node = network_nodes[nearest_idx]
    G_aug.add_edge(station_name, nearest_node, weight=0.0001)

# Add point nodes
for idx, row in points.iterrows():
    G_aug.add_node(f"point_{idx}", geometry=row.geometry)

# === CONNECT POINTS TO NEAREST STATIONS ===
print("Adding walking edges from points to their 3 nearest stations...")

station_coords = np.array([[geom.x, geom.y] for geom in stations.geometry])
station_kdtree = cKDTree(station_coords)

point_coords = np.array([[geom.x, geom.y] for geom in points.geometry])

for i in tqdm(range(len(points)), desc="Point-to-station edges"):
    distances, indices = station_kdtree.query(point_coords[i], k=3)
    p_node = f"point_{i}"
    for dist_m, j in zip(distances, indices):
        s_node = f"station_{j}"
        time_min = (dist_m / 1000) / walking_speed_kmh * 60
        G_aug.add_edge(p_node, s_node, weight=time_min)

# === ADD WALKING EDGES TO NEAREST NEIGHBORS ONLY ===
print("Adding walking edges to 5 nearest neighbors per point...")

point_kdtree = cKDTree(point_coords)

for i in tqdm(range(len(points)), desc="Point-to-point nearest neighbors"):
    distances, neighbors = point_kdtree.query(point_coords[i], k=6)  # k=6 to include self
    p_node_i = f"point_{i}"
    for neighbor_idx, distance_m in zip(neighbors[1:], distances[1:]):  # skip self
        p_node_j = f"point_{neighbor_idx}"
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

# === EXPORT ALL GRAPH EDGES AS SHAPEFILE (INCLUDING POINT & STATION LINKS) ===
print("Exporting full graph edges (network + walking) as shapefile...")

edge_records = []

for u, v, data in G_aug.edges(data=True):
    try:
        geom_u = Point(u) if isinstance(u, tuple) else G_aug.nodes[u]["geometry"]
        geom_v = Point(v) if isinstance(v, tuple) else G_aug.nodes[v]["geometry"]
        line = LineString([geom_u, geom_v])
        edge_records.append({
            "from_node": str(u),
            "to_node": str(v),
            "time_min": data["weight"],
            "geometry": line
        })
    except Exception as e:
        print(f"Skipped edge ({u}, {v}): {e}")

edges_gdf = gpd.GeoDataFrame(edge_records, crs=points.crs)
edges_out_path = os.path.join(output_dir, output_edges_shapefile)
edges_gdf.to_file(edges_out_path)
print(f"Saved graph edges to: {edges_out_path}")

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
