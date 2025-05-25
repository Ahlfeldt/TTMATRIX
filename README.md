# TTMATRIX
Toolkit for computing simple public transit travel time matrices
(c) Gabriel M. Ahlfeldt, version 0.1.0., 2025-05

# About

This repository provides a ready-to-use `Python` toolkit for computing simple **public transport travel time matrices**. It is designed to be **fast**, **convenient**, and **minimalist**, requiring only three input shapefiles:

1. **Points** to be connected (e.g. centroids of spatial units),
2. **Public transport stations** (e.g. subway stops), and
3. **Network geometry** (e.g. subway lines).

## Key Features

- **Customizable speeds**: Choose walking and transit speeds (e.g. 4 km/h and 35 km/h).
- **Smart routing**: Computes least-time routes for all origin-destination pairs, using the network **only if faster** than walking.
- **Flexible station choice**: No fixed assignment to nearest station — the toolkit chooses the most time-efficient entry and exit.
- **Automatic snapping and connections**: Stations are snapped to the network, and all points are connected to all stations automatically.
- **Output**: 
  - Travel time matrix (`.csv`)
  - Enriched shapefile with mean travel times per point
  - Optional visualization of travel time map

## Getting Started

See the script `TTMATRIX.py` for usage. All dependencies are installed automatically if missing. 

# === USER SETTINGS ===
Folder | File  | Description |
|:------------------------|:-----------------------|
working_dir = r"A:\Research\TTMATRIX-toolkit" | Set your working directory by replacing `A:\Research\TTMATRIX-toolkit` with the correct path |          
points_file = "B4m_com_ll.shp"
stations_file = "UBahn2020_stops_ll.shp"
network_file = "UBahn2020_lines_ll.shp"
point_id_field = "STAT_BLOCK"
walking_speed_kmh = 4
network_speed_kmh = 35
output_matrix_file = "TTMATRIX-final.csv"
output_shapefile = "ATT-final.shp"


## Example Output

![Example Map](example_map.png) <!-- Optional: Replace with your own example image -->

## Methodology: Routing and Graph Construction

The travel time matrix is computed by constructing an **augmented graph** that combines the **public transport network** with **walking access**. The graph is undirected and weighted by travel times in minutes. The following nodes and edges are included:

### 1. Transit Network Edges
- Each transit line (e.g., a subway line) is decomposed into segments between consecutive coordinates.
- Each segment becomes an edge with a weight equal to the travel time:
time = (segment length in km) / (network speed in km/h) × 60

### 2. Station Nodes and Snapping Edges
- Each public transport station is added as a node.
- Each station is connected to its **nearest point** on the network via a **nearly zero-cost edge** (~0.0001 minutes) to allow entry and exit from the network.

### 3. Point Nodes and Access Edges
- Each point to be included in the travel time matrix is added as a node.
- Each point is connected to **every station** by a walking edge, weighted by:
time = (Euclidean distance in km) / (walking speed in km/h) × 60


- This design allows the algorithm to determine the most efficient **entry and exit** stations for each journey — not just the nearest ones.

### 4. Direct Walking Edges Between Points
- To allow for walking-only trips, a walking edge is also added between every unique pair of points.
- These are calculated the same way using walking speed and straight-line distance.

### 5. Shortest Path Computation
- The resulting graph allows both multimodal and walking-only routes.
- For each origin point, the shortest paths to all other points are computed using **Dijkstra’s algorithm**, weighted by travel time.
- The final travel time between any two points reflects the **least-cost route**, whether via transit or direct walking.

### Output
- A complete origin-destination travel time matrix (`.csv`)
- A shapefile enriched with each point’s **mean travel time**
- An optional map visualization of accessibility patterns





---

*Developed for quick and practical applications in spatial analysis and urban transport research.*
