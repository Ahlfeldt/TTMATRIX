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

## Example Output

![Example Map](example_map.png) <!-- Optional: Replace with your own example image -->

---

*Developed for quick and practical applications in spatial analysis and urban transport research.*
