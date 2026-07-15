#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "cartopy>=0.25",
#   "matplotlib>=3.9",
#   "netCDF4>=1.7",
#   "numpy>=2.0",
#   "rasterio>=1.4",
#   "xarray>=2025.1.0",
# ]
# ///

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr


NET_AXES_POSITIONS = {
    (-1, 0): [0.00, 0.333333, 0.25, 0.333333],
    (0, 0): [0.25, 0.333333, 0.25, 0.333333],
    (1, 0): [0.50, 0.333333, 0.25, 0.333333],
    (2, 0): [0.75, 0.333333, 0.25, 0.333333],
    (0, 1): [0.25, 0.666666, 0.25, 0.333333],
    (0, -1): [0.25, 0.00, 0.25, 0.333333],
}

DISPLAY_TRANSFORM_EDGE_MAPS = {
    "m0r0": {
        "top": ("top", False),
        "right": ("right", False),
        "bottom": ("bottom", False),
        "left": ("left", False),
    },
    "m0r1": {
        "top": ("left", True),
        "right": ("top", False),
        "bottom": ("right", True),
        "left": ("bottom", False),
    },
    "m0r2": {
        "top": ("bottom", True),
        "right": ("left", True),
        "bottom": ("top", True),
        "left": ("right", True),
    },
    "m0r3": {
        "top": ("right", False),
        "right": ("bottom", True),
        "bottom": ("left", False),
        "left": ("top", True),
    },
    "m1r0": {
        "top": ("top", True),
        "right": ("left", False),
        "bottom": ("bottom", True),
        "left": ("right", False),
    },
    "m1r1": {
        "top": ("right", True),
        "right": ("top", True),
        "bottom": ("left", True),
        "left": ("bottom", True),
    },
    "m1r2": {
        "top": ("bottom", False),
        "right": ("right", True),
        "bottom": ("top", False),
        "left": ("left", True),
    },
    "m1r3": {
        "top": ("left", False),
        "right": ("bottom", False),
        "bottom": ("right", False),
        "left": ("top", False),
    },
}

BASE_ARROW_ORIGIN = np.array([0.12, 0.86])
BASE_U_ENDPOINT = np.array([0.30, 0.86])
BASE_V_ENDPOINT = np.array([0.12, 0.66])

EARTH_IMAGE = None
EARTH_RASTER = None

BLUE_MARBLE_URL = (
    "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/"
    "bmng/bmng-topography-bathymetry/august/"
    "world.topo.bathy.200408.3x5400x2700.jpg"
)
BLUE_MARBLE_CACHE_DIR = Path.home() / ".cache" / "cubeplot"
BLUE_MARBLE_CACHE_PATH = BLUE_MARBLE_CACHE_DIR / "blue_marble_5400x2700.jpg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a cubed-sphere dataset as a flat cube net."
    )
    parser.add_argument("dataset", help="Path to the cubed-sphere NetCDF file")
    parser.add_argument(
        "--variable",
        default="SLP",
        help="Variable to plot (default: SLP)",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to plot when the variable has a time dimension",
    )
    parser.add_argument(
        "--level-index",
        type=int,
        default=0,
        help="Level index to plot when the variable has a lev dimension",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo)",
    )
    parser.add_argument(
        "--background",
        choices=("earth", "data"),
        default="earth",
        help="Use a shaded Earth basemap or a dataset variable fill (default: earth)",
    )
    parser.add_argument(
        "--earth-image",
        help=(
            "Optional path to a global Earth raster, or 'blue-marble' to download "
            "and cache NASA Blue Marble automatically"
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path for saving the figure instead of showing it",
    )
    return parser.parse_args()


def wrap_relative(lon, center):
    return ((lon - center + 180.0) % 360.0) - 180.0 + center


def infer_face_center(face_lons, face_lats):
    lon_radians = np.deg2rad(face_lons)
    lat_radians = np.deg2rad(face_lats)

    x_coord = np.cos(lat_radians) * np.cos(lon_radians)
    y_coord = np.cos(lat_radians) * np.sin(lon_radians)
    z_coord = np.sin(lat_radians)

    center = np.array([x_coord.mean(), y_coord.mean(), z_coord.mean()])
    center /= np.linalg.norm(center)

    center_lon = np.rad2deg(np.arctan2(center[1], center[0])) % 360.0
    center_lat = np.rad2deg(np.arcsin(center[2]))

    return float(center_lon), float(center_lat)


def build_edge_vectors(corner_lons, corner_lats):
    lon_radians = np.deg2rad(corner_lons)
    lat_radians = np.deg2rad(corner_lats)

    x_coord = np.cos(lat_radians) * np.cos(lon_radians)
    y_coord = np.cos(lat_radians) * np.sin(lon_radians)
    z_coord = np.sin(lat_radians)
    vectors = np.stack([x_coord, y_coord, z_coord], axis=-1)

    edges = {}
    for face_index in range(vectors.shape[0]):
        face = face_index + 1
        face_vectors = vectors[face_index]
        edges[(face, "top")] = face_vectors[0, :, :]
        edges[(face, "bottom")] = face_vectors[-1, :, :]
        edges[(face, "left")] = face_vectors[:, 0, :]
        edges[(face, "right")] = face_vectors[:, -1, :]

    return edges


def build_edge_matches(corner_lons, corner_lats):
    edges = build_edge_vectors(corner_lons, corner_lats)
    matches = {}

    for edge_key, edge_values in edges.items():
        best_score = None
        best_match = None

        for other_key, other_values in edges.items():
            if edge_key[0] == other_key[0]:
                continue

            forward = np.max(np.linalg.norm(edge_values - other_values, axis=1))
            reverse = np.max(np.linalg.norm(edge_values - other_values[::-1], axis=1))
            score = min(forward, reverse)

            if best_score is None or score < best_score:
                best_score = score
                best_match = (other_key[0], other_key[1], reverse < forward)

        matches[edge_key] = best_match

    return matches


def opposite_side(side_name):
    return {
        "top": "bottom",
        "right": "left",
        "bottom": "top",
        "left": "right",
    }[side_name]


def build_face_positions(edge_matches, center_face=1):
    left_face = edge_matches[(center_face, "left")][0]
    right_face = edge_matches[(center_face, "right")][0]
    top_face = edge_matches[(center_face, "bottom")][0]
    bottom_face = edge_matches[(center_face, "top")][0]

    shared_edge_on_right_face = edge_matches[(center_face, "right")][1]
    far_right_face = edge_matches[(right_face, opposite_side(shared_edge_on_right_face))][0]

    return {
        left_face: (-1, 0),
        center_face: (0, 0),
        right_face: (1, 0),
        far_right_face: (2, 0),
        top_face: (0, 1),
        bottom_face: (0, -1),
    }


def build_adjacencies(face_positions):
    adjacencies = []
    for face_a, position_a in face_positions.items():
        for face_b, position_b in face_positions.items():
            if face_a >= face_b:
                continue

            if abs(position_a[0] - position_b[0]) + abs(position_a[1] - position_b[1]) != 1:
                continue

            if position_b[0] == position_a[0] + 1:
                adjacencies.append((face_a, "right", face_b, "left"))
            elif position_b[0] == position_a[0] - 1:
                adjacencies.append((face_a, "left", face_b, "right"))
            elif position_b[1] == position_a[1] + 1:
                adjacencies.append((face_a, "top", face_b, "bottom"))
            elif position_b[1] == position_a[1] - 1:
                adjacencies.append((face_a, "bottom", face_b, "top"))

    return adjacencies


def solve_face_rotations(edge_matches, face_positions):
    adjacencies = build_adjacencies(face_positions)

    transform_names = tuple(DISPLAY_TRANSFORM_EDGE_MAPS)

    for transform_indices in np.ndindex(*(len(transform_names),) * len(face_positions)):
        rotations = {
            face: transform_names[transform_indices[face - 1]]
            for face in sorted(face_positions)
        }

        valid = True
        for face_a, side_a, face_b, side_b in adjacencies:
            edge_a, reverse_a = DISPLAY_TRANSFORM_EDGE_MAPS[rotations[face_a]][side_a]
            edge_b, reverse_b = DISPLAY_TRANSFORM_EDGE_MAPS[rotations[face_b]][side_b]
            match_face, match_edge, match_reversed = edge_matches[(face_a, edge_a)]

            if match_face != face_b or match_edge != edge_b:
                valid = False
                break

            if (reverse_a ^ reverse_b) != match_reversed:
                valid = False
                break

        if valid:
            return rotations

    raise RuntimeError("Unable to solve face rotations for the cube net")


def select_face_field(raw, variable_name, time_index, level_index):
    field = raw[variable_name]

    if "time" in field.dims:
        field = field.isel(time=time_index)
    if "lev" in field.dims:
        field = field.isel(lev=level_index)

    if tuple(field.dims) != ("nf", "Ydim", "Xdim"):
        raise ValueError(
            f"{variable_name} must reduce to ('nf', 'Ydim', 'Xdim'); got {field.dims}"
        )

    return field.values


def load_earth_image():
    global EARTH_IMAGE

    if EARTH_IMAGE is None:
        cartopy_root = Path(cartopy.__file__).resolve().parent
        earth_image_path = (
            cartopy_root
            / "data"
            / "raster"
            / "natural_earth"
            / "50-natural-earth-1-downsampled.png"
        )
        EARTH_IMAGE = mpimg.imread(earth_image_path)[..., :3]

    return EARTH_IMAGE


def load_earth_raster(earth_image_path):
    global EARTH_RASTER

    if earth_image_path == "blue-marble":
        earth_image_path = ensure_blue_marble_raster()

    raster_path = Path(earth_image_path).expanduser().resolve()
    if EARTH_RASTER is not None and EARTH_RASTER["path"] == str(raster_path):
        return EARTH_RASTER

    suffix = raster_path.suffix.lower()
    if suffix in {".tif", ".tiff", ".jp2"}:
        with rasterio.open(raster_path) as dataset:
            image = dataset.read()
            transform = dataset.transform
            crs = dataset.crs

        if image.shape[0] < 3:
            raise ValueError(f"Earth raster must have at least 3 bands: {raster_path}")

        EARTH_RASTER = {
            "path": str(raster_path),
            "mode": "georeferenced",
            "image": np.moveaxis(image[:3], 0, -1),
            "transform": transform,
            "crs": crs,
        }
        return EARTH_RASTER

    EARTH_RASTER = {
        "path": str(raster_path),
        "mode": "plain",
        "image": mpimg.imread(raster_path)[..., :3],
    }
    return EARTH_RASTER


def ensure_blue_marble_raster():
    BLUE_MARBLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not BLUE_MARBLE_CACHE_PATH.exists():
        urlretrieve(BLUE_MARBLE_URL, BLUE_MARBLE_CACHE_PATH)
    return BLUE_MARBLE_CACHE_PATH


def normalize_raster_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value > 1.0:
            image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def sample_plain_earth_image(earth_image, face_lons, face_lats):
    height, width = earth_image.shape[:2]
    # Cartopy's bundled Natural Earth raster is stored on a [-180, 180] grid,
    # so shift longitudes before converting them to pixel columns.
    wrapped_lons = np.mod(face_lons + 180.0, 360.0)
    clipped_lats = np.clip(face_lats, -90.0, 90.0)

    x_index = np.rint((wrapped_lons / 360.0) * (width - 1)).astype(int)
    y_index = np.rint(((90.0 - clipped_lats) / 180.0) * (height - 1)).astype(int)

    return earth_image[y_index, x_index]


def sample_georeferenced_earth_image(raster_info, face_lons, face_lats):
    if raster_info["crs"] is None:
        return sample_plain_earth_image(raster_info["image"], face_lons, face_lats)

    if raster_info["crs"].to_epsg() != 4326:
        raise ValueError(
            f"Only EPSG:4326 Earth rasters are supported right now, got {raster_info['crs']}"
        )

    image = normalize_raster_image(raster_info["image"])
    transform = raster_info["transform"]
    inverse_transform = ~transform

    sample_lons = np.where(face_lons > 180.0, face_lons - 360.0, face_lons)
    pixel_cols, pixel_rows = inverse_transform * (sample_lons, face_lats)
    pixel_rows = np.rint(pixel_rows).astype(int)
    pixel_cols = np.rint(pixel_cols).astype(int)

    pixel_rows = np.clip(pixel_rows, 0, image.shape[0] - 1)
    pixel_cols = np.clip(pixel_cols, 0, image.shape[1] - 1)

    return image[pixel_rows, pixel_cols]


def sample_earth_image(earth_source, face_lons, face_lats):
    if isinstance(earth_source, dict):
        if earth_source["mode"] == "georeferenced":
            return sample_georeferenced_earth_image(earth_source, face_lons, face_lats)
        return sample_plain_earth_image(
            normalize_raster_image(earth_source["image"]),
            face_lons,
            face_lats,
        )

    return sample_plain_earth_image(normalize_raster_image(earth_source), face_lons, face_lats)


def draw_face_background(ax, square_x, square_y, face_lons, face_lats, face_field, cmap, background):
    if background == "data":
        ax.pcolormesh(
            square_x,
            square_y,
            face_field,
            shading="auto",
            cmap=cmap,
        )
        return

    ax.pcolormesh(
        square_x,
        square_y,
        face_field,
        shading="flat",
    )


def build_square_transform(corner_lons, corner_lats, face_center, plate_carree):
    projection = ccrs.Gnomonic(
        central_longitude=face_center[0],
        central_latitude=face_center[1],
    )

    wrapped_lons = wrap_relative(corner_lons, face_center[0])
    projected = projection.transform_points(plate_carree, wrapped_lons, corner_lats)
    projected_xy = projected[..., :2]

    origin = projected_xy[0, 0]
    u_basis = projected_xy[0, -1] - origin
    v_basis = projected_xy[-1, 0] - origin
    inverse_basis = np.linalg.inv(np.column_stack([u_basis, v_basis]))

    square = (projected_xy - origin) @ inverse_basis.T
    square[..., 1] = 1.0 - square[..., 1]

    return projection, origin, inverse_basis, square[..., 0], square[..., 1]


def apply_display_transform(x_coords, y_coords, transform_name):
    if transform_name == "m0r0":
        return x_coords, y_coords
    if transform_name == "m0r1":
        return y_coords, 1.0 - x_coords
    if transform_name == "m0r2":
        return 1.0 - x_coords, 1.0 - y_coords
    if transform_name == "m0r3":
        return 1.0 - y_coords, x_coords
    if transform_name == "m1r0":
        return 1.0 - x_coords, y_coords
    if transform_name == "m1r1":
        return y_coords, x_coords
    if transform_name == "m1r2":
        return x_coords, 1.0 - y_coords
    if transform_name == "m1r3":
        return 1.0 - y_coords, 1.0 - x_coords
    raise ValueError(f"Unsupported display transform: {transform_name}")


def extract_lines(geometry):
    if geometry.is_empty:
        return

    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from extract_lines(part)
        return

    if hasattr(geometry, "coords"):
        coords = np.asarray(geometry.coords)
        if coords.size:
            yield coords


def transform_line_to_square(coords, projection, origin, inverse_basis, face_center, plate_carree):
    wrapped_lons = wrap_relative(coords[:, 0], face_center[0])
    projected = projection.transform_points(plate_carree, wrapped_lons, coords[:, 1])
    projected_xy = projected[:, :2]

    square = (projected_xy - origin) @ inverse_basis.T
    square[:, 1] = 1.0 - square[:, 1]

    valid = np.all(np.isfinite(square), axis=1)
    if not np.any(valid):
        return []

    square = square[valid]
    inside = (
        (square[:, 0] >= -0.05)
        & (square[:, 0] <= 1.05)
        & (square[:, 1] >= -0.05)
        & (square[:, 1] <= 1.05)
    )
    if not np.any(inside):
        return []

    square = square[inside]
    jumps = np.sqrt(np.sum(np.diff(square, axis=0) ** 2, axis=1))
    split_points = np.where(jumps > 0.2)[0] + 1

    segments = []
    start = 0
    for stop in split_points:
        if stop - start >= 2:
            segments.append(square[start:stop])
        start = stop
    if len(square) - start >= 2:
        segments.append(square[start:])

    return segments


def draw_uv_axes(ax, display_transform):
    origin_x, origin_y = apply_display_transform(
        BASE_ARROW_ORIGIN[0],
        BASE_ARROW_ORIGIN[1],
        display_transform,
    )
    u_end_x, u_end_y = apply_display_transform(
        BASE_U_ENDPOINT[0],
        BASE_U_ENDPOINT[1],
        display_transform,
    )
    v_end_x, v_end_y = apply_display_transform(
        BASE_V_ENDPOINT[0],
        BASE_V_ENDPOINT[1],
        display_transform,
    )
    origin = (float(origin_x), float(origin_y))
    u_vector = (float(u_end_x - origin_x), float(u_end_y - origin_y))
    v_vector = (float(v_end_x - origin_x), float(v_end_y - origin_y))

    ax.annotate(
        "",
        xy=(origin[0] + u_vector[0], origin[1] + u_vector[1]),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops=dict(color="white", width=1.5, headwidth=10, headlength=10),
    )
    ax.annotate(
        "",
        xy=(origin[0] + v_vector[0], origin[1] + v_vector[1]),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops=dict(color="white", width=1.5, headwidth=10, headlength=10),
    )

    ax.text(
        origin[0] + u_vector[0] * 1.05,
        origin[1] + u_vector[1] * 1.05,
        "u",
        transform=ax.transAxes,
        color="white",
        fontsize=18,
        fontstyle="italic",
        weight="bold",
    )
    ax.text(
        origin[0] + v_vector[0] * 1.05,
        origin[1] + v_vector[1] * 1.05,
        "v",
        transform=ax.transAxes,
        color="white",
        fontsize=18,
        fontstyle="italic",
        weight="bold",
    )


def plot_cube_net(
    dataset_path,
    variable_name=None,
    time_index=0,
    level_index=0,
    cmap="turbo",
    background="earth",
    earth_image_path=None,
):
    raw = xr.open_dataset(dataset_path)
    face_field = None
    if background == "data":
        if variable_name is None:
            raise ValueError("variable_name is required when background='data'")
        face_field = select_face_field(raw, variable_name, time_index, level_index)
    face_lons = raw["lons"].values
    face_lats = raw["lats"].values
    corner_lons = raw["corner_lons"].values
    corner_lats = raw["corner_lats"].values
    edge_matches = build_edge_matches(corner_lons, corner_lats)
    face_positions = build_face_positions(edge_matches)
    face_rotations = solve_face_rotations(edge_matches, face_positions)
    plate_carree = ccrs.PlateCarree()
    coastline_path = shpreader.natural_earth(
        resolution="110m",
        category="physical",
        name="coastline",
    )
    coastlines = list(shpreader.Reader(coastline_path).geometries())
    earth_source = None
    if background == "earth":
        earth_source = (
            load_earth_raster(earth_image_path)
            if earth_image_path is not None
            else load_earth_image()
        )

    fig = plt.figure(figsize=(16, 12), facecolor="black")

    for face, net_position in sorted(face_positions.items(), key=lambda item: (item[1][1], item[1][0])):
        position = NET_AXES_POSITIONS[net_position]
        face_index = face - 1
        face_center = infer_face_center(face_lons[face_index], face_lats[face_index])
        projection, origin, inverse_basis, square_x, square_y = build_square_transform(
            corner_lons[face_index],
            corner_lats[face_index],
            face_center,
            plate_carree,
        )
        ax = fig.add_axes(position, facecolor="black")
        square_x, square_y = apply_display_transform(square_x, square_y, face_rotations[face])
        if background == "data":
            face_background = face_field[face_index]
        else:
            face_background = sample_earth_image(
                earth_source,
                face_lons[face_index],
                face_lats[face_index],
            )

        draw_face_background(
            ax,
            square_x,
            square_y,
            face_lons[face_index],
            face_lats[face_index],
            face_background,
            cmap,
            background,
        )

        for row in range(square_x.shape[0]):
            ax.plot(
                square_x[row, :],
                square_y[row, :],
                color="black",
                linewidth=0.35,
            )
        for col in range(square_x.shape[1]):
            ax.plot(
                square_x[:, col],
                square_y[:, col],
                color="black",
                linewidth=0.35,
            )

        for coastline in coastlines:
            for coords in extract_lines(coastline):
                for segment in transform_line_to_square(
                    coords,
                    projection,
                    origin,
                    inverse_basis,
                    face_center,
                    plate_carree,
                ):
                    segment_x, segment_y = apply_display_transform(
                        segment[:, 0],
                        segment[:, 1],
                        face_rotations[face],
                    )
                    ax.plot(segment_x, segment_y, color="black", linewidth=0.6)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.text(
            0.08,
            0.88,
            str(face),
            transform=ax.transAxes,
            color="white",
            fontsize=28,
            weight="bold",
        )
        draw_uv_axes(ax, face_rotations[face])

    raw.close()
    return fig


def main():
    args = parse_args()
    fig = plot_cube_net(
        dataset_path=args.dataset,
        variable_name=args.variable,
        time_index=args.time_index,
        level_index=args.level_index,
        cmap=args.cmap,
        background=args.background,
        earth_image_path=args.earth_image,
    )

    if args.output:
        fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
